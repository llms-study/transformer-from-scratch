import warnings
from pathlib import Path

import torch
import torch.nn as nn
from datasets import load_dataset
from fsspec.utils import tokenize
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import Whitespace
from tokenizers.trainers import WordLevelTrainer
from torch.utils.data import DataLoader, Dataset, Subset, random_split
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from config import get_config, get_weights_file_path
from dataset import BilingualDataSet, causal_mask
from transformer import build_transformer


def get_all_sentences(ds, lang):
    for item in ds:
        yield item["translation"][lang]


def get_or_build_tokenizer(config, ds, lang):
    tokenizer_path = Path(config["tokenizer_file"].format(lang))
    if not tokenizer_path.exists():
        tokenizer = Tokenizer(WordLevel(unk_token="[UNK]"))
        tokenizer.pre_tokenizer = Whitespace()
        trainer = WordLevelTrainer(
            min_frequency=2,
            special_tokens=[
                "[UNK]",  # Unknown
                "[PAD]",  # for padding
                "[SOS]",  # start of sentence
                "[EOS]",  # end of sentence
            ],
        )
        tokenizer.train_from_iterator(get_all_sentences(ds, lang), trainer=trainer)
    else:
        tokenizer = Tokenizer.from_file(str(tokenizer_path))
    return tokenizer


def get_ds(config):
    ds_raw = load_dataset(
        path=f"{config['datasource']}",
        name=f"{config['lang_src']}-{config['lang_tgt']}",
        split="train",
    )
    # Build tokenizers
    tokenizer_src = get_or_build_tokenizer(config, ds_raw, config["lang_src"])
    tokenizer_tgt = get_or_build_tokenizer(config, ds_raw, config["lang_tgt"])

    # training validation split
    raw_ds_size = len(ds_raw)
    train_ds_size = int(0.9 * raw_ds_size)
    val_ds_size = raw_ds_size - train_ds_size
    split = ds_raw.train_test_split(test_size=0.1)
    train_ds_raw = split["train"]
    val_ds_raw = split["test"]

    train_ds = BilingualDataSet(
        train_ds_raw,
        tokenizer_src,
        tokenizer_tgt,
        config["lang_src"],
        config["lang_tgt"],
        config["seq_len"],
    )

    val_ds = BilingualDataSet(
        val_ds_raw,
        tokenizer_src,
        tokenizer_tgt,
        config["lang_src"],
        config["lang_tgt"],
        config["seq_len"],
    )

    # what is the max sequence length
    max_seq_len_src = 0
    max_seq_len_tgt = 0

    for item in ds_raw:
        src_ids = tokenizer_src.encode(item["translation"][config["lang_src"]]).ids
        tgt_ids = tokenizer_tgt.encode(item["translation"][config["lang_tgt"]]).ids
        max_seq_len_src = max(len(src_ids), max_seq_len_src)
        max_seq_len_tgt = max(len(tgt_ids), max_seq_len_tgt)
    print(f"max_seq_len_src: {max_seq_len_src}")
    print(f"max_seq_len_tgt:{max_seq_len_tgt}")

    train_dataloader = DataLoader(
        train_ds, batch_size=config["batch_size"], shuffle=True
    )
    # for validation use batch size of 1, to see it one by one
    val_dataloader = DataLoader(val_ds, batch_size=1, shuffle=True)

    return train_dataloader, val_dataloader, tokenizer_src, tokenizer_tgt


def get_model(config, vocab_src_len, vocab_tgt_len):
    model = build_transformer(
        vocab_src_len,
        vocab_tgt_len,
        config["seq_len"],
        config["seq_len"],
        config["d_model"],
    )
    return model


def train_model(config):
    # define the device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device {device}")

    path = Path(config["model_folder"]).mkdir(parents=True, exist_ok=True)

    train_dataloader, val_dataloader, tokenizer_src, tokenizer_tgt = get_ds(config)
    model = get_model(
        config, tokenizer_src.get_vocab_size(), tokenizer_tgt.get_vocab_size()
    ).to(device)
    # Tensorboard
    writer = SummaryWriter(config["experiment_name"])

    optimizer = torch.optim.Adam(model.parameters(), lr=config["lr"], eps=1e-9)

    # saving state of partial training result, state of optimizer
    initial_epoch = 0
    global_step = 0
    if config["preload"]:
        model_filename = get_weights_file_path(config, config["preload"])
        print(f"preload model: {model_filename}")
        state = torch.load(model_filename)
        initial_epoch = state["epoch"] + 1
        optimizer.load_state_dict(state["optimizer_state_dict"])
        global_step = state["global_step"]

    # loss function
    loss_fn: nn.CrossEntropyLoss = nn.CrossEntropyLoss(
        ignore_index=tokenizer_src.token_to_id("[PAD]"), label_smoothing=0.1
    )
    loss_fn = loss_fn.to(device=device)

    for epoch in range(initial_epoch, config["num_epochs"]):
        model.train()
        batch_iterator = tqdm(train_dataloader, desc=f"Processing batch: {epoch:02d}")
        for batch in batch_iterator:
            encoder_input = batch["encoder_input"].to(device)  # (Batch, Seq_Len)
            decoder_input = batch["decoder_input"].to(device)  # (Batch, Seq_Len)
            encoder_mask = batch["encoder_mask"].to(device)  # (Batch, 1, 1, Seq_Len)
            decoder_mask = batch["decoder_mask"].to(
                device
            )  # (Batch, 1, Seq_Len, Seq_Len)

            # Run the tensors through the transformer
            encoder_output = model.encode(encoder_input, encoder_mask)
            decoder_output = model.decode(
                decoder_input,
                encoder_output,
                encoder_mask,
                decoder_mask,
            )
            # projection
            project_output: torch.Tensor = model.project(
                decoder_output
            )  # (Batch, Seq_Len, tgt_vocab_size)
            label = batch["label"].to(device)
            loss: torch.Tensor = loss_fn(
                project_output.view(-1, tokenizer_tgt.get_vocab_size()), label.view(-1)
            )
            batch_iterator.set_postfix({f"{loss}": f"{loss.item():6.3f}"})

            # Log the loss
            writer.add_scalar("training_loss", loss.item(), global_step)
            writer.flush()

            # Backpropagate the loss
            loss.backward()

            # Update the weights
            optimizer.step()
            optimizer.zero_grad()

            global_step += 1

        # Save the model at every epoch
        model_filename = get_weights_file_path(config, f"{epoch:02d}")
        torch.save(
            {
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "global_step": global_step,
            },
            model_filename,
        )


if __name__ == "__main__":
    warnings.filterwarnings("ignore")
    print(f"hello")
    config = get_config()
    train_model(config)
