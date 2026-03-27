# Conversation Notes - 2026-03-26

## Git & Project Setup
- Initialized git repo with `.gitignore` for Python/ML project (bytecode, venvs, model weights, datasets, logs, IDE files)
- Virtual environment placed outside repo at `~/src/AI-ML/venv-AI-ML/` for sharing across projects
- Created bash function `ai_venv_banner` for `~/.bashrc` that warns if the venv is not activated

## Neovim Snippets
- Added custom VS Code-style snippets at `~/.config/nvim/snippets/python.json` with `package.json`
- Snippets: `__init__` (simple init) and `forward` (nn.Module forward method)
- Required adding a `lazy_load` call **before** the friendly-snippets one in `init.lua`:
  ```lua
  require('luasnip.loaders.from_vscode').lazy_load({ paths = { vim.fn.stdpath('config') .. '/snippets' } })
  require('luasnip.loaders.from_vscode').lazy_load()
  ```
- Note: LSP completions may appear first; choose the snippet option from the list

## Code Discussions

### PositionalEncoding.forward
- `self.pe[:, :x.shape[1], :]` — slices positional encodings to match actual batch sequence length
- `x` shape is `(batch, seq_len, d_model)`, `pe` shape is `(1, max_seq_len, d_model)`
- The `1` in pe broadcasts across all sequences in the batch
- `:x.shape[1]` handles batches shorter than max `seq_len`

### Linter Issues
- `self.register_buffer("pe", pe)` causes pyright to not recognize `self.pe` as a Tensor
- Fix: add `self.pe: torch.Tensor` type annotation before `register_buffer`
- Do NOT use `self.pe = torch.Tensor` (assignment) — use `self.pe: torch.Tensor` (annotation)

### train.py
- `random_split` expects PyTorch `Dataset`, but HuggingFace returns its own `Dataset`
- Fix: use `ds_raw.train_test_split(test_size=0.1)` instead
- The max sequence length loop (encoding entire dataset just to print) is wasteful — commented out

### Parameter Counting
- `t._parameters.__sizeof__()` gives dict memory size in bytes, not parameter count
- Correct: `sum(p.numel() for p in t.parameters())` — iterates over a few dozen tensors, not millions of values
