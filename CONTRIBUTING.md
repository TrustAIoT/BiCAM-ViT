# Contributing

Thank you for your interest in BiCAM. Please open an issue before proposing a
large change. Bug reports should include the model name, `torch` and `timm`
versions, a minimal input example, and the full traceback.

For a development installation:

```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -e ".[dev,viz]"
pytest -q
```

Do not commit datasets, checkpoints, generated heatmaps, or notebook outputs.
