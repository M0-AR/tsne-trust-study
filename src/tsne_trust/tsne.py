"""Shim so `from .sne import run_sne` and `from .tsne import run_tsne` both work."""
from .sne import run_sne, run_tsne  # noqa
