"""Small compatibility setup shared by notebook and command-line entry points."""
import os


def configure_runtime():
    # pandas 3 + PyArrow can raise a native access violation after Torch loads
    # on this Windows environment. Use Python string storage, leaving eval.py intact.
    if os.name == "nt":
        import pandas as pd
        pd.options.mode.string_storage = "python"


configure_runtime()
