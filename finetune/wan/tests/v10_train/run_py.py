"""Run a repo script as __main__ with common.py's CPU-only setup (musubi's t5.py calls
torch.cuda.current_device() at import).   python run_py.py <script> [args...]"""
import runpy
import sys

import common  # noqa: F401  (patches torch.cuda.current_device, sets env defaults)

script = sys.argv[1]
sys.argv = sys.argv[1:]
runpy.run_path(script, run_name="__main__")
