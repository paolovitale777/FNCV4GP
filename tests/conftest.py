import sys
from pathlib import Path

# Make the repository root importable so `import NV4GP` (or FNV4GP, depending
# on what the main module ends up being named after the rename) works when
# pytest is run from anywhere, including CI.
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
