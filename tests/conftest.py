import sys
from pathlib import Path

# the repo root is itself the `oetoolbox` package (relative imports), so put its parent on sys.path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
