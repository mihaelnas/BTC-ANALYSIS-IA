import sys
from pathlib import Path

# Add project root to sys.path so that tests can import src/ and config/ modules
project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))
