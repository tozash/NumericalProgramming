"""Pytest configuration to ensure src module is importable."""

import sys
from pathlib import Path

# Add the project root to Python path so tests can import src
project_root = Path(__file__).parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

