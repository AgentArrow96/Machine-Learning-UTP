"""Start downloading with the system Python before installing the ML environment."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from floodnet.common import environment
from floodnet.acquire import download

if __name__ == "__main__":
    environment()
    download()
