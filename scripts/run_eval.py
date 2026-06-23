import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
from apertura.eval.runner import main
if __name__ == "__main__":
    main()