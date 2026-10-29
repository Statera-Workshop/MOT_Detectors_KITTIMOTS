"""Train image YOLOv13 for 10 epochs, then evaluate on the test split."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from common.runner import main
from detectors.yolov13.build import build_model

if __name__ == "__main__":
    main("train", task="image", version="v13", build_model=build_model)
