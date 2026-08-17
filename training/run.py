#!/usr/bin/env python3
"""Main entry point for cattle ReID training pipeline.

Usage:
    python -m training.run                # full pipeline
    python -m training.run --step download # download CID dataset
    python -m training.run --step process  # YOLO crop + split
    python -m training.run --step train    # train OSNet
    python -m training.run --step evaluate # evaluate
    python -m training.run --step export   # export ONNX
"""
import argparse
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


STEP_FUNCTIONS = {
    "download": lambda: __import__("training.download", fromlist=[""]).download(),
    "process": lambda: __import__("training.process", fromlist=[""]).process_images(),
    "train": lambda: __import__("training.train", fromlist=[""]).train(),
    "evaluate": lambda: __import__("training.evaluate", fromlist=[""]).evaluate(),
    "export": lambda: __import__("training.export", fromlist=[""]).export_onnx(),
}

ALL_STEPS = ["download", "process", "train", "evaluate", "export"]


def run(step="all"):
    if step == "all":
        for s in ALL_STEPS:
            run(s)
        return

    if step not in STEP_FUNCTIONS:
        print(f"Unknown step: {step}. Choose from: {', '.join(ALL_STEPS)}")
        sys.exit(1)

    STEP_FUNCTIONS[step]()


def main():
    ap = argparse.ArgumentParser(description="Cattle ReID training pipeline")
    ap.add_argument("--step", default="all",
                    choices=ALL_STEPS + ["all"],
                    help="Which pipeline step to run (default: all)")
    args = ap.parse_args()
    run(args.step)


if __name__ == "__main__":
    main()
