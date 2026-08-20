#!/usr/bin/env python3
"""Main entry point for the cattle ReID training pipeline.

Usage:
    python -m training.run                     # full pipeline
    python -m training.run --step download     # download CID dataset
    python -m training.run --step process      # YOLO crop + split
    python -m training.run --step train        # train HanwooReID (ViT + PHE)
    python -m training.run --step evaluate     # query vs gallery evaluation
    python -m training.run --step export       # export ONNX
"""
import argparse
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

ALL_STEPS = ["download", "process", "train", "evaluate", "export"]


def run(step="all"):
    if step == "all":
        for s in ALL_STEPS:
            run(s)
        return

    if step == "download":
        from training.download import download, extract
        download()
        extract()
    elif step == "process":
        from training.process import process_images
        process_images()
    elif step == "train":
        from training.train_v3 import parse_args as train_args
        from training.train_v3 import train
        train(train_args([]))
    elif step == "evaluate":
        from training.test_model_v3 import main as evaluate
        evaluate([])
    elif step == "export":
        from training.export import export_onnx
        export_onnx()
    else:
        print(f"Unknown step: {step}. Choose from: {', '.join(ALL_STEPS)}")
        sys.exit(1)


def main():
    ap = argparse.ArgumentParser(description="Cattle ReID training pipeline")
    ap.add_argument("--step", default="all",
                    choices=ALL_STEPS + ["all"],
                    help="Which pipeline step to run (default: all)")
    args = ap.parse_args()
    run(args.step)


if __name__ == "__main__":
    main()