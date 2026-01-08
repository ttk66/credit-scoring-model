#!/usr/bin/env python3
"""
Скрипт для обучения нейронной сети и конвертации в ONNX
"""
from src.models.convert_to_onnx import run_complete_pipeline
from src.models.nn_model import main as train_nn
import argparse
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT_DIR))


def main():
    parser = argparse.ArgumentParser(
        description="Train NN model and convert to ONNX")
    parser.add_argument(
        "--train-only",
        action="store_true",
        help="Only train NN model")
    parser.add_argument(
        "--convert-only", action="store_true", help="Only convert to ONNX"
    )
    parser.add_argument(
        "--epochs", type=int, default=50, help="Number of training epochs"
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=128,
        help="Batch size")

    args = parser.parse_args()

    print("=" * 60)
    print("NEURAL NETWORK TRAINING AND ONNX CONVERSION")
    print("=" * 60)

    if not args.convert_only:
        print("\nTRAINING NEURAL NETWORK")
        print("-" * 40)

        model, scaler, test_auc = train_nn()
        print(f"\nNeural network training completed")
        print(f"Test AUC: {test_auc:.4f}")

    # Конвертация в ONNX
    if not args.train_only:
        print("\nCONVERTING TO ONNX")
        print("-" * 40)

        results = run_complete_pipeline()

        if results["conversion_valid"]:
            print("\nONNX conversion completed successfully!")
            print(f"Speedup: {results['performance']['speedup']:.2f}x")
        else:
            print("\nONNX conversion failed validation!")

    print("\n" + "=" * 60)
    print("PROCESS COMPLETED")
    print("=" * 60)


if __name__ == "__main__":
    main()
