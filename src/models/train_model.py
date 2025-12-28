#!/usr/bin/env python3
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.models.train import train_model
from src.models.plots import plot_roc_curve

if __name__ == "__main__":
    best_model, X_test, y_test, y_prob, metrics = train_model()
    plot_roc_curve(y_test, y_prob)