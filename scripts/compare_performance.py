#!/usr/bin/env python3
"""
Сравнение производительности всех моделей: Random Forest, PyTorch NN, ONNX, Оптимизированная
"""

import warnings
from typing import Dict, List
import matplotlib.pyplot as plt
import json
import time
import pandas as pd
import numpy as np
import torch
import argparse
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT_DIR))


warnings.filterwarnings("ignore")

# Пути к моделям
MODEL_PATHS = {
    "random_forest": Path("models/best_model.joblib"),
    "pytorch_nn": Path("models/nn_model.pth"),
    "onnx_nn": Path("models/nn_model.onnx"),
    "optimized_nn": Path("models/nn_model_pruned.pth"),
    "onnx_optimized": Path("models/nn_model_quantized.onnx"),
}

SCALER_PATH = Path("models/nn_scaler.joblib")
RESULTS_PATH = Path("models/performance_comparison_final.json")


class PerformanceComparator:
    """Сравнение производительности всех моделей"""

    def __init__(self, iterations: int = 1000, batch_sizes: List[int] = None):
        self.iterations = iterations
        self.batch_sizes = batch_sizes or [1, 4, 16, 64]
        self.models = {}
        self.scaler = None
        self.results = {}

        self.devices = ["cpu"]
        if torch.cuda.is_available():
            self.devices.append("cuda")
            print(f"GPU доступен: {torch.cuda.get_device_name(0)}")
        else:
            print("GPU не доступен, тестирование только на CPU")

    def load_all_models(self):
        """Загрузка всех доступных моделей"""
        print("\n" + "=" * 60)
        print("LOADING MODELS")
        print("=" * 60)

        # Загрузка Random Forest
        if MODEL_PATHS["random_forest"].exists():
            try:
                import joblib

                self.models["random_forest"] = joblib.load(
                    MODEL_PATHS["random_forest"])
                print("Random Forest model loaded")
            except Exception as e:
                print(f"Failed to load Random Forest: {e}")

        # Загрузка PyTorch NN
        if MODEL_PATHS["pytorch_nn"].exists():
            try:
                import numpy

                torch.serialization.add_safe_globals(
                    [numpy._core.multiarray.scalar])

                checkpoint = torch.load(
                    MODEL_PATHS["pytorch_nn"], map_location="cpu", weights_only=False
                )

                from src.models.nn_model import CreditScoringNN

                model = CreditScoringNN(
                    input_size=checkpoint.get(
                        "input_size", 32))
                model.load_state_dict(checkpoint["model_state_dict"])
                model.eval()

                self.models["pytorch_nn"] = model
                print("PyTorch NN model loaded")
            except Exception as e:
                print(f"Failed to load PyTorch NN: {e}")

        # Загрузка ONNX NN
        if MODEL_PATHS["onnx_nn"].exists():
            try:
                import onnxruntime as ort

                self.models["onnx_nn"] = ort.InferenceSession(
                    str(MODEL_PATHS["onnx_nn"]), providers=["CPUExecutionProvider"]
                )
                print("ONNX NN model loaded")
            except Exception as e:
                print(f"Failed to load ONNX NN: {e}")

        # Загрузка оптимизированной модели
        if MODEL_PATHS["optimized_nn"].exists():
            try:
                import numpy

                torch.serialization.add_safe_globals(
                    [numpy._core.multiarray.scalar])

                checkpoint = torch.load(
                    MODEL_PATHS["optimized_nn"], map_location="cpu", weights_only=False
                )

                from src.models.nn_model import CreditScoringNN

                model = CreditScoringNN(
                    input_size=checkpoint.get(
                        "input_size", 32))
                model.load_state_dict(checkpoint["model_state_dict"])
                model.eval()

                self.models["optimized_nn"] = model
                print("Optimized NN model loaded")
            except Exception as e:
                print(f"Failed to load optimized NN: {e}")

        # Загрузка квантованной ONNX модели
        if MODEL_PATHS["onnx_optimized"].exists():
            try:
                import onnxruntime as ort

                self.models["onnx_optimized"] = ort.InferenceSession(
                    str(MODEL_PATHS["onnx_optimized"]),
                    providers=["CPUExecutionProvider"],
                )
                print("Quantized ONNX model loaded")
            except Exception as e:
                print(f"Failed to load quantized ONNX: {e}")

        # Загрузка скейлера
        try:
            import joblib

            self.scaler = joblib.load(SCALER_PATH)
            print("Scaler loaded")
        except Exception as e:
            print(f"Failed to load scaler: {e}")
            self.scaler = None

        print(f"\nTotal models loaded: {len(self.models)}")
        for name in self.models.keys():
            print(f"  - {name}")

    def prepare_test_data(self, n_samples: int = 1000) -> np.ndarray:
        """Подготовка тестовых данных"""
        print(f"\nPreparing test data ({n_samples} samples)...")

        # Генерация реалистичных данных
        np.random.seed(42)

        # Основные фичи
        n_features = 32

        # Генерация данных
        test_data = np.random.randn(n_samples, n_features)

        # Масштабирование если есть скейлер
        if self.scaler is not None:
            test_data = self.scaler.transform(test_data)

        print(f"  Data shape: {test_data.shape}")
        print(f"  Data range: [{test_data.min():.4f}, {test_data.max():.4f}]")

        return test_data.astype(np.float32)

    def benchmark_random_forest(
        self, model, test_data: np.ndarray, batch_size: int
    ) -> Dict:
        """Бенчмарк Random Forest модели"""
        results = {"time_ms": 0, "predictions_per_second": 0, "memory_mb": 0}

        # Создаем DataFrame с именами колонок
        feature_names = [
            "limit_bal",
            "sex",
            "education",
            "marriage",
            "age",
            "pay_0",
            "pay_2",
            "pay_3",
            "pay_4",
            "pay_5",
            "pay_6",
            "bill_amt1",
            "bill_amt2",
            "bill_amt3",
            "bill_amt4",
            "bill_amt5",
            "bill_amt6",
            "pay_amt1",
            "pay_amt2",
            "pay_amt3",
            "pay_amt4",
            "pay_amt5",
            "pay_amt6",
            "limit_age_ratio",
            "avg_bill_amt",
            "avg_pay_amt",
            "bill_pay_ratio",
            "num_late_payments",
            "max_delay",
            "avg_delay",
            "bill_trend",
            "age_bin",
        ]

        df = pd.DataFrame(test_data,
                          columns=feature_names[: test_data.shape[1]])

        # Warmup
        for _ in range(10):
            _ = model.predict_proba(df.iloc[:1])

        # Бенчмарк
        start_time = time.time()

        for i in range(0, len(df), batch_size):
            batch = df.iloc[i: i + batch_size]
            _ = model.predict_proba(batch)

        total_time = time.time() - start_time
        n_predictions = len(df)

        results["time_ms"] = total_time * 1000
        results["predictions_per_second"] = n_predictions / total_time

        # Оценка памяти
        import joblib
        import io

        buffer = io.BytesIO()
        joblib.dump(model, buffer)
        results["memory_mb"] = len(buffer.getvalue()) / (1024 * 1024)

        return results

    def measure_system_resources(self) -> Dict:
        """Измерение системных ресурсов"""
        import psutil

        resources = {
            "cpu_percent": psutil.cpu_percent(interval=0.1),
            "memory_percent": psutil.virtual_memory().percent,
            "memory_used_mb": psutil.virtual_memory().used / (1024 * 1024),
            "memory_total_mb": psutil.virtual_memory().total / (1024 * 1024),
        }

        # Измерение GPU если доступно
        if torch.cuda.is_available():
            resources["gpu_name"] = torch.cuda.get_device_name(0)
            resources["gpu_memory_total_mb"] = torch.cuda.get_device_properties(
                0
            ).total_memory / (1024 * 1024)

        return resources

    def benchmark_pytorch(
        self, model, test_data: np.ndarray, batch_size: int, device: str = "cpu"
    ) -> Dict:
        """Бенчмарк PyTorch модели на указанном устройстве"""
        results = {
            "time_ms": 0,
            "predictions_per_second": 0,
            "memory_mb": 0,
            "device": device,
        }

        import io

        # Переносим модель на устройство
        device_obj = torch.device(device)
        model_on_device = model.to(device_obj)
        model_on_device.eval()

        # Конвертация в тензор и перенос на устройство
        test_tensor = torch.FloatTensor(test_data).to(device_obj)

        # Warmup
        with torch.no_grad():
            for _ in range(10):
                _ = model_on_device(test_tensor[:1])

            # Синхронизация для GPU
            if device == "cuda":
                torch.cuda.synchronize()

        # Бенчмарк
        start_time = time.time()

        with torch.no_grad():
            for i in range(0, len(test_tensor), batch_size):
                batch = test_tensor[i: i + batch_size]
                _ = model_on_device(batch)

            # Синхронизация для GPU
            if device == "cuda":
                torch.cuda.synchronize()

        total_time = time.time() - start_time
        n_predictions = len(test_data)

        results["time_ms"] = total_time * 1000
        results["predictions_per_second"] = n_predictions / total_time

        # Измерение использования памяти GPU
        if device == "cuda":
            results["gpu_memory_mb"] = torch.cuda.max_memory_allocated() / \
                (1024 * 1024)
            torch.cuda.reset_peak_memory_stats()
        else:
            results["gpu_memory_mb"] = 0

        # Оценка памяти модели
        buffer = io.BytesIO()
        torch.save(model.state_dict(), buffer)
        results["memory_mb"] = len(buffer.getvalue()) / (1024 * 1024)

        return results

    def benchmark_onnx(self, session, test_data: np.ndarray,
                       batch_size: int) -> Dict:
        """Бенчмарк ONNX модели"""
        results = {"time_ms": 0, "predictions_per_second": 0, "memory_mb": 0}

        input_name = session.get_inputs()[0].name

        # Warmup
        for _ in range(10):
            session.run(None, {input_name: test_data[:1]})

        # Бенчмарк
        start_time = time.time()

        for i in range(0, len(test_data), batch_size):
            batch = test_data[i: i + batch_size]
            session.run(None, {input_name: batch})

        total_time = time.time() - start_time
        n_predictions = len(test_data)

        results["time_ms"] = total_time * 1000
        results["predictions_per_second"] = n_predictions / total_time

        # Оценка памяти

        for path in MODEL_PATHS.values():
            if "onnx" in str(path).lower() and path.exists():
                results["memory_mb"] = path.stat().st_size / (1024 * 1024)
                break

        return results

    def run_benchmarks(self):
        """Запуск всех бенчмарков"""
        print("\n" + "=" * 60)
        print("RUNNING BENCHMARKS")
        print("=" * 60)
        print(f"Iterations per batch: {self.iterations}")
        print(f"Batch sizes: {self.batch_sizes}")
        print(f"Devices: {self.devices}")

        # Измерение системных ресурсов до тестов
        print("\nSystem resources before tests:")
        before_resources = self.measure_system_resources()
        print(f"  CPU: {before_resources['cpu_percent']:.1f}%")
        print(f"  Memory: {before_resources['memory_used_mb']:.1f} MB / {before_resources['memory_total_mb']:.1f} MB")

        # Подготовка данных
        test_data = self.prepare_test_data(n_samples=1000)

        all_results = {}

        for device in self.devices:
            print(f"\n{'=' * 40}")
            print(f"TESTING ON: {device.upper()}")
            print("=" * 40)

            for batch_size in self.batch_sizes:
                print(f"\nBatch size: {batch_size} (Device: {device})")
                print("-" * 40)

                batch_results = {}

                for model_name, model in self.models.items():
                    # Тестируем только PyTorch модели на GPU
                    if device == "cuda" and model_name not in [
                        "pytorch_nn",
                        "optimized_nn",
                    ]:
                        continue

                    print(f"  Testing {model_name:20}...", end=" ", flush=True)

                    try:
                        if model_name == "random_forest":
                            results = self.benchmark_random_forest(
                                model, test_data, batch_size
                            )
                        elif "onnx" in model_name:
                            results = self.benchmark_onnx(
                                model, test_data, batch_size)
                        else:
                            # Для PyTorch моделей указываем устройство
                            results = self.benchmark_pytorch(
                                model, test_data, batch_size, device
                            )

                        results["device"] = device
                        batch_results[f"{model_name}_{device}"] = results

                        # Добавляем информацию о памяти GPU
                        if device == "cuda" and "gpu_memory_mb" in results:
                            print(
                                f"{
                                    results['predictions_per_second']:.1f} pred/s (GPU Mem: {
                                    results['gpu_memory_mb']:.1f} MB)"
                            )
                        else:
                            print(
                                f"{results['predictions_per_second']:.1f} pred/s")

                    except Exception as e:
                        print(f"Failed: {str(e)[:50]}")
                        batch_results[model_name] = {
                            "time_ms": 0,
                            "predictions_per_second": 0,
                            "memory_mb": 0,
                            "device": device,
                        }

                all_results[f"batch_{batch_size}_{device}"] = batch_results

        # Измерение системных ресурсов после тестов
        print("\nSystem resources after tests:")
        after_resources = self.measure_system_resources()
        print(f"  CPU: {after_resources['cpu_percent']:.1f}%")
        print(
            f"  Memory: {
                after_resources['memory_used_mb']:.1f} MB / {
                after_resources['memory_total_mb']:.1f} MB"
        )

        self.results = all_results
        return all_results

    def calculate_speedup(self):
        """Расчет ускорения между моделями"""
        speedup_results = {}

        # Используем средний batch size для сравнения
        if not self.batch_sizes:
            return {}

        mid_batch = self.batch_sizes[len(self.batch_sizes) // 2]
        batch_key = f"batch_{mid_batch}_cpu"

        if batch_key not in self.results:
            return {}

        batch_results = self.results[batch_key]
        speedups = {}

        # Получаем скорости для всех моделей
        model_speeds = {}
        for model_key, metrics in batch_results.items():
            speed = metrics.get("predictions_per_second", 0)
            model_name = model_key.replace("_cpu", "")
            model_speeds[model_name] = speed

        # Сравнение PyTorch vs ONNX
        if "pytorch_nn" in model_speeds and "onnx_nn" in model_speeds:
            pytorch_speed = model_speeds["pytorch_nn"]
            onnx_speed = model_speeds["onnx_nn"]

            if pytorch_speed > 0:
                speedup = onnx_speed / pytorch_speed
                speedups["pytorch_vs_onnx"] = {
                    "pytorch_speed": pytorch_speed,
                    "onnx_speed": onnx_speed,
                    "speedup": speedup,
                    "description": f"ONNX is {speedup:.2f}x faster than PyTorch",
                }

        # Сравнение Random Forest vs ONNX
        if "random_forest" in model_speeds and "onnx_nn" in model_speeds:
            rf_speed = model_speeds["random_forest"]
            onnx_speed = model_speeds["onnx_nn"]

            if rf_speed > 0:
                speedup = onnx_speed / rf_speed
                speedups["rf_vs_onnx"] = {
                    "rf_speed": rf_speed,
                    "onnx_speed": onnx_speed,
                    "speedup": speedup,
                    "description": f"ONNX is {speedup:.2f}x faster than Random Forest",
                }

        # Сравнение оригинальной vs оптимизированной
        if "pytorch_nn" in model_speeds and "optimized_nn" in model_speeds:
            original_speed = model_speeds["pytorch_nn"]
            optimized_speed = model_speeds["optimized_nn"]

            if original_speed > 0:
                speedup = optimized_speed / original_speed
                speedups["original_vs_optimized"] = {
                    "original_speed": original_speed,
                    "optimized_speed": optimized_speed,
                    "speedup": speedup,
                    "description": f"Optimized is {speedup:.2f}x faster than original",
                }

        if speedups:
            speedup_results[batch_key] = speedups

        return speedup_results

    def create_visualizations(self):
        """Создание визуализаций сравнения"""
        print("\nCreating visualizations...")

        try:
            # 1. Сравнение CPU vs GPU для PyTorch моделей
            plt.figure(figsize=(15, 10))

            # График 1: Сравнение CPU vs GPU
            plt.subplot(2, 2, 1)

            cpu_gpu_data = {}
            for model_name in ["pytorch_nn", "optimized_nn"]:
                cpu_speeds = []
                gpu_speeds = []
                batch_labels = []

                for batch_size in self.batch_sizes:
                    cpu_key = f"batch_{batch_size}_cpu"
                    gpu_key = f"batch_{batch_size}_cuda"

                    if (
                        cpu_key in self.results
                        and f"{model_name}_cpu" in self.results[cpu_key]
                    ):
                        cpu_speed = self.results[cpu_key][f"{model_name}_cpu"][
                            "predictions_per_second"
                        ]
                        cpu_speeds.append(cpu_speed)

                    if (
                        gpu_key in self.results
                        and f"{model_name}_cuda" in self.results[gpu_key]
                    ):
                        gpu_speed = self.results[gpu_key][f"{model_name}_cuda"][
                            "predictions_per_second"
                        ]
                        gpu_speeds.append(gpu_speed)

                    batch_labels.append(str(batch_size))

                if cpu_speeds and gpu_speeds:
                    x = np.arange(len(batch_labels))
                    width = 0.35

                    plt.bar(
                        x - width / 2,
                        cpu_speeds,
                        width,
                        label=f"{model_name} CPU",
                        alpha=0.8,
                    )
                    plt.bar(
                        x + width / 2,
                        gpu_speeds,
                        width,
                        label=f"{model_name} GPU",
                        alpha=0.8,
                    )

            plt.xlabel("Batch Size")
            plt.ylabel("Predictions per Second")
            plt.title("CPU vs GPU Performance Comparison")
            plt.xticks(np.arange(len(batch_labels)), batch_labels)
            plt.legend()
            plt.grid(True, alpha=0.3)

            # График 2: Ускорение GPU vs CPU
            plt.subplot(2, 2, 2)

            speedup_by_batch = {}
            for batch_size in self.batch_sizes:
                cpu_key = f"batch_{batch_size}_cpu"
                gpu_key = f"batch_{batch_size}_cuda"

                for model_name in ["pytorch_nn", "optimized_nn"]:
                    cpu_model_key = f"{model_name}_cpu"
                    gpu_model_key = f"{model_name}_cuda"

                    if (
                        cpu_key in self.results
                        and cpu_model_key in self.results[cpu_key]
                        and gpu_key in self.results
                        and gpu_model_key in self.results[gpu_key]
                    ):

                        cpu_speed = self.results[cpu_key][cpu_model_key][
                            "predictions_per_second"
                        ]
                        gpu_speed = self.results[gpu_key][gpu_model_key][
                            "predictions_per_second"
                        ]

                        if cpu_speed > 0:
                            speedup = gpu_speed / cpu_speed
                            if batch_size not in speedup_by_batch:
                                speedup_by_batch[batch_size] = []
                            speedup_by_batch[batch_size].append(
                                (model_name, speedup))

            if speedup_by_batch:
                x = np.arange(len(speedup_by_batch))
                batch_sizes_list = list(speedup_by_batch.keys())

                for i, (batch_size, speedups) in enumerate(
                        speedup_by_batch.items()):
                    for model_name, speedup in speedups:
                        color = "green" if speedup >= 1 else "red"
                        plt.bar(
                            i + 0.2 if "pytorch" in model_name else i - 0.2,
                            speedup,
                            width=0.4,
                            color=color,
                            alpha=0.7,
                            label=model_name if i == 0 else "",
                        )

                plt.axhline(
                    y=1, color="black", linestyle="--", alpha=0.5, label="Baseline (1x)"
                )
                plt.xlabel("Batch Size")
                plt.ylabel("Speedup (GPU/CPU)")
                plt.title("GPU Speedup Over CPU")
                plt.xticks(x, batch_sizes_list)
                plt.legend()
                plt.grid(True, alpha=0.3)

            # Сохраняем визуализацию CPU/GPU
            plt.tight_layout()
            plt.savefig(
                "models/cpu_gpu_comparison.png",
                dpi=150,
                bbox_inches="tight")
            plt.close()

            print("CPU/GPU comparison saved: models/cpu_gpu_comparison.png")

            # 2. Сравнение всех моделей
            plt.figure(figsize=(14, 10))

            model_names = []
            avg_speeds = []
            avg_times = []
            model_sizes = []

            for model_name in self.models.keys():
                speeds = []
                times = []

                for batch_size in self.batch_sizes:
                    batch_key = f"batch_{batch_size}"
                    if (
                        batch_key in self.results
                        and model_name in self.results[batch_key]
                    ):
                        results = self.results[batch_key][model_name]
                        if results["predictions_per_second"] > 0:
                            speeds.append(results["predictions_per_second"])
                            times.append(results["time_ms"])

                if speeds:
                    model_names.append(model_name.replace("_", " ").title())
                    avg_speeds.append(np.mean(speeds))
                    avg_times.append(np.mean(times))

                    # Получаем размер модели
                    model_size = 0
                    for path in MODEL_PATHS.values():
                        if model_name in str(path).lower() and path.exists():
                            model_size = path.stat().st_size / (1024 * 1024)
                            break
                    model_sizes.append(model_size)

            # Средняя скорость
            plt.subplot(2, 2, 1)
            colors = plt.cm.Set3(np.linspace(0, 1, len(model_names)))
            bars1 = plt.bar(model_names, avg_speeds, color=colors)
            plt.xlabel("Model")
            plt.ylabel("Average Predictions per Second")
            plt.title("Average Inference Speed")
            plt.xticks(rotation=45, ha="right")
            plt.grid(True, alpha=0.3)

            # Добавляем значения
            for bar, speed in zip(bars1, avg_speeds):
                plt.text(
                    bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 10,
                    f"{speed:.0f}",
                    ha="center",
                    fontsize=9,
                )

            # Время инференса
            plt.subplot(2, 2, 2)
            bars2 = plt.bar(model_names, avg_times, color=colors)
            plt.xlabel("Model")
            plt.ylabel("Average Time per Inference (ms)")
            plt.title("Average Inference Time")
            plt.xticks(rotation=45, ha="right")
            plt.grid(True, alpha=0.3)

            for bar, time_val in zip(bars2, avg_times):
                plt.text(
                    bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 0.1,
                    f"{time_val:.2f}",
                    ha="center",
                    fontsize=9,
                )

            # Размер модели
            plt.subplot(2, 2, 3)
            bars3 = plt.bar(model_names, model_sizes, color=colors)
            plt.xlabel("Model")
            plt.ylabel("Model Size (MB)")
            plt.title("Model Size Comparison")
            plt.xticks(rotation=45, ha="right")
            plt.grid(True, alpha=0.3)

            for bar, size in zip(bars3, model_sizes):
                plt.text(
                    bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 0.1,
                    f"{size:.2f}",
                    ha="center",
                    fontsize=9,
                )

            # Ускорение относительно PyTorch
            plt.subplot(2, 2, 4)
            speedup_data = self.calculate_speedup()

            if speedup_data:
                first_batch = list(speedup_data.keys())[0]
                comparisons = speedup_data[first_batch]

                comparison_names = []
                speedup_values = []

                for comp_name, comp_data in comparisons.items():
                    comparison_names.append(
                        comp_name.replace("_", " ").title())
                    speedup_values.append(comp_data["speedup"])

                if comparison_names:
                    colors_comp = [
                        "green" if x >= 1 else "red" for x in speedup_values]
                    bars4 = plt.bar(
                        comparison_names, speedup_values, color=colors_comp, alpha=0.6
                    )
                    plt.axhline(
                        y=1,
                        color="black",
                        linestyle="--",
                        alpha=0.5,
                        label="Baseline (1x)",
                    )
                    plt.xlabel("Comparison")
                    plt.ylabel("Speedup (x)")
                    plt.title("Performance Speedup")
                    plt.xticks(rotation=45, ha="right")
                    plt.legend()
                    plt.grid(True, alpha=0.3)

                    for bar, speedup in zip(bars4, speedup_values):
                        plt.text(
                            bar.get_x() + bar.get_width() / 2,
                            bar.get_height() + 0.1,
                            f"{speedup:.2f}x",
                            ha="center",
                            fontsize=9,
                        )

            plt.tight_layout()
            plt.savefig(
                "models/performance_comparison_final.png", dpi=150, bbox_inches="tight"
            )
            plt.close()

            print("Visualization saved: models/performance_comparison_final.png")

            plt.figure(figsize=(12, 6))

            for model_name in self.models.keys():
                speeds_by_batch = []
                batch_labels = []

                for batch_size in self.batch_sizes:
                    batch_key = f"batch_{batch_size}"
                    if (
                        batch_key in self.results
                        and model_name in self.results[batch_key]
                    ):
                        speed = self.results[batch_key][model_name][
                            "predictions_per_second"
                        ]
                        if speed > 0:
                            speeds_by_batch.append(speed)
                            batch_labels.append(str(batch_size))

                if speeds_by_batch:
                    plt.plot(
                        batch_labels,
                        speeds_by_batch,
                        "o-",
                        label=model_name.replace("_", " "),
                    )

            plt.xlabel("Batch Size")
            plt.ylabel("Predictions per Second")
            plt.title("Inference Speed by Batch Size")
            plt.legend()
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.savefig(
                "models/batch_size_comparison.png", dpi=150, bbox_inches="tight"
            )
            plt.close()

            print("Batch size comparison saved: models/batch_size_comparison.png")

        except Exception as e:
            print(f"Visualization error: {e}")

    def generate_summary_report(self):
        """Генерация итогового отчета"""
        print("\nGenerating summary report...")

        # Находим лучшую модель
        best_model_name = None
        best_speed = 0
        best_batch_size = None

        for batch_key, batch_results in self.results.items():
            for model_key, metrics in batch_results.items():
                speed = metrics.get("predictions_per_second", 0)
                if speed > best_speed:
                    best_speed = speed
                    # Извлекаем имя модели без суффикса _cpu
                    best_model_name = model_key.replace("_cpu", "")
                    # Извлекаем batch size из ключа
                    best_batch_size = batch_key.replace("batch_", "").replace(
                        "_cpu", ""
                    )

        # Собираем метрики по всем моделям
        performance_summary = {}

        # Список имен моделей без суффиксов
        base_model_names = [
            "random_forest",
            "pytorch_nn",
            "onnx_nn",
            "optimized_nn"]

        for base_name in base_model_names:
            if base_name not in self.models:
                continue

            speeds = []
            times = []

            for batch_size in self.batch_sizes:
                batch_key = f"batch_{batch_size}_cpu"
                if batch_key in self.results:
                    model_key_with_suffix = f"{base_name}_cpu"
                    model_key = (
                        model_key_with_suffix
                        if model_key_with_suffix in self.results[batch_key]
                        else base_name
                    )

                    if model_key in self.results[batch_key]:
                        metrics = self.results[batch_key][model_key]
                        speed = metrics.get("predictions_per_second", 0)
                        time_ms = metrics.get("time_ms", 0)

                        if speed > 0:
                            speeds.append(speed)
                            times.append(time_ms)

            if speeds:
                model_size_mb = 0
                model_path = MODEL_PATHS.get(base_name)
                if model_path and model_path.exists():
                    try:
                        model_size_mb = model_path.stat().st_size / (1024 * 1024)
                    except BaseException:
                        model_size_mb = 0

                performance_summary[base_name] = {
                    "avg_speed": np.mean(speeds),
                    "avg_time_ms": np.mean(times),
                    "min_speed": np.min(speeds),
                    "max_speed": np.max(speeds),
                    "speed_std": np.std(speeds),
                    "model_size_mb": model_size_mb,
                }

        # Формируем отчет
        summary = {
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "benchmark_config": {
                "iterations": self.iterations,
                "batch_sizes": self.batch_sizes,
                "devices": self.devices,
                "models_tested": list(self.models.keys()),
            },
            "best_model": {
                "name": best_model_name if best_model_name else "unknown",
                "avg_speed": best_speed,
                "optimal_batch_size": best_batch_size,
            },
            "performance_summary": performance_summary,
            "speedup_analysis": self.calculate_speedup(),
            "recommendations": [
                "Use ONNX for production deployment",
                "Batch processing improves throughput significantly",
                "Consider model size vs speed trade-offs",
                "Monitor accuracy after optimization",
            ],
        }

        # Добавляем анализ ускорения
        if "onnx_nn" in performance_summary and "pytorch_nn" in performance_summary:
            onnx_speed = performance_summary["onnx_nn"]["avg_speed"]
            pytorch_speed = performance_summary["pytorch_nn"]["avg_speed"]

            if pytorch_speed > 0:
                speedup = onnx_speed / pytorch_speed
                summary["recommendations"].insert(
                    0, f"ONNX provides {speedup:.1f}x speedup over PyTorch"
                )

        return summary

    def save_results(self):
        """Сохранение результатов"""
        print("\nSaving results...")

        summary = self.generate_summary_report()

        full_results = {
            "summary": summary,
            "detailed_results": self.results,
            "speedup_analysis": self.calculate_speedup(),
            "model_info": {
                model_name: {
                    "path": str(MODEL_PATHS.get(model_name, "")),
                    "exists": MODEL_PATHS.get(model_name, Path("")).exists(),
                    "size_mb": (
                        MODEL_PATHS.get(model_name, Path("")).stat().st_size
                        / (1024 * 1024)
                        if MODEL_PATHS.get(model_name, Path("")).exists()
                        else 0
                    ),
                }
                for model_name in self.models.keys()
            },
        }

        RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)

        with open(RESULTS_PATH, "w", encoding="utf-8") as f:
            json.dump(full_results, f, indent=2, ensure_ascii=False)

        print(f"Results saved: {RESULTS_PATH}")

        return full_results

    def print_final_report(self):
        """Вывод финального отчета"""
        print("\n" + "=" * 70)
        print("FINAL PERFORMANCE COMPARISON REPORT")
        print("=" * 70)

        summary = self.generate_summary_report()

        # Лучшая модель
        print(
            f"\nBEST MODEL: {
                summary['best_model']['name'].replace(
                    '_', ' ').title()}"
        )
        print(
            f"   Average Speed: {
                summary['best_model']['avg_speed']:.1f} predictions/second"
        )

        # Сводка по всем моделям
        print(f"\nPERFORMANCE SUMMARY:")
        print("-" * 60)
        print(f"{'Model':25} {'Speed (pred/s)':15} {'Time (ms)':12} {'Size (MB)':10}")
        print("-" * 60)

        for model_name, metrics in summary["performance_summary"].items():
            display_name = model_name.replace("_", " ").title()
            print(
                f"{
                    display_name:25} {
                    metrics['avg_speed']:15.1f} {
                    metrics['avg_time_ms']:12.3f} {
                    metrics['model_size_mb']:10.2f}"
            )

        # Анализ ускорения
        print(f"\nSPEEDUP ANALYSIS:")
        print("-" * 60)

        speedup_data = summary["speedup_analysis"]
        if speedup_data:
            first_batch = list(speedup_data.keys())[0]
            for comp_name, comp_data in speedup_data[first_batch].items():
                comp_display = comp_name.replace("_", " ").title()
                print(f"  {comp_display}:")
                print(f"    {comp_data['description']}")

        # Рекомендации
        print(f"\nRECOMMENDATIONS:")
        print("-" * 60)
        for i, rec in enumerate(summary["recommendations"], 1):
            print(f"  {i}. {rec}")

        print(f"\nFiles created:")
        print(f"  - {RESULTS_PATH}")
        print(f"  - models/performance_comparison_final.png")
        print(f"  - models/batch_size_comparison.png")

        print(f"\n" + "=" * 70)
        print("PERFORMANCE COMPARISON COMPLETED!")
        print("=" * 70)


def main():
    parser = argparse.ArgumentParser(
        description="Compare performance of all models")
    parser.add_argument(
        "--iterations", type=int, default=500, help="Number of iterations per benchmark"
    )
    parser.add_argument(
        "--batch-sizes",
        type=str,
        default="1,4,16,64",
        help="Comma-separated list of batch sizes",
    )
    parser.add_argument(
        "--simple", action="store_true", help="Run simplified benchmark (faster)"
    )

    args = parser.parse_args()

    # Преобразуем batch sizes
    batch_sizes = [int(bs.strip()) for bs in args.batch_sizes.split(",")]

    if args.simple:
        batch_sizes = [1, 16]
        args.iterations = 100

    print("=" * 70)
    print("COMPREHENSIVE MODEL PERFORMANCE COMPARISON")
    print("=" * 70)
    print(f"Iterations: {args.iterations}")
    print(f"Batch sizes: {batch_sizes}")
    print(f"Mode: {'Simple' if args.simple else 'Full'}")

    # Создаем компаратор
    comparator = PerformanceComparator(
        iterations=args.iterations, batch_sizes=batch_sizes
    )

    # Загрузка моделей
    comparator.load_all_models()

    if not comparator.models:
        print("\nNo models loaded! Exiting.")
        return

    comparator.run_benchmarks()
    comparator.create_visualizations()
    comparator.save_results()
    comparator.print_final_report()


if __name__ == "__main__":
    main()
