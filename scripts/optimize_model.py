#!/usr/bin/env python3
"""
Скрипт для оптимизации модели через pruning и quantization
"""

import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT_DIR))

import argparse
import json
from src.models.quantization import run_optimization_pipeline
from src.models.advanced_quantization import ONNXQuantizer

def main():
    parser = argparse.ArgumentParser(description="Optimize model with pruning and quantization")
    parser.add_argument("--technique", choices=['all', 'pruning', 'quantization', 'compare'], 
                       default='all', help="Optimization technique to apply")
    parser.add_argument("--pruning-amount", type=float, default=0.3, 
                       help="Amount of pruning (0.0 to 1.0)")
    parser.add_argument("--quantization-type", choices=['int8', 'float16', 'dynamic'], 
                       default='int8', help="Type of quantization")
    parser.add_argument("--onnx-model", type=str, default="models/nn_model.onnx",
                       help="Path to ONNX model for quantization")
    
    args = parser.parse_args()
    
    print("="*60)
    print("MODEL OPTIMIZATION SCRIPT")
    print("="*60)
    
    if args.technique in ['all', 'pruning', 'quantization']:
        print(f"\nRunning optimization pipeline...")
        print(f"  Technique: {args.technique}")
        print(f"  Pruning amount: {args.pruning_amount}")
        print(f"  Quantization type: {args.quantization_type}")
        
        results = run_optimization_pipeline()
        
        print(f"\nOptimization completed!")
        
        # Сохраняем краткий отчет
        report = {
            'technique': args.technique,
            'pruning_amount': args.pruning_amount,
            'quantization_type': args.quantization_type,
            'results_summary': {
                'speedup': results['metrics']['improvement']['speedup'],
                'size_reduction': results['metrics']['improvement']['size_reduction'],
                'auc_change': results['metrics']['improvement']['auc_change']
            }
        }
        
        with open('models/optimization_report.json', 'w') as f:
            json.dump(report, f, indent=2)
        
        print(f"Report saved to: models/optimization_report.json")
    
    if args.technique == 'compare':
        print("\nComparing quantization modes...")
        
        # Загрузка тестовых данных для калибровки
        from src.models.train import load_data
        import joblib
        
        X, y = load_data()
        scaler = joblib.load(Path("models/nn_scaler.joblib"))
        X_scaled = scaler.transform(X[:100])
        
        # Сравнение режимов квантования
        quantizer = ONNXQuantizer(Path(args.onnx_model))
        results = quantizer.compare_quantization_modes(X_scaled, n_iterations=100)
        
        print("\nQuantization Comparison Results:")
        print("-"*40)
        for mode, data in results.items():
            print(f"{mode.upper()}:")
            print(f"  Time per inference: {data['time_ms']:.2f} ms")
            print(f"  Model size: {data['size_mb']:.2f} MB")
        
        # Сохраняем результаты сравнения
        with open('models/quantization_comparison.json', 'w') as f:
            json.dump(results, f, indent=2)
        
        print(f"\nResults saved to: models/quantization_comparison.json")
    
    print("\n" + "="*60)
    print("OPTIMIZATION PROCESS COMPLETED")
    print("="*60)

if __name__ == "__main__":
    main()