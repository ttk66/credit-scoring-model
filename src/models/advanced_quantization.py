import onnx
import onnxruntime as ort
import numpy as np
from pathlib import Path
import time
import io

class ONNXQuantizer:
    """Продвинутое квантование ONNX моделей"""
    
    def __init__(self, model_path: Path):
        self.model_path = model_path
        self.model = onnx.load(str(model_path))
        
    def analyze_model(self):
        """Анализ модели для квантования"""
        print("Analyzing model for quantization...")
        
        # Анализ операций
        op_types = {}
        for node in self.model.graph.node:
            op_types[node.op_type] = op_types.get(node.op_type, 0) + 1
        
        print("Operation types in model:")
        for op_type, count in sorted(op_types.items()):
            print(f"  {op_type}: {count}")
        
        return op_types
    
    def quantize_to_int8(self, output_path: Path, calibration_data: np.ndarray = None):
        """Квантование модели в INT8"""
        import onnxruntime.quantization as quant
        
        # Создаем DataReader для калибровки
        class DataReader(quant.CalibrationDataReader):
            def __init__(self, data):
                self.data = data
                self.index = 0
            
            def get_next(self):
                if self.index >= len(self.data):
                    return None
                
                sample = self.data[self.index]
                self.index += 1
                return {'input': sample.astype(np.float32)}
        
        # Конфигурация квантования
        quantization_config = quant.QuantizationConfig(
            is_static=True,
            format=quant.QuantFormat.QDQ,
            mode=quant.QuantizationMode.IntegerOps,
            activations_type=quant.QuantType.QInt8,
            weights_type=quant.QuantType.QInt8,
            calibrate_method=quant.CalibrationMethod.MinMax,
            per_channel=False,
            reduce_range=False
        )
        
        # Квантование
        if calibration_data is not None:
            data_reader = DataReader(calibration_data)
            quant.quantize_static(
                model_input=str(self.model_path),
                model_output=str(output_path),
                calibration_data_reader=data_reader,
                quant_config=quantization_config
            )
        else:
            # Используем динамическое квантование
            quant.quantize_dynamic(
                model_input=str(self.model_path),
                model_output=str(output_path),
                weight_type=quant.QuantType.QInt8
            )
        
        return output_path
    
    def quantize_to_float16(self, output_path: Path):
        """Квантование модели в Float16"""
        import onnxconverter_common
        
        # Конвертация в Float16
        fp16_model = onnxconverter_common.convert_float_to_float16(self.model)
        
        # Сохранение
        onnx.save(fp16_model, str(output_path))
        
        return output_path
    
    def compare_quantization_modes(self, test_data: np.ndarray, n_iterations: int = 100):
        """Сравнение разных режимов квантования"""
        
        results = {}
        
        # Оригинальная модель (FP32)
        print("\nTesting original FP32 model...")
        session_fp32 = ort.InferenceSession(str(self.model_path))
        time_fp32 = self._benchmark_inference(session_fp32, test_data, n_iterations)
        results['fp32'] = {'time_ms': time_fp32, 'size_mb': self._get_model_size()}
        
        # INT8 квантованная модель
        print("\nTesting INT8 quantized model...")
        int8_path = self.model_path.parent / f"{self.model_path.stem}_int8.onnx"
        self.quantize_to_int8(int8_path, test_data[:100])
        
        session_int8 = ort.InferenceSession(str(int8_path))
        time_int8 = self._benchmark_inference(session_int8, test_data, n_iterations)
        results['int8'] = {'time_ms': time_int8, 'size_mb': self._get_model_size(int8_path)}
        
        # Float16 квантованная модель
        print("\nTesting Float16 quantized model...")
        fp16_path = self.model_path.parent / f"{self.model_path.stem}_fp16.onnx"
        self.quantize_to_float16(fp16_path)
        
        session_fp16 = ort.InferenceSession(str(fp16_path))
        time_fp16 = self._benchmark_inference(session_fp16, test_data, n_iterations)
        results['fp16'] = {'time_ms': time_fp16, 'size_mb': self._get_model_size(fp16_path)}
        
        return results
    
    def _benchmark_inference(self, session: ort.InferenceSession, 
                           test_data: np.ndarray, n_iterations: int) -> float:
        """Бенчмарк инференса"""
        input_name = session.get_inputs()[0].name
        
        # Warmup
        for _ in range(10):
            session.run(None, {input_name: test_data[:1].astype(np.float32)})
        
        # Измерение времени
        start_time = time.time()
        for _ in range(n_iterations):
            session.run(None, {input_name: test_data[:1].astype(np.float32)})
        
        total_time = time.time() - start_time
        return total_time / n_iterations * 1000
    
    def _get_model_size(self, model_path: Path = None) -> float:
        """Получение размера модели в MB"""
        if model_path is None:
            model_path = self.model_path
        
        return model_path.stat().st_size / (1024 * 1024)