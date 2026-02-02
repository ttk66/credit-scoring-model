import numpy
import torch
import onnx
import onnxruntime as ort
import numpy as np
import time
from pathlib import Path
import joblib
from typing import Dict, Tuple

MODEL_PATH = Path("models/nn_model.pth")
SCALER_PATH = Path("models/nn_scaler.joblib")
ONNX_PATH = Path("models/nn_model.onnx")

#      

torch.serialization.add_safe_globals([numpy._core.multiarray.scalar])


class CreditScoringNN(torch.nn.Module):
    """    """

    def __init__(self, input_size: int, dropout_rate: float = 0.3):
        super(CreditScoringNN, self).__init__()

        self.network = torch.nn.Sequential(
            torch.nn.Linear(input_size, 32),
            torch.nn.ReLU(),
            torch.nn.Dropout(dropout_rate),
            torch.nn.Linear(32, 16),
            torch.nn.ReLU(),
            torch.nn.Dropout(dropout_rate),
            torch.nn.Linear(16, 8),
            torch.nn.ReLU(),
            torch.nn.Linear(8, 1),
            torch.nn.Sigmoid(),
        )

    def forward(self, x):
        return self.network(x)


def load_pytorch_model(model_path: Path) -> Tuple[torch.nn.Module, Dict]:
    """ PyTorch     """
    print(f"Loading PyTorch model from {model_path}...")

    try:
        #   weights_only=False
        checkpoint = torch.load(
            model_path, map_location=torch.device("cpu"), weights_only=False
        )
        print("Model loaded with weights_only=False")
    except Exception as e:
        print(f"First attempt failed: {e}")
        try:
            #   
            if "input_size" in locals() or "input_size" in globals():
                input_size = checkpoint.get("input_size", 32)
            else:
                #   input_size   
                import pickle

                with open(model_path, "rb") as f:
                    temp_checkpoint = pickle.load(f)
                    input_size = temp_checkpoint.get("input_size", 32)

            model = CreditScoringNN(input_size=input_size)

            #   state_dict
            state_dict = torch.load(model_path, map_location="cpu")["model_state_dict"]
            model.load_state_dict(state_dict)
            model.eval()

            checkpoint = {"model_state_dict": state_dict, "input_size": input_size}
            print("Model loaded with manual state_dict loading")
        except Exception as e2:
            print(f"All loading methods failed: {e2}")
            raise

    input_size = checkpoint.get("input_size", 32)
    model = CreditScoringNN(input_size=input_size)

    #  
    if "model_state_dict" in checkpoint:
        model.load_state_dict(checkpoint["model_state_dict"])
    else:
        model = checkpoint.get("model", model)

    model.eval()

    return model, checkpoint


def convert_to_onnx(
    pytorch_model: torch.nn.Module,
    input_size: int,
    onnx_path: Path,
    opset_version: int = 13,
):
    """ PyTorch   ONNX """

    dummy_input = torch.randn(1, input_size, dtype=torch.float32)

    #   ONNX
    torch.onnx.export(
        pytorch_model,
        dummy_input,
        onnx_path,
        export_params=True,
        opset_version=opset_version,
        do_constant_folding=True,
        input_names=["input"],
        output_names=["output"],
        dynamic_axes={"input": {0: "batch_size"}, "output": {0: "batch_size"}},
        verbose=True,
    )

    print(f"Model converted to ONNX and saved to: {onnx_path}")

    #  ONNX 
    try:
        onnx_model = onnx.load(onnx_path)
        onnx.checker.check_model(onnx_model)
        print("ONNX model validation passed")
    except Exception as e:
        print(f"ONNX validation warning: {e}")

    return onnx_model


def create_onnx_session(onnx_path: Path) -> ort.InferenceSession:
    """ ONNX Runtime """
    if not onnx_path.exists():
        raise FileNotFoundError(f"ONNX model not found: {onnx_path}")

    providers = ["CPUExecutionProvider"]
    session = ort.InferenceSession(str(onnx_path), providers=providers)
    input_name = session.get_inputs()[0].name
    output_name = session.get_outputs()[0].name

    print(f"ONNX Model Info:")
    print(f"  Input name: {input_name}")
    print(f"  Input shape: {session.get_inputs()[0].shape}")
    print(f"  Output name: {output_name}")
    print(f"  Output shape: {session.get_outputs()[0].shape}")

    return session


def validate_conversion(
    pytorch_model: torch.nn.Module,
    onnx_session: ort.InferenceSession,
    test_data: np.ndarray,
    tolerance: float = 1e-4,
):
    """  """

    print("\n" + "=" * 60)
    print("CONVERSION VALIDATION")
    print("=" * 60)

    #    PyTorch
    pytorch_model.eval()
    with torch.no_grad():
        pytorch_output = pytorch_model(torch.FloatTensor(test_data)).numpy()

    #    ONNX
    input_name = onnx_session.get_inputs()[0].name
    onnx_output = onnx_session.run(None, {input_name: test_data.astype(np.float32)})[0]

    #  
    abs_diff = np.abs(pytorch_output - onnx_output)
    max_diff = np.max(abs_diff)
    mean_diff = np.mean(abs_diff)

    print(f"Max absolute difference: {max_diff:.10f}")
    print(f"Mean absolute difference: {mean_diff:.10f}")

    if max_diff < tolerance:
        print("Conversion validation PASSED")
        return True
    else:
        print(f"Conversion validation warning (tolerance: {tolerance})")
        print(f"PyTorch output: {pytorch_output.flatten()[:5]}")
        print(f"ONNX output:    {onnx_output.flatten()[:5]}")
        return False


def compare_performance(
    pytorch_model: torch.nn.Module,
    onnx_session: ort.InferenceSession,
    test_data: np.ndarray,
    n_iterations: int = 100,
):
    """  PyTorch  ONNX """

    print("\n" + "=" * 60)
    print("PERFORMANCE COMPARISON")
    print("=" * 60)

    #  
    pytorch_input = torch.FloatTensor(test_data)
    onnx_input = {onnx_session.get_inputs()[0].name: test_data.astype(np.float32)}

    #  PyTorch
    print("\nPyTorch Inference:")
    print("-" * 40)

    # Warmup
    for _ in range(10):
        with torch.no_grad():
            _ = pytorch_model(pytorch_input[:1])

    #  
    start_time = time.time()

    for i in range(n_iterations):
        with torch.no_grad():
            _ = pytorch_model(pytorch_input[:1])

    torch_time = time.time() - start_time
    print(f"Total time for {n_iterations} iterations: {torch_time:.4f} seconds")
    print(f"Time per inference: {(torch_time / n_iterations * 1000):.4f} ms")

    #  ONNX
    print("\nONNX Inference:")
    print("-" * 40)

    # Warmup
    for _ in range(10):
        _ = onnx_session.run(None, onnx_input)

    #  
    start_time = time.time()

    for i in range(n_iterations):
        _ = onnx_session.run(None, onnx_input)

    onnx_time = time.time() - start_time
    print(f"Total time for {n_iterations} iterations: {onnx_time:.4f} seconds")
    print(f"Time per inference: {(onnx_time / n_iterations * 1000):.4f} ms")

    # 
    print("\n" + "=" * 60)
    print("PERFORMANCE SUMMARY")
    print("=" * 60)

    if onnx_time > 0:
        speedup = torch_time / onnx_time
        print(
            f"ONNX is {
                speedup:.2f}x {
                'faster' if speedup > 1 else 'slower'} than PyTorch"
        )
    else:
        speedup = 0
        print("Could not calculate speedup")

    return {
        "pytorch_time_ms": (torch_time / n_iterations * 1000),
        "onnx_time_ms": (onnx_time / n_iterations * 1000),
        "speedup": speedup,
    }


def run_complete_pipeline():
    """    """

    print("=" * 60)
    print("ONNX CONVERSION PIPELINE")
    print("=" * 60)

    #  PyTorch 
    print("\nLoading PyTorch model...")
    try:
        pytorch_model, checkpoint = load_pytorch_model(MODEL_PATH)
        input_size = checkpoint.get("input_size", 32)
        print(f"Model loaded: input_size={input_size}")
    except Exception as e:
        print(f"Failed to load model: {e}")
        print("\nTrying alternative loading method...")

        #   
        try:
            #    
            pytorch_model = torch.jit.load(str(MODEL_PATH))
            input_size = 32  # 
            checkpoint = {"input_size": input_size}
            print(f"Model loaded with torch.jit")
        except BaseException:
            print("All loading methods failed")
            return None

    #      
    print("\n2. Preparing test data...")
    try:
        scaler = joblib.load(SCALER_PATH)

        #   
        n_test_samples = 100
        test_data = np.random.randn(n_test_samples, input_size)
        test_data_scaled = scaler.transform(test_data)

        print(f"Test data prepared: {test_data_scaled.shape}")
    except Exception as e:
        print(f"Could not load scaler: {e}")
        print("Using random test data without scaling")
        test_data_scaled = np.random.randn(10, input_size).astype(np.float32)

    #   ONNX
    print("\nConverting to ONNX...")
    try:
        onnx_model = convert_to_onnx(pytorch_model, input_size, ONNX_PATH)
        print("Conversion completed")
    except Exception as e:
        print(f"Conversion failed: {e}")
        return None

    #  ONNX Runtime 
    print("\nCreating ONNX Runtime session...")
    try:
        onnx_session = create_onnx_session(ONNX_PATH)
        print("ONNX session created")
    except Exception as e:
        print(f"Failed to create ONNX session: {e}")
        return None

    #  
    print("\n5. Validating conversion...")
    is_valid = validate_conversion(
        pytorch_model,
        onnx_session,
        test_data_scaled[:10],
        tolerance=1e-3,
    )

    #  
    print("\n6. Comparing performance...")
    try:
        performance = compare_performance(
            pytorch_model,
            onnx_session,
            test_data_scaled[:50],
            n_iterations=100,
        )
    except Exception as e:
        print(f"Performance comparison failed: {e}")
        performance = {"pytorch_time_ms": 0, "onnx_time_ms": 0, "speedup": 0}

    #  
    print("\n7. Saving results...")
    import json

    results = {
        "conversion_valid": is_valid,
        "performance": performance,
        "model_info": {
            "input_size": input_size,
            "onnx_opset": 13,
            "test_auc": checkpoint.get("test_auc", "N/A"),
            "file_size_mb": (
                ONNX_PATH.stat().st_size / (1024 * 1024) if ONNX_PATH.exists() else 0
            ),
        },
    }

    results_path = Path("models/onnx_conversion_results.json")
    results_path.parent.mkdir(parents=True, exist_ok=True)

    with open(results_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    print(f"Results saved to: {results_path}")

    #  
    print("\nAdditional checks...")

    # ,  ONNX  
    try:
        sample_input = np.random.randn(1, input_size).astype(np.float32)
        if scaler:
            sample_input = scaler.transform(sample_input)

        onnx_pred = onnx_session.run(
            None, {onnx_session.get_inputs()[0].name: sample_input}
        )[0]
        print(f"ONNX model inference test: output = {onnx_pred[0][0]:.6f}")
    except BaseException:
        print("Could not test ONNX inference")

    print("\n" + "=" * 60)
    print("CONVERSION PIPELINE COMPLETED")
    print("=" * 60)

    if is_valid:
        print("SUCCESS: ONNX conversion completed and validated!")
        if performance["speedup"] > 0:
            print(f"Speedup: {performance['speedup']:.2f}x")
    else:
        print("WARNING: Conversion completed but validation failed")

    return results


#    
def simple_convert():
    """    """
    print("=" * 60)
    print("SIMPLE ONNX CONVERSION")
    print("=" * 60)

    try:
        #  
        print("\n1. Loading model...")
        import pickle

        with open(MODEL_PATH, "rb") as f:
            checkpoint = pickle.load(f)

        input_size = checkpoint.get("input_size", 32)

        #  
        model = CreditScoringNN(input_size=input_size)

        if "model_state_dict" in checkpoint:
            model.load_state_dict(checkpoint["model_state_dict"])
        model.eval()

        print(f"Model loaded: input_size={input_size}")

        # 
        print("\nConverting to ONNX...")
        dummy_input = torch.randn(1, input_size, dtype=torch.float32)

        torch.onnx.export(
            model,
            dummy_input,
            ONNX_PATH,
            export_params=True,
            opset_version=13,
            input_names=["input"],
            output_names=["output"],
            verbose=False,
        )

        print(f"ONNX model saved: {ONNX_PATH}")

        #  
        print("\nSimple validation...")
        ort_session = ort.InferenceSession(str(ONNX_PATH))

        test_input = torch.randn(5, input_size)

        # PyTorch
        with torch.no_grad():
            pt_out = model(test_input).numpy()

        # ONNX
        ort_out = ort_session.run(
            None, {"input": test_input.numpy().astype(np.float32)}
        )[0]

        diff = np.abs(pt_out - ort_out).max()
        print(f"Max difference: {diff:.6f}")

        if diff < 0.001:
            print("Validation passed")
        else:
            print("Validation warning")

        return True

    except Exception as e:
        print(f"Error: {e}")
        import traceback

        traceback.print_exc()
        return False


if __name__ == "__main__":
    try:
        results = run_complete_pipeline()
        if not results or not results.get("conversion_valid", False):
            print("\nTrying simple conversion...")
            simple_convert()
    except Exception as e:
        print(f"\nMain pipeline failed: {e}")
        print("\nFalling back to simple conversion...")
        success = simple_convert()

        if success:
            print("\nSimple conversion successful!")
        else:
            print("\nAll conversion methods failed")
