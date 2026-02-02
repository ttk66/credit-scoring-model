import onnx
m = onnx.load("models/nn_model.onnx", load_external_data=True)
onnx.save_model(m, "models/nn_model.onnx", save_as_external_data=False)
print("saved single-file ONNX")