"""
export_onnx.py — Exports trained DualViewCNN to ONNX format.
"""
import torch
import config
from model import DualViewCNN

def export():
    device = torch.device("cpu")

    # 1. Load trained PyTorch model
    model = DualViewCNN().to(device)
    state_dict = torch.load(config.DATA_DIR / "best_model.pt", map_location=device)
    model.load_state_dict(state_dict)
    model.eval()

    # 2. Match exact dataset shapes [batch_size, sequence_length] (2D tensors)
    # DualViewCNN internal forward() will add the channel dimension automatically
    dummy_global = torch.randn(1, 2001, dtype=torch.float32)
    dummy_local = torch.randn(1, 201, dtype=torch.float32)

    onnx_output_path = "model.onnx"

    # 3. Export using legacy tracer (dynamo=False) to avoid Python 3.14 torch.export strictness
    torch.onnx.export(
        model,
        (dummy_global, dummy_local),
        onnx_output_path,
        export_params=True,
        opset_version=14,
        do_constant_folding=True,
        input_names=["global_view", "local_view"],
        output_names=["logit"],
        dynamic_axes={
            "global_view": {0: "batch_size"},
            "local_view": {0: "batch_size"},
            "logit": {0: "batch_size"},
        },
        dynamo=False,
    )
    print(f"Export successful -> {onnx_output_path}")

if __name__ == "__main__":
    export()