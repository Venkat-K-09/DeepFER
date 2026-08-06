import torch
from src.models import build_model

def load_model(model_path: str, model_type: str = 'custom', backbone_name: str = 'resnet18', device: str = None):
    """Load a model checkpoint and return a ready‑to‑use model.

    Args:
        model_path: Path to the .pth checkpoint file.
        model_type: 'custom' or 'transfer'.
        backbone_name: Backbone name used for transfer models.
        device: Torch device string (e.g., 'cuda' or 'cpu'). If None, auto‑detect.
    """
    dev = device or ('cuda' if torch.cuda.is_available() else 'cpu')
    model = build_model(model_type=model_type, backbone_name=backbone_name).to(dev)
    ckpt = torch.load(model_path, map_location=dev)
    state_dict = ckpt.get('state_dict', ckpt)
    model.load_state_dict(state_dict)
    model.eval()
    return model

if __name__ == '__main__':
    # Simple sanity check
    import os
    chk = 'models/best_custom_cnn.pth'
    if os.path.exists(chk):
        mdl = load_model(chk)
        print('Model loaded successfully, device:', next(mdl.parameters()).device)
    else:
        print('Checkpoint not found:', chk)
