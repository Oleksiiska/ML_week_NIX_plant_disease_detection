import io
from PIL import Image
import torch
import torchvision.transforms as T


def get_device() -> torch.device:
    # Apple Silicon: mps, інакше cpu (CUDA на macOS зазвичай нема)
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def build_transform(image_size: int = 224) -> T.Compose:
    # Базовий трансформ як для torchvision-класифікацій
    return T.Compose([
        T.Resize((image_size, image_size)),
        T.ToTensor(),
        T.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    ])


def load_image_to_tensor(image_bytes: bytes, transform: T.Compose, device: torch.device) -> torch.Tensor:
    img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    x = transform(img).unsqueeze(0)  # [1, C, H, W]
    return x.to(device)


def softmax_topk(logits: torch.Tensor, k: int = 3):
    probs = torch.softmax(logits, dim=1)
    top_probs, top_idx = torch.topk(probs, k=min(k, probs.shape[1]), dim=1)
    return top_probs[0].detach().cpu().tolist(), top_idx[0].detach().cpu().tolist()
