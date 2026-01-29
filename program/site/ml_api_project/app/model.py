# app/model.py
import json
from pathlib import Path
from typing import Any, Dict, List

import torch
from torchvision import models

from app.utils import get_device, build_transform, load_image_to_tensor, softmax_topk


PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODEL_DIR = PROJECT_ROOT / "model"
MODEL_PATH = MODEL_DIR / "best_optimized_model.pth"
CLASSES_PATH = MODEL_DIR / "class_names.json"


class Predictor:
    def __init__(self):
        self.device = get_device()
        self.transform = build_transform(image_size=224)
        self.class_names = self._load_class_names(CLASSES_PATH)
        self.model = self._load_model(MODEL_PATH, num_classes=len(self.class_names))
        self.model.eval()

    def _load_class_names(self, path: Path) -> List[str]:
        """
        Supported formats:
        1) ["healthy","sick"]
        2) {"0":"healthy","1":"sick"}
        3) {"healthy":0,"sick":1}
        4) {"class_mapping":{"0":"healthy","1":"sick"}}  <- your current format
        """
        if not path.exists():
            raise FileNotFoundError(f"class_names.json not found: {path}")

        text = path.read_text(encoding="utf-8").strip()
        if not text:
            raise ValueError(f"class_names.json is empty: {path}")

        data = json.loads(text)

        # ✅ your format: {"class_mapping": {"0":"healthy","1":"sick"}}
        if isinstance(data, dict) and "class_mapping" in data and isinstance(data["class_mapping"], dict):
            m = data["class_mapping"]
            return [m[str(i)] for i in range(len(m))]

        # list: ["healthy","sick"]
        if isinstance(data, list):
            return data

        # dict index->name: {"0":"healthy","1":"sick"}
        if isinstance(data, dict) and all(str(k).isdigit() for k in data.keys()):
            return [data[str(i)] for i in range(len(data))]

        # dict name->index: {"healthy":0,"sick":1}
        if isinstance(data, dict) and all(isinstance(v, int) for v in data.values()):
            inv = {v: k for k, v in data.items()}
            return [inv[i] for i in range(len(inv))]

        raise ValueError("Unsupported class_names.json format")

    def _build_model(self, num_classes: int) -> torch.nn.Module:
        """
        Default architecture (change if your training architecture differs).
        If you trained e.g. efficientnet/mobilenet/custom, tell me and I'll adjust.
        """
        model = models.resnet18(weights=None)
        model.fc = torch.nn.Linear(model.fc.in_features, num_classes)
        return model

    def _load_model(self, path: Path, num_classes: int) -> torch.nn.Module:
        if not path.exists():
            raise FileNotFoundError(f"Model file not found: {path}")

        obj = torch.load(path, map_location="cpu")

        # Case A: whole model was saved with torch.save(model, ...)
        if isinstance(obj, torch.nn.Module):
            model = obj
        else:
            # Case B: state_dict or checkpoint dict
            state_dict = obj.get("state_dict", obj) if isinstance(obj, dict) else obj
            model = self._build_model(num_classes=num_classes)
            model.load_state_dict(state_dict, strict=False)

        model.to(self.device)
        return model

    @torch.inference_mode()
    def predict_bytes(self, image_bytes: bytes, topk: int = 3) -> Dict[str, Any]:
        x = load_image_to_tensor(image_bytes, self.transform, self.device)
        logits = self.model(x)

        probs, idxs = softmax_topk(logits, k=topk)
        best_idx = idxs[0]

        return {
            "label": self.class_names[best_idx],
            "confidence": float(probs[0]),
            "topk": [{"label": self.class_names[i], "confidence": float(p)} for p, i in zip(probs, idxs)],
            "device": str(self.device),
        }
