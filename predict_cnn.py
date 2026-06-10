"""
Inferencia foto -> peso (kg) con CNN.
Pipeline: YOLOv8-seg detecta la vaca -> bbox -> crop con margen -> EfficientNet-B0 -> peso.
Uso: python predict_cnn.py vaca.jpg [M|F]
"""
import sys, math
import numpy as np
from PIL import Image
import torch, torch.nn as nn
from torchvision import transforms as T
from torchvision.models import efficientnet_b0
from ultralytics import YOLO

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
CKPT = "modelo_peso_cnn.pt"
COW = 19  # clase 'cow' en COCO
_seg = None


class WeightNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = efficientnet_b0(weights=None).features
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.head = nn.Sequential(
            nn.Linear(1280 + 1, 256), nn.SiLU(), nn.Dropout(0.3),
            nn.Linear(256, 64), nn.SiLU(), nn.Dropout(0.2),
            nn.Linear(64, 1))

    def forward(self, x, sexo):
        f = self.pool(self.features(x)).flatten(1)
        return self.head(torch.cat([f, sexo], 1))


_ck = torch.load(CKPT, map_location=DEVICE, weights_only=False)
_model = WeightNet().to(DEVICE)
_model.load_state_dict(_ck["state_dict"])
_model.eval()
IMG = _ck["img"]; YMEAN = _ck["ymean"]; YSTD = _ck["ystd"]
_tf = T.Compose([T.Resize((IMG, IMG)), T.ToTensor(),
                 T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])


def cow_bbox(img_rgb):
    """bbox (l,t,r,b) de la vaca mas grande, o None."""
    global _seg
    if _seg is None:
        _seg = YOLO("yolov8n-seg.pt")
    res = _seg.predict(img_rgb, verbose=False, device=0 if DEVICE == "cuda" else "cpu")[0]
    best, ba = None, 0
    for box in res.boxes:
        if int(box.cls) == COW:
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            a = (x2 - x1) * (y2 - y1)
            if a > ba:
                ba, best = a, (x1, y1, x2, y2)
    return best


def predict(path, sexo="F"):
    img = Image.open(path).convert("RGB")
    bb = cow_bbox(np.array(img))
    if bb is None:
        return {"error": "No se detecto ninguna vaca en la imagen."}
    l, t, r, b = bb
    mx, my = 0.10 * (r - l), 0.10 * (b - t)
    crop = img.crop((max(0, int(l - mx)), max(0, int(t - my)),
                     min(img.width, int(r + mx)), min(img.height, int(b + my))))
    x = _tf(crop).unsqueeze(0).to(DEVICE)
    s = torch.tensor([[1.0 if sexo.upper().startswith("M") else 0.0]], device=DEVICE)
    with torch.no_grad(), torch.autocast("cuda", enabled=DEVICE == "cuda"):
        z = _model(x, s).item()
    peso = z * YSTD + YMEAN
    return {"peso_estimado_kg": round(peso, 1),
            "bbox": [round(v) for v in bb],
            "nota": f"CNN EfficientNet-B0. Val MAE~{_ck['val_mae']:.0f}kg, MAPE~{_ck['val_mape']:.0f}%."}


if __name__ == "__main__":
    p = sys.argv[1] if len(sys.argv) > 1 else "vaca.jpg"
    sx = sys.argv[2] if len(sys.argv) > 2 else "F"
    print(predict(p, sx))
