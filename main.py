"""
API BovweightML - Peso por foto lateral (CNN, sin sticker).
Pipeline:
  1) YOLOv8-seg detecta la vaca -> bbox.
  2) Crop a la bbox (margen 10%) -> EfficientNet-B0 (foto + sexo) -> peso (kg).

Modelo: modelo_peso_cnn.pt  (entrenado en dataset 12k, side view).
Precision validacion: MAE ~24 kg, R2 ~0.42, MAPE ~15%.
"""
import io
import numpy as np
from PIL import Image
from fastapi import FastAPI, UploadFile, File, Form
import torch
import torch.nn as nn
from torchvision import transforms as T
from torchvision.models import efficientnet_b0
from ultralytics import YOLO

app = FastAPI(title="API BovweightML - Peso CNN")

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
COW = 19  # clase 'cow' en COCO
CKPT = "modelo_peso_cnn.pt"


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
IMG, YMEAN, YSTD = _ck["img"], _ck["ymean"], _ck["ystd"]
def _a_float(v):
    # Las metricas del checkpoint pueden ser numpy.float32 y FastAPI/pydantic
    # no las serializa a JSON. Convertimos a float nativo.
    return float(v) if v is not None else None

_VAL = {
    "mae": _a_float(_ck.get("val_mae")),
    "r2": _a_float(_ck.get("val_r2")),
    "mape": _a_float(_ck.get("val_mape")),
}

_tf = T.Compose([T.Resize((IMG, IMG)), T.ToTensor(),
                 T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
_seg = YOLO("yolov8n-seg.pt")


def cow_bbox(img_rgb):
    res = _seg.predict(img_rgb, verbose=False, device=0 if DEVICE == "cuda" else "cpu")[0]
    best, ba = None, 0
    for box in res.boxes:
        if int(box.cls) == COW:
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            a = (x2 - x1) * (y2 - y1)
            if a > ba:
                ba, best = a, (x1, y1, x2, y2)
    return best


def estimar(img, sexo_str):
    bb = cow_bbox(np.array(img))
    if bb is None:
        return {"error": "No se detecto ninguna vaca en la imagen."}
    l, t, r, b = bb
    mx, my = 0.10 * (r - l), 0.10 * (b - t)
    crop = img.crop((max(0, int(l - mx)), max(0, int(t - my)),
                     min(img.width, int(r + mx)), min(img.height, int(b + my))))
    x = _tf(crop).unsqueeze(0).to(DEVICE)
    s = torch.tensor([[1.0 if sexo_str.upper().startswith("M") else 0.0]], device=DEVICE)
    with torch.no_grad(), torch.autocast("cuda", enabled=DEVICE == "cuda"):
        z = _model(x, s).item()
    peso = z * YSTD + YMEAN
    return {
        "peso_estimado_kg": round(peso, 1),
        "bbox": [round(v) for v in bb],
        "nota": "Estimacion CNN (~MAPE 15%). Foto lateral, vaca completa, buena luz.",
    }


@app.get("/")
def home():
    return {"mensaje": "API Bovweight CNN lista.", "device": DEVICE, "modelo": _ck.get("arch")}


@app.get("/health")
def health():
    return {"ok": True, "device": DEVICE, "val": _VAL}


@app.post("/predict")
async def predict(file: UploadFile = File(...), sexo: str = Form("F", description="M o F")):
    try:
        raw = await file.read()
        img = Image.open(io.BytesIO(raw)).convert("RGB")
        return estimar(img, sexo)
    except Exception as e:
        return {"error": f"Problema en el servidor: {str(e)}"}
