# BovweightML

API FastAPI que estima el peso de una vaca a partir de una foto lateral.
Pipeline: **YOLOv8-seg** detecta la vaca → **EfficientNet-B0** estima el peso.

> Modelo: `modelo_peso_cnn.pt` (entrenado en dataset 12k, side view).
> Precisión validación: MAE ~24 kg, R² ~0.42, MAPE ~15%.

---

## Requisitos

- **Python 3.10 o superior** ([descargar](https://www.python.org/downloads/))
- **Git** ([descargar](https://git-scm.com/downloads))
- ~3 GB libres en disco (modelos + dependencias)
- (Opcional) GPU NVIDIA con CUDA para mayor velocidad. Sin GPU corre en CPU.

---

## Pasos para correrlo

### 1. Clonar el repositorio y entrar a la carpeta

```bash
git clone <URL-del-repo>
cd ProyectoBovweight/BovweightML
```

### 2. Verificar que Python esté instalado

```bash
python --version
```

Debe mostrar `Python 3.10` o superior. Si no, instálalo desde [python.org](https://www.python.org/downloads/).

### 3. Crear el entorno virtual

```bash
python -m venv venv
```

Esto crea la carpeta `venv/` con un Python aislado del sistema.

### 4. Activar el entorno virtual

**Windows (PowerShell):**
```powershell
venv\Scripts\Activate.ps1
```

> Si PowerShell bloquea el script con `running scripts is disabled`, ejecuta una sola vez:
> ```powershell
> Set-ExecutionPolicy -Scope CurrentUser -ExecutionPolicy RemoteSigned
> ```
> y vuelve a activar.

**Windows (CMD):**
```cmd
venv\Scripts\activate.bat
```

**Linux / macOS:**
```bash
source venv/bin/activate
```

Sabrás que está activado porque el prompt empieza con `(venv)`.

### 5. Actualizar pip (recomendado)

```bash
python -m pip install --upgrade pip
```

### 6. Instalar dependencias

```bash
pip install -r requirements.txt
```

Tarda varios minutos (PyTorch + Ultralytics pesan).

### 7. Levantar la API

```bash
uvicorn main:app --host 127.0.0.1 --port 8001 --reload
```

La API queda escuchando en `http://127.0.0.1:8001`.

> El modelo `modelo_peso_cnn.pt` ya viene incluido en el repositorio.
> El modelo `yolov8n-seg.pt` se descarga solo la primera vez que arranca.

### 8. Verificar que funciona

Abre en el navegador:
- `http://127.0.0.1:8001/` → debe responder con un JSON con `"mensaje": "API Bovweight CNN lista."`
- `http://127.0.0.1:8001/health` → estado + métricas de validación

---

## Cómo volver a correrlo en sesiones siguientes

Ya no hay que crear el venv ni reinstalar. Solo:

```bash
cd ProyectoBovweight/BovweightML
venv\Scripts\Activate.ps1                 # (Windows PowerShell)
uvicorn main:app --host 127.0.0.1 --port 8001 --reload
```

Para salir del venv cuando termines:
```bash
deactivate
```

---

## Probar `/predict` desde terminal

```bash
curl -X POST http://127.0.0.1:8001/predict \
  -F "file=@ruta/a/foto-vaca.jpg" \
  -F "sexo=F"
```

Respuesta esperada:
```json
{
  "peso_estimado_kg": 175.5,
  "bbox": [68, 38, 1841, 1360],
  "nota": "Estimacion CNN (~MAPE 15%). Foto lateral, vaca completa, buena luz."
}
```

---

## Integración con el backend Laravel

El backend (`BovweightBackend`) lo consume vía la variable `ML_API_URL` en su `.env`:

```
ML_API_URL=http://127.0.0.1:8001
```

Levanta primero esta API y luego el backend.

---

## Problemas frecuentes

| Síntoma | Causa / solución |
|---|---|
| `ModuleNotFoundError: torch` | No activaste el venv. Activa con paso 3 |
| Tarda mucho la primera predicción | Normal: YOLO descarga `yolov8n-seg.pt` la primera vez |
| `CUDA out of memory` | Sin GPU suficiente. Pon `DEVICE = "cpu"` en `main.py` |
| Puerto 8001 ocupado | Cambia con `--port 8002` y actualiza `ML_API_URL` |
