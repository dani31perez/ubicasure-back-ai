from dotenv import load_dotenv
load_dotenv()

import os
from collections import defaultdict
from io import BytesIO

from fastapi import FastAPI, HTTPException, Header
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from PIL import Image
from ultralytics import YOLO, YOLOE
import httpx
import torch
import google.auth
import google.auth.transport.requests


DEVICE = "0" if torch.cuda.is_available() else "cpu"
INCIDENT_CONF = 0.35
ANALYSIS_CONF = 0.20
IMAGE_SIZE = 640
AI_SERVICE_SECRET = os.environ.get("AI_SERVICE_SECRET")

_credentials, _ = google.auth.default(
    scopes=["https://www.googleapis.com/auth/devstorage.read_only"]
)

def get_gcs_auth_header():
    if not _credentials.valid:
        _credentials.refresh(google.auth.transport.requests.Request())
    return {"Authorization": f"Bearer {_credentials.token}"}

INCIDENT_CLASSES = [
    "vehicle collision", "fire", "smoke", "damaged vehicle",
    "severely damaged vehicle", "overturned vehicle", "fallen person",
    "person on ground", "weapon", "knife", "handgun", "rifle",
    "fallen tree", "fallen pole", "damaged pole", "fallen power line",
    "damaged building", "collapsed building", "flood", "flooded road",
    "landslide", "debris", "blocked road",
]

ANALYSIS_CLASSES = [
    "car", "motorcycle", "bus", "truck", "bicycle", "vehicle collision",
    "damaged vehicle", "severely damaged vehicle", "overturned vehicle",
    "fire", "smoke", "person", "fallen person", "person on ground",
    "visible burn", "weapon", "knife", "handgun", "rifle",
    "person with weapon", "fallen tree", "fallen pole", "damaged pole",
    "power line", "fallen power line", "damaged building",
    "collapsed building", "damaged storefront", "broken window", "flood",
    "flooded road", "landslide", "debris", "blocked road",
]

SPANISH = {
    "car": "carro", "motorcycle": "moto", "bus": "bus", "truck": "camión",
    "bicycle": "bicicleta", "vehicle collision": "choque de vehículos",
    "damaged vehicle": "vehículo con daño visible",
    "severely damaged vehicle": "vehículo severamente dañado",
    "overturned vehicle": "vehículo volcado", "fire": "fuego", "smoke": "humo",
    "person": "persona", "fallen person": "persona caída",
    "person on ground": "persona en el suelo",
    "visible burn": "posible quemadura visible", "weapon": "arma",
    "knife": "cuchillo", "handgun": "arma corta", "rifle": "rifle",
    "person with weapon": "persona con arma", "fallen tree": "árbol caído",
    "fallen pole": "poste caído", "damaged pole": "poste dañado",
    "power line": "cable eléctrico", "fallen power line": "cable eléctrico caído",
    "damaged building": "edificio/estructura dañada",
    "severely damaged building": "edificio/estructura severamente dañado",
    "collapsed building": "edificio/estructura colapsada",
    "damaged storefront": "local/fachada dañada",
    "severely damaged storefront": "local/fachada severamente dañada",
    "broken window": "ventana rota", "flood": "inundación",
    "flooded road": "carretera inundada", "landslide": "derrumbe/deslizamiento",
    "debris": "escombros", "blocked road": "carretera bloqueada",
}

DAMAGE_LEVEL = {
    "damaged vehicle": "visible", "severely damaged vehicle": "severo",
    "overturned vehicle": "severo", "damaged pole": "visible",
    "fallen pole": "severo", "fallen power line": "severo",
    "damaged building": "visible", "severely damaged building": "severo",
    "collapsed building": "severo", "damaged storefront": "visible",
    "severely damaged storefront": "severo", "broken window": "visible",
}

print("Cargando YOLO26n...")
vehicle_model = YOLO("yolo26n.pt")

print("Cargando YOLOE-26n...")
incident_model = YOLOE("yoloe-26n-seg.pt")
incident_model.set_classes(INCIDENT_CLASSES)

analysis_model = YOLOE("yoloe-26n-seg.pt")
analysis_model.set_classes(ANALYSIS_CLASSES)

print(f"Modelos cargados. Dispositivo: {DEVICE}")

app = FastAPI(title="Ubicasure AI Detection API", version="0.1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

class AnalyzeRequest(BaseModel):
    image_url: str

def collect_detections(result, allowed_names=None):
    detections = []
    if result.boxes is None:
        return detections
    for box in result.boxes:
        class_id = int(box.cls[0])
        confidence = float(box.conf[0])
        class_name = result.names[class_id]
        if allowed_names is not None and class_name not in allowed_names:
            continue
        detections.append({"class": class_name, "confidence": confidence})
    return detections

def summarize(detections):
    grouped = defaultdict(lambda: {"count": 0, "max_confidence": 0.0})
    for item in detections:
        name = item["class"]
        grouped[name]["count"] += 1
        grouped[name]["max_confidence"] = max(
            grouped[name]["max_confidence"], item["confidence"]
        )

    objects = []
    damage = []
    for class_name, values in grouped.items():
        item = {
            "type": class_name,
            "label": SPANISH.get(class_name, class_name),
            "count": values["count"],
            "max_confidence": round(values["max_confidence"], 4),
        }
        if class_name in DAMAGE_LEVEL:
            item["damage_level"] = DAMAGE_LEVEL[class_name]
            damage.append(item)
        else:
            objects.append(item)

    objects.sort(key=lambda x: x["max_confidence"], reverse=True)
    damage.sort(key=lambda x: x["max_confidence"], reverse=True)
    return objects, damage

def detect_incident(image):
    results = incident_model.predict(
        source=image, conf=INCIDENT_CONF, imgsz=IMAGE_SIZE,
        device=DEVICE, verbose=False,
    )
    detections = collect_detections(results[0], allowed_names=set(INCIDENT_CLASSES))

    if not detections:
        return {"detected": False, "confidence": 0.0, "detections": []}

    strongest_detection = max(detections, key=lambda x: x["confidence"])

    return {
        "detected": True,
        "confidence": round(strongest_detection["confidence"], 4),
        "evidence": {
            "type": strongest_detection["class"],
            "label": strongest_detection.get(
                "label",
                SPANISH.get(strongest_detection["class"], strongest_detection["class"]),
            ),
        },
        "detections": summarize(detections),
    }

def analyze_incident(image):
    vehicle_results = vehicle_model.predict(
        source=image, conf=ANALYSIS_CONF, imgsz=IMAGE_SIZE,
        device=DEVICE, verbose=False,
    )
    vehicle_detections = collect_detections(
        vehicle_results[0],
        allowed_names={"car", "motorcycle", "bus", "truck", "bicycle"},
    )

    analysis_results = analysis_model.predict(
        source=image, conf=ANALYSIS_CONF, imgsz=IMAGE_SIZE,
        device=DEVICE, verbose=False,
    )
    incident_detections = collect_detections(
        analysis_results[0], allowed_names=set(ANALYSIS_CLASSES)
    )

    all_detections = vehicle_detections + incident_detections

    return {
        "detections": summarize(all_detections),
        "raw_detections": len(all_detections),
    }

@app.get("/")
def root():
    return {
        "service": "Ubicasure AI Detection API",
        "status": "running",
        "device": "cuda" if torch.cuda.is_available() else "cpu",
        "endpoint": "POST /analyze",
        "expects": "json body with image_url",
    }

@app.get("/health")
def health():
    return {"ok": True, "device": "cuda" if torch.cuda.is_available() else "cpu"}

@app.post("/analyze")
async def analyze_image(payload: AnalyzeRequest, x_api_key: str = Header(None)):
    if AI_SERVICE_SECRET and x_api_key != AI_SERVICE_SECRET:
        raise HTTPException(status_code=401, detail="No autorizado.")

    try:
        headers = get_gcs_auth_header()
        async with httpx.AsyncClient(timeout=15.0) as client:
            response = await client.get(payload.image_url, headers=headers)
            response.raise_for_status()
        image = Image.open(BytesIO(response.content)).convert("RGB")
    except Exception as e:
        raise HTTPException(
            status_code=400,
            detail="No se pudo descargar o abrir la imagen desde la URL proporcionada.",
        )

    incident = detect_incident(image)

    if not incident["detected"]:
        return {
            "success": True,
            "incident_detected": False,
            "incident": incident,
            "analysis": None,
            "message": "No se encontró evidencia visual suficiente de un incidente.",
            "device": "cuda" if torch.cuda.is_available() else "cpu",
        }

    analysis = analyze_incident(image)

    return {
        "success": True,
        "incident_detected": True,
        "incident": incident,
        "analysis": analysis,
        "device": "cuda" if torch.cuda.is_available() else "cpu",
        "warning": (
            "El análisis visual es experimental y no constituye una "
            "evaluación pericial, médica o policial."
        ),
    }