from flask import Flask, request, jsonify
import requests
import google.generativeai as genai
from flask_cors import CORS
import re
from deep_translator import GoogleTranslator
import os
from werkzeug.utils import secure_filename
import pandas as pd
import numpy as np
from statsmodels.tsa.statespace.sarimax import SARIMAX
from datetime import timedelta, datetime
import traceback
import joblib
import json
from geopy.geocoders import Nominatim
import random
import sqlite3
from sklearn.cluster import KMeans
from pathlib import Path  # NEW: for artifacts path

import base64
from io import BytesIO

import torch
from PIL import Image, UnidentifiedImageError
from transformers import ViTForImageClassification, ViTFeatureExtractor, ViTImageProcessor


app = Flask(__name__)
CORS(app)

# ------------------------------------------------------------------
# API Keys / Config
# ------------------------------------------------------------------
WEATHER_API_KEY = 'f28861253e574c589d5111924242807'
GEMINI_API_KEY = 'AIzaSyDNOtokPHTUm9WCJ1pOPaweUp_Rks9DhjI'
UNSPLASH_ACCESS_KEY = 'YAd-Af7cIyfplIFBCWaKRL1XiNE6VsFULmx-ln-_HfY'

# Ollama / Gemma (for crop explanations and disease guidance)
OLLAMA_URL = os.getenv("OLLAMA_URL", "http://127.0.0.1:11434")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "gemma3:4b")

DB_NAME = "complaints.db"

# Initialize Gemini Model (for chatbot, calendar, etc.)
genai.configure(api_key=GEMINI_API_KEY)
model = genai.GenerativeModel("gemini-2.5-flash")

# ---------------- Disease Detection: HF ViT + Gemma Vision ----------------
PRIMARY_REPO  = "wambugu71/crop_leaf_diseases_vit"
FALLBACK_REPO = "wambugu1738/crop_leaf_diseases_vit"
DEVICE        = "cpu"   # set to "cuda" if you run with GPU
TOPK          = 5
VISION_TIMEOUT_S = 120

SUPPORTED_CROPS = {"corn", "potato", "rice", "wheat"}

# Map HF labels to (crop, pretty disease name)
LABEL_PRETTY = {
    "Corn___Common_Rust":           ("corn",   "Common Rust"),
    "Corn___Gray_Leaf_Spot":        ("corn",   "Gray Leaf Spot"),
    "Corn___Healthy":               ("corn",   "Healthy"),
    "Corn___Northern_Leaf_Blight":  ("corn",   "Leaf Blight"),
    "Potato___Early_Blight":        ("potato", "Early Blight"),
    "Potato___Healthy":             ("potato", "Healthy"),
    "Potato___Late_Blight":         ("potato", "Late Blight"),
    "Rice___Brown_Spot":            ("rice",   "Brown Spot"),
    "Rice___Healthy":               ("rice",   "Healthy"),
    "Rice___Leaf_Blast":            ("rice",   "Leaf Blast"),
    "Wheat___Brown_Rust":           ("wheat",  "Brown Rust"),
    "Wheat___Healthy":              ("wheat",  "Healthy"),
    "Wheat___Yellow_Rust":          ("wheat",  "Yellow Rust"),
}

vit_model = None
vit_processor = None

app.config['UPLOAD_FOLDER'] = './uploads'

# ------------------------------------------------------------------
# Yield Prediction: ModifiedSVR & stacked pipeline (unchanged)
# ------------------------------------------------------------------
class ModifiedSVR:
    def __init__(self, C=1.0, base_epsilon=0.05, delta=0.02, lambda1=0.0, lr=0.005,
                 n_iter=3000, batch_size=16, random_state=0, early_stopping=False, stop_patience=500):
        self.C = C
        self.base_epsilon = base_epsilon
        self.delta = delta
        self.lambda1 = lambda1
        self.lr = lr
        self.n_iter = n_iter
        self.batch_size = batch_size
        self.rng = np.random.RandomState(random_state)
        self.w = None
        self.b = 0.0
        self.early_stopping = early_stopping
        self.stop_patience = stop_patience

    def fit(self, X, y, X_val=None, y_val=None):
        n_samples, n_features = X.shape
        if self.w is None:
            self.w = np.zeros(n_features)
        m = np.zeros_like(self.w); v = np.zeros_like(self.w)
        mb = 0.0; vb = 0.0
        beta1, beta2, eps_adam = 0.9, 0.999, 1e-8
        best_val_loss = np.inf; best_wb = None; patience = 0

        for it in range(1, self.n_iter + 1):
            idx = self.rng.choice(n_samples, size=min(self.batch_size, n_samples), replace=False)
            Xb = X[idx]; yb = y[idx]
            preds = Xb.dot(self.w) + self.b
            resid = yb - preds
            t = np.abs(resid) - self.base_epsilon
            dL_df = np.where(
                t <= 0, 0.0,
                np.where(t <= self.delta, -(t/self.delta)*np.sign(resid), -np.sign(resid))
            )
            grad_w = self.w.copy()
            if self.lambda1 != 0.0:
                grad_w += self.lambda1 * (self.w / (np.sqrt(self.w**2 + 1e-8)))
            grad_w += self.C * (dL_df[:, None] * Xb).sum(axis=0)
            grad_b = self.C * dL_df.sum()

            # Adam updates
            m = beta1*m + (1-beta1)*grad_w
            v = beta2*v + (1-beta2)*(grad_w**2)
            m_hat = m / (1 - beta1**it)
            v_hat = v / (1 - beta2**it)
            self.w -= self.lr * m_hat / (np.sqrt(v_hat) + eps_adam)

            mb = beta1*mb + (1-beta1)*grad_b
            vb = beta2*vb + (1-beta2)*(grad_b**2)
            mb_hat = mb / (1 - beta1**it)
            vb_hat = vb / (1 - beta2**it)
            self.b -= self.lr * mb_hat / (np.sqrt(vb_hat) + eps_adam)

            # validation
            if (it % 200 == 0 or it == 1) and (X_val is not None and y_val is not None):
                vpred = X_val.dot(self.w) + self.b
                vr = y_val - vpred
                vt = np.abs(vr) - self.base_epsilon
                vloss_terms = np.where(
                    vt <= 0, 0.0,
                    np.where(vt <= self.delta, vt**2/(2*self.delta), vt - self.delta/2)
                )
                val_loss = (
                    0.5*np.sum(self.w**2)
                    + self.lambda1*np.sum(np.sqrt(self.w**2 + 1e-8))
                    + self.C*np.sum(vloss_terms)
                )
                if val_loss < best_val_loss - 1e-9:
                    best_val_loss = val_loss
                    best_wb = (self.w.copy(), self.b)
                    patience = 0
                else:
                    patience += 200
                if self.early_stopping and patience >= self.stop_patience:
                    break
        if best_wb is not None:
            self.w, self.b = best_wb

    def predict(self, X):
        return X.dot(self.w) + self.b

stacked_pipeline = joblib.load("full_stacked_pipeline.pkl")
model_columns = stacked_pipeline["model_columns"]

# Helper: convert scaled logy -> raw yield
def scaled_logy_to_raw(y_scaled_array):
    inv = stacked_pipeline["scaler_y"].inverse_transform(
        np.array(y_scaled_array).reshape(-1, 1)
    ).ravel()
    return np.expm1(inv)

# Simple mapping for state climate data (for yield)
states_climate = {
    "Punjab":         {"AvgTemperature(C)": 25, "AnnualRainfall(mm)": 650},
    "Haryana":        {"AvgTemperature(C)": 26, "AnnualRainfall(mm)": 600},
    "Uttar Pradesh":  {"AvgTemperature(C)": 27, "AnnualRainfall(mm)": 900},
    "Maharashtra":    {"AvgTemperature(C)": 28, "AnnualRainfall(mm)": 1000},
    "West Bengal":    {"AvgTemperature(C)": 29, "AnnualRainfall(mm)": 1500},
}

# ------------------------------------------------------------------
# DB + Complaints + Clustering (unchanged)
# ------------------------------------------------------------------
def get_db_connection():
    conn = sqlite3.connect(DB_NAME, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    return conn

def init_db():
    conn = get_db_connection()
    conn.execute("""
        CREATE TABLE IF NOT EXISTS complaints (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            text TEXT NOT NULL,
            latitude REAL,
            longitude REAL,
            embedding TEXT,
            cluster_id INTEGER,
            created_at DATETIME DEFAULT CURRENT_TIMESTAMP
        );
    """)
    conn.commit()
    conn.close()

def run_clustering():
    conn = get_db_connection()
    rows = conn.execute("SELECT id, embedding, latitude, longitude FROM complaints").fetchall()

    vectors = []
    ids = []

    for row in rows:
        if row["embedding"] is None:
            continue
        try:
            emb = json.loads(row["embedding"])
        except:
            continue

        # You can optionally use lat/lon to enrich the vector
        combined_vector = emb
        vectors.append(combined_vector)
        ids.append(row["id"])

    if not vectors:
        conn.close()
        return 0

    X = np.array(vectors)
    kmeans = KMeans(n_clusters=3, random_state=42)
    labels = kmeans.fit_predict(X)

    for i, label in enumerate(labels):
        cid = int(label)
        complaint_id = ids[i]
        conn.execute(
            "UPDATE complaints SET cluster_id=? WHERE id=?",
            (cid, complaint_id)
        )

    conn.commit()
    conn.close()
    return 3

init_db()

# --------- HF ViT model (CNN block) ---------
def _load_processor(repo_id: str):
    try:
        return ViTImageProcessor.from_pretrained(repo_id)
    except Exception:
        # older versions use ViTFeatureExtractor
        return ViTFeatureExtractor.from_pretrained(repo_id)

def load_vit_model():
    """
    Lazy-load the ViT model & processor once, reuse for all requests.
    """
    global vit_model, vit_processor
    if vit_model is not None and vit_processor is not None:
        return vit_model, vit_processor

    try:
        processor = _load_processor(PRIMARY_REPO)
        model = ViTForImageClassification.from_pretrained(
            PRIMARY_REPO, ignore_mismatched_sizes=True
        )
    except Exception as e1:
        # fallback repo
        processor = _load_processor(FALLBACK_REPO)
        model = ViTForImageClassification.from_pretrained(
            FALLBACK_REPO, ignore_mismatched_sizes=True
        )

    model = model.to(DEVICE).eval()
    vit_model, vit_processor = model, processor
    return vit_model, vit_processor

def preprocess_vit(pil_img: Image.Image, processor):
    return processor(images=pil_img, return_tensors="pt")

@torch.no_grad()
def predict_vit(model, processor, pil_img: Image.Image, topk: int = 5):
    """
    Run ViT classifier and return:
      - top1: (label, prob)
      - topk list: [(label, prob), ...]
    """
    inputs = preprocess_vit(pil_img, processor)
    if DEVICE == "cuda":
        inputs = {k: v.cuda() for k, v in inputs.items()}

    outputs = model(**inputs)
    logits = outputs.logits
    probs = torch.softmax(logits, dim=-1).cpu().numpy().squeeze()

    id2label = model.config.id2label
    idx = int(np.argmax(probs))
    top1 = (id2label[idx], float(probs[idx]))

    k = min(topk, probs.shape[-1])
    top_indices = np.argsort(probs)[-k:][::-1]
    topk_pairs = [(id2label[int(i)], float(probs[int(i)])) for i in top_indices]
    return top1, topk_pairs

# --------- Ollama vision (Gemma 3) + text helper ----------
def _pil_to_base64(pil_img: Image.Image, fmt: str = "JPEG") -> str:
    # Downscale large images for faster vision inference
    max_side = 512
    w, h = pil_img.size
    scale = min(max_side / max(w, h), 1.0)
    if scale < 1.0:
        pil_img = pil_img.resize((int(w * scale), int(h * scale)))

    buf = BytesIO()
    pil_img.save(buf, format=fmt, quality=90)
    return base64.b64encode(buf.getvalue()).decode("utf-8")

def ollama_text(prompt: str,
                system: str | None = None,
                temperature: float = 0.2,
                max_tokens: int = 256) -> str:
    """
    Call Ollama (Gemma) as a TEXT-ONLY LLM (no images, no JSON schema).
    Used for disease guidance when ViT has already classified the image.
    """
    try:
        payload = {
            "model": OLLAMA_MODEL,
            "prompt": (f"{system}\n\n{prompt}" if system else prompt),
            "stream": False,
            "options": {
                "temperature": temperature,
                "num_predict": max_tokens,
            },
        }
        resp = requests.post(f"{OLLAMA_URL}/api/generate", json=payload, timeout=30)
        resp.raise_for_status()
        data = resp.json()
        return data.get("response", "").strip()
    except Exception as e:
        print("ollama_text error:", e)
        return ""

def build_disease_guidance_text(crop: str | None,
                                disease: str,
                                confidence: float | None) -> str:
    """
    Use Gemma (text-only) to generate structured, practical field guidance.

    ✅ Does NOT send the image
    ✅ Uses crop + disease + model confidence
    ✅ Follows your 7-point structure
    """
    crop_str = crop.capitalize() if crop else "Unknown crop"
    conf_str = f"{confidence:.4f}" if confidence is not None else "N/A"

    prompt = f"""
You are an agronomy expert for small and marginal farmers in India.

Crop: {crop_str}
Disease: {disease}
Model confidence: {conf_str}

Provide practical field guidance in simple, non-technical language.

Structure your answer using these exact headings in this order:

1) One-line summary:
2) Symptoms to check:
3) Immediate actions (first 24–48 hours):
4) Treatment options (cultural, biological, chemical):
5) Monitoring and thresholds:
6) Prevention for next season:
7) Red flags / when to consult an agronomist:

Formatting rules:
- Under every heading EXCEPT "One-line summary", use short bullet points (each line starting with "- ").
- Keep total length reasonably concise and actionable (roughly 200–300 words).

Special rule if the leaf is healthy:
- If the disease name contains "Healthy" or clearly means no disease, treat the plant as healthy.
- For sections 1), 2), 3), 4), 7) write a very short note like "Not applicable for healthy plants."
- Focus sections 5) and 6) on monitoring and prevention only.
"""

    system = (
        "You are a safe, conservative agronomy assistant. "
        "Avoid giving exact chemical doses or brand names; "
        "recommend consulting local agriculture officers or agronomists for prescriptions."
    )

    return ollama_text(prompt, system=system, temperature=0.2, max_tokens=256)

def ollama_vision(prompt: str, pil_img: Image.Image, system: str | None = None,
                  temperature: float = 0.2, max_tokens: int = 256) -> str:
    """
    Calls Ollama /api/generate with an image for Gemma vision.
    """
    b64 = _pil_to_base64(pil_img, fmt="JPEG")
    payload = {
        "model": OLLAMA_MODEL,
        "prompt": (f"{system}\n\n{prompt}" if system else prompt),
        "images": [b64],
        "stream": False,
        "options": {"temperature": temperature, "num_predict": max_tokens},
    }
    resp = requests.post(f"{OLLAMA_URL}/api/generate", json=payload, timeout=VISION_TIMEOUT_S)
    resp.raise_for_status()
    data = resp.json()
    return data.get("response", "").strip()

# --------- Prompts for Gemma guidance ---------
def prompt_supported(crop: str, disease: str, confidence: float) -> str:
    return f"""
Analyze the attached leaf image to validate and expand on the detected disease.

Crop: {crop}
Detected disease from classifier: {disease}
Classifier confidence: {confidence:.4f}

Your tasks:
- Confirm whether the image is consistent with this disease (brief).
- Provide practical field guidance:
  1) One-line summary
  2) Symptoms to check (bullets)
  3) Immediate actions (first 24–48 hours)
  4) Treatment options (cultural, biological, chemical)
  5) Monitoring/thresholds
  6) Prevention for next season
  7) Red flags / when to consult an agronomist
If the leaf looks Healthy, provide prevention and monitoring only.
Keep it concise and actionable.
"""

def prompt_unsupported(crop: str) -> str:
    return f"""
Analyze the attached leaf image and infer the most likely disease for the crop below.
Crop: {crop}

Your tasks:
- Name the most likely disease (or 'Healthy' if no disease).
- Give a short rationale referencing visual cues in the image.
- Provide actionable guidance:
  1) Immediate actions (first 24–48 hours)
  2) Treatment options (cultural, biological, chemical)
  3) Monitoring/thresholds
  4) Prevention
  5) Prevention tips for next season
  6) Red flags / when to consult an agronomist

Be practical and safety-first. If unsure, state uncertainty and offer differential possibilities.
"""


# ------------------------------------------------------------------
# NEW: Crop Recommendation Engine – ML + KB + Ollama Gemma + Open-Meteo
# ------------------------------------------------------------------

# Artifacts directory
ARTIFACT_DIR = Path(__file__).parent / "artifacts"

# ML artifacts for crop recommendation
feature_cols = json.load(open(ARTIFACT_DIR / "feature_cols.json", "r"))
class_names  = json.load(open(ARTIFACT_DIR / "class_names.json", "r"))
ml_model     = joblib.load(ARTIFACT_DIR / "model.joblib")
scaler_path  = ARTIFACT_DIR / "scaler.joblib"
scaler       = joblib.load(scaler_path) if scaler_path.exists() else None

# Minimal KB (same as previous ML backend)
CROP_KB = {
    "rice": {
        "season": ["kharif"],
        "temp_c": [20, 35],
        "rain_mm_year": [800, 2000],
        "water_need": "high",
        "states_common": ["West Bengal","Odisha","Assam","Andhra Pradesh","Tamil Nadu","Punjab"],
        "tags": ["cereals"],
        "notes": "Requires assured water; bunding & drainage control recommended."
    },
    "wheat": {
        "season": ["rabi"],
        "temp_c": [10, 25],
        "rain_mm_year": [300, 600],
        "water_need": "medium",
        "states_common": ["Punjab","Haryana","Uttar Pradesh","Rajasthan","Madhya Pradesh"],
        "tags": ["cereals"],
        "notes": "Cool-season crop; critical irrigation at CRI stage."
    },
    "maize": {
        "season": ["kharif","rabi"],
        "temp_c": [18, 32],
        "rain_mm_year": [500, 1000],
        "water_need": "medium",
        "states_common": ["Karnataka","Andhra Pradesh","Maharashtra","Bihar"],
        "tags": ["cereals"],
        "notes": "Well-drained soils; avoid waterlogging at flowering."
    },
    "chickpea": {
        "season": ["rabi"],
        "temp_c": [10, 28],
        "rain_mm_year": [300, 700],
        "water_need": "low",
        "states_common": ["Madhya Pradesh","Maharashtra","Rajasthan","Uttar Pradesh"],
        "tags": ["pulses"],
        "notes": "Sensitive to waterlogging; suits rice-fallow residual moisture."
    },
    "lentil": {
        "season": ["rabi"],
        "temp_c": [8, 25],
        "rain_mm_year": [300, 600],
        "water_need": "low",
        "states_common": ["Uttar Pradesh","Madhya Pradesh","Bihar","West Bengal"],
        "tags": ["pulses"],
        "notes": "Short duration; prefers cool, dry conditions."
    },
    "sorghum": {
        "season": ["kharif","rabi"],
        "temp_c": [20, 34],
        "rain_mm_year": [400, 900],
        "water_need": "low",
        "states_common": ["Maharashtra","Karnataka","Telangana"],
        "tags": ["millets"],
        "notes": "Drought-tolerant; fits low rainfall tracts."
    },
    "pearl millet": {
        "season": ["kharif"],
        "temp_c": [22, 36],
        "rain_mm_year": [300, 700],
        "water_need": "low",
        "states_common": ["Rajasthan","Gujarat","Haryana","Uttar Pradesh"],
        "tags": ["millets"],
        "notes": "High drought hardiness; suitable for arid/semi-arid zones."
    },
    "soybean": {
        "season": ["kharif"],
        "temp_c": [20, 32],
        "rain_mm_year": [600, 1200],
        "water_need": "medium",
        "states_common": ["Madhya Pradesh","Maharashtra","Rajasthan"],
        "tags": ["oilseeds"],
        "notes": "Well-drained soils; avoid waterlogging during flowering."
    },
    "groundnut": {
        "season": ["kharif","rabi"],
        "temp_c": [20, 33],
        "rain_mm_year": [500, 1000],
        "water_need": "medium",
        "states_common": ["Gujarat","Andhra Pradesh","Tamil Nadu","Karnataka"],
        "tags": ["oilseeds"],
        "notes": "Sandy loam preferred; pegging requires loose soil and timely gypsum."
    },
    "cotton": {
        "season": ["kharif"],
        "temp_c": [20, 35],
        "rain_mm_year": [600, 1200],
        "water_need": "high",
        "states_common": ["Maharashtra","Gujarat","Telangana","Punjab"],
        "tags": ["cash"],
        "notes": "Long duration; plan IPM for bollworms/whitefly."
    },
}

KB_SYNONYMS = {
    "bajra": "pearl millet",
    "mothbeans": "sorghum",
    "arhar": "pigeonpea",
    "gram": "chickpea"
}

def kb_name(crop_label: str) -> str:
    key = crop_label.strip().lower()
    return KB_SYNONYMS.get(key, key)

def kb_get(crop_label: str):
    return CROP_KB.get(kb_name(crop_label))

# State-level soil priors
SOIL_PRIORS = {
    "Punjab":        {"N":55,"P":40,"K":35,"ph":7.3},
    "Haryana":       {"N":50,"P":38,"K":36,"ph":7.4},
    "Uttar Pradesh": {"N":45,"P":35,"K":34,"ph":7.0},
    "Maharashtra":   {"N":40,"P":32,"K":38,"ph":6.7},
    "West Bengal":   {"N":48,"P":37,"K":39,"ph":6.2},
}
DEFAULT_SOIL = {"N":42,"P":34,"K":36,"ph":6.8}

# --------- Helpers: Open-Meteo (for recommendation) ---------
def fetch_weather_open_meteo(lat: float, lon: float) -> dict:
    url = (
        "https://api.open-meteo.com/v1/forecast"
        f"?latitude={lat}&longitude={lon}"
        "&hourly=temperature_2m,relative_humidity_2m,precipitation"
        "&past_days=7&forecast_days=1&timezone=auto"
    )
    r = requests.get(url, timeout=10)
    r.raise_for_status()
    js = r.json()
    t = float(np.mean(js["hourly"]["temperature_2m"]))
    h = float(np.mean(js["hourly"]["relative_humidity_2m"]))
    rsum = float(np.sum(js["hourly"]["precipitation"]))
    return {
        "source": "open-meteo",
        "temperature_avg_c": round(t, 2),
        "humidity_avg_pct": round(h, 1),
        "rainfall_7d_mm": round(rsum, 1),
        "timestamp_utc": datetime.utcnow().isoformat() + "Z"
    }

# --------- Helpers: LLM (Ollama / Gemma) ---------
def _strip_code_fences(text: str):
    s = text.strip()
    s = re.sub(r"^```(?:json)?\s*", "", s)
    s = re.sub(r"\s*```$", "", s)
    return s

def llm_json_generate(prompt: str, system: str) -> dict | list | str:
    try:
        resp = requests.post(
            f"{OLLAMA_URL}/api/generate",
            json={
                "model": OLLAMA_MODEL,
                "prompt": prompt,
                "system": system,
                "options": {"temperature": 0.2, "num_ctx": 4096},
                "format": "json",
                "stream": False
            },
            timeout=30
        )
        resp.raise_for_status()
        text = resp.json().get("response", "").strip()
        try:
            return json.loads(text)
        except Exception:
            try:
                return json.loads(_strip_code_fences(text))
            except Exception:
                return text[:800]
    except Exception as e:
        return {"error": f"llm_call_failed: {e}"}

# --------- ML Inference for top-k crops ---------
def infer_topk(features: dict, top_k: int = 3):
    """
    features: dict with keys exactly matching feature_cols.
    Uses DataFrame to keep feature names for the scaler.
    """
    X = pd.DataFrame(
        [[float(features[f]) for f in feature_cols]],
        columns=feature_cols
    )
    if scaler is not None:
        X_scaled = scaler.transform(X)
    else:
        X_scaled = X.values

    if hasattr(ml_model, "predict_proba"):
        probs = ml_model.predict_proba(X_scaled)[0]
        pairs = sorted(
            zip(class_names, probs),
            key=lambda t: t[1],
            reverse=True
        )[:top_k]
        return [{"crop": c, "confidence": round(float(p), 3)} for c, p in pairs]

    pred = ml_model.predict(X_scaled)[0]
    return [{"crop": str(pred), "confidence": None}]

# --------- Per-crop explainer via Gemma ---------
def gemma_explain_crop(crop_name: str, kb: dict, weather: dict | None,
                       state: str | None, locale: str = "en-IN"):
    system = (
        "You are Krishi Vikas Advisor. Use ONLY the provided KB facts and context. "
        "Write short, farmer-friendly guidance. Return STRICT JSON with keys: "
        "summary, why_suitable_now, water_and_soil, basic_management (array), "
        "risks (array), disclaimers (array)."
    )
    ctx = {
        "crop": crop_name,
        "kb": kb,
        "weather": weather or {},
        "state": state,
        "locale": locale
    }
    schema = (
        '{'
        '"summary":"...",'
        '"why_suitable_now":"...",'
        '"water_and_soil":"...",'
        '"basic_management":["...","..."],'
        '"risks":["...","..."],'
        '"disclaimers":["...","..."]'
        '}'
    )
    prompt = (
        f"LOCALE: {locale}\n"
        f"CONTEXT: {json.dumps(ctx, ensure_ascii=False)}\n\n"
        "TASK:\n"
        "- SUMMARIZE what this crop needs (season/temp/water) in <= 35 words.\n"
        "- WHY_SUITABLE_NOW: If weather fits KB ranges, explain in <= 40 words; else say 'conditions are marginal'.\n"
        "- WATER_AND_SOIL: 1–2 sentences (<= 40 words) using water_need and any KB note.\n"
        "- BASIC_MANAGEMENT: 3–5 bullet tips (short strings: sowing window, spacing/variety or seed treatment, irrigation, fertilizer timing/IPM cue).\n"
        "- RISKS: 2–3 short risks (e.g., waterlogging, drought, specific pests if hinted in notes).\n"
        "- DISCLAIMERS: 2 brief items.\n"
        "OUTPUT JSON ONLY that matches exactly this schema:\n"
        f"{schema}"
    )

    out = llm_json_generate(prompt, system)
    if not isinstance(out, dict) or "summary" not in out:
        return {
            "summary": f"{crop_name.title()} — season {', '.join(kb.get('season', []))}; water need {kb.get('water_need','unknown')}.",
            "why_suitable_now": "Weather fit checked against KB temperature & rainfall bands.",
            "water_and_soil": kb.get("notes", "Manage irrigation and drainage as per local practice."),
            "basic_management": [
                "Use certified seed",
                "Prepare well-drained field",
                "Irrigate at critical stages",
                "Follow local fertilizer schedule"
            ],
            "risks": ["Weather variability", "Pest/disease surges"],
            "disclaimers": [
                "Consult local agri advisory for doses/timings.",
                "Availability of inputs may vary."
            ]
        }
    return out

@app.route('/recommendations', methods=['POST'])
def recommendations():
    """
    Existing frontend contract:
      - Endpoint: /recommendations
      - Body: { latitude, longitude, category, language }
      - Response used: { "Recommendations": [ { name, ename, description, image } ] }

    Also supports legacy body keys:
      - lat / lon instead of latitude / longitude
      - locale instead of language
    """
    data = request.get_json(force=True) or {}

    # Support both new (latitude/longitude) and old (lat/lon) keys
    lat = data.get('latitude', data.get('lat'))
    lon = data.get('longitude', data.get('lon'))

    category = data.get('category')  # currently not used in ML, but kept
    lang = data.get('language', data.get('locale', 'en'))  # Default to English

    if lat is None or lon is None:
        return jsonify({'error': 'Latitude and Longitude are required'}), 400

    try:
        lat = float(lat)
        lon = float(lon)
    except ValueError:
        return jsonify({'error': 'Invalid latitude/longitude'}), 400

    # 1) Weather via Open-Meteo
    try:
        wx = fetch_weather_open_meteo(lat, lon)
    except Exception as e:
        print(f"Open-Meteo error: {e}")
        return jsonify({'error': 'Failed to get weather data'}), 500

    # 2) Soil: either manual from body OR estimated via state priors
    #    Expecting keys: "ph", "N", "P", "K" (optional)
    ph_val = data.get("ph")
    N_val  = data.get("N")
    P_val  = data.get("P")
    K_val  = data.get("K")

    if all(v not in (None, "") for v in (ph_val, N_val, P_val, K_val)):
        # Use user-provided soil values
        try:
            soil = {
                "ph": float(ph_val),
                "N":  float(N_val),
                "P":  float(P_val),
                "K":  float(K_val),
            }
        except ValueError:
            return jsonify({"error": "Invalid soil values"}), 400
        state = get_state_from_coordinates(lat, lon)
        estimated_soil = False
    else:
        # Fall back to state-level priors
        state = get_state_from_coordinates(lat, lon)
        soil = SOIL_PRIORS.get(state or "", DEFAULT_SOIL).copy()
        estimated_soil = True

    # 3) Build ML feature vector
    features = {
        "temperature": wx["temperature_avg_c"],
        "humidity":    wx["humidity_avg_pct"],
        "rainfall":    wx["rainfall_7d_mm"],
        "ph":          soil["ph"],
        "N":           soil["N"],
        "P":           soil["P"],
        "K":           soil["K"],
    }

    # 4) Run ML model for top-k
    top_k = int(data.get("top_k", 3))
    try:
        recs = infer_topk(features, top_k=top_k)
    except Exception as e:
        print("Model inference failed:", e)
        return jsonify({'error': 'Model inference failed'}), 500

    if not recs:
        return jsonify({'error': 'No recommendations generated'}), 500

    # Flags
    top_score = max((r["confidence"] or 0) for r in recs)
    flags = {
        "low_confidence": bool(top_score < 0.30),
        "estimated_soil": estimated_soil
    }

    # 5) Top-1 explanation (KB + Gemma)
    top_crop = recs[0]["crop"]
    kb = kb_get(top_crop) or {}

    wx_for_expl = {
        "temperature_avg_c": wx["temperature_avg_c"],
        "humidity_avg_pct":  wx["humidity_avg_pct"],
        "rainfall_7d_mm":    wx["rainfall_7d_mm"],
    }

    explanation = gemma_explain_crop(
        kb_name(top_crop),
        kb,
        wx_for_expl,
        state,
        locale="en-IN"
    )

    # Optional: translate explanation to requested language
    top_expl = explanation
    if lang != 'en':
        try:
            top_expl = {
                k: (
                    [translate_text(x, lang) for x in v]
                    if isinstance(v, list)
                    else translate_text(v, lang)
                )
                for k, v in explanation.items()
            }
        except Exception as e:
            print("Explanation translation error:", e)

    # 6) Build Recommendations list in OLD shape: name, ename, description, image
    ui_recs = []
    for r in recs:
        name_en = kb_name(r["crop"])
        conf_pct = int((r["confidence"] or 0) * 100)
        desc_en = (
            f"Recommended based on current temperature ({wx['temperature_avg_c']}°C), "
            f"humidity ({wx['humidity_avg_pct']}%) and soil profile. "
            f"Model confidence ~{conf_pct}%."
        )
        image_url = get_crop_image(name_en)

        name_out = name_en
        desc_out = desc_en
        if lang != 'en':
            try:
                name_out = translate_text(name_en, lang)
                desc_out = translate_text(desc_en, lang)
            except Exception as e:
                print("Recommendation translation error:", e)

        ui_recs.append({
            "name":        name_out,
            "ename":       name_en,
            "description": desc_out,
            "image":       image_url,
            "confidence":  r["confidence"]
        })

    # 7) Final response
    return jsonify({
        "Recommendations": ui_recs,
        "location": {
            "latitude":  lat,
            "longitude": lon,
            "state":     state
        },
        "weather": wx_for_expl,
        "soil":    soil,
        "top_recommendations": recs,
        "top_explanation": {
            "crop":    ui_recs[0]["name"],
            "ename":   ui_recs[0]["ename"],
            "kb":      kb,
            "details": top_expl
        },
        "flags": flags,
        "meta": {
            "llm_model":      OLLAMA_MODEL,
            "weather_source": wx.get("source")
        }
    })


# Legacy endpoint for older frontend / clients
@app.route('/recommend', methods=['POST'])
def recommend_legacy():
    """
    Thin wrapper to keep backward compatibility.

    Any client calling /recommend will be handled by the same logic
    as /recommendations. We also support both lat/lon and
    latitude/longitude in the body.
    """
    return recommendations()

# ------------------------------------------------------------------
# Translation + Image helpers (unchanged)
# ------------------------------------------------------------------
def translate_text(text, lang):
    try:
        return GoogleTranslator(source='auto', target=lang).translate(text)
    except Exception as e:
        print(f"Translation error: {e}")
        return text

def get_crop_image(crop_name):
    url = f"https://api.unsplash.com/search/photos?query={crop_name}&client_id={UNSPLASH_ACCESS_KEY}&per_page=1"
    try:
        response = requests.get(url)
        response.raise_for_status()
        data = response.json()
        return data['results'][0]['urls']['small'] if data['results'] else None
    except requests.exceptions.RequestException as e:
        print(f"Error fetching image for {crop_name}: {e}")
        return None

# ------------------------------------------------------------------
# WeatherAPI-based helpers (used by calendar, not recommendation)
# ------------------------------------------------------------------
def get_weather_data(lat, lon):
    url = f'http://api.weatherapi.com/v1/current.json?key={WEATHER_API_KEY}&q={lat},{lon}'
    try:
        response = requests.get(url)
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        print(f"Error fetching weather data: {e}")
        return None

# ------------------------------------------------------------------
# Crop steps, disease prediction, chatbot, calendar
# ------------------------------------------------------------------
@app.route('/crop_steps', methods=['POST'])
def crop_steps():
    data = request.json
    crop_name = data.get('crop_name')
    lang = data.get('language', 'en')
    category = data.get('category')

    if not crop_name:
        return jsonify({'error': 'Crop name is required'}), 400

    if lang != 'en':
        crop_name = translate_text(crop_name, lang)

    try:
        if category:
            queries = [f"Give me a small paragraph on {category} for {crop_name} in language {lang}"]

        response = model.generate_content(queries)
        recommendations_text = response.text
        return recommendations_text

    except Exception as e:
        print(f"Error generating crop growing steps: {e}")
        return jsonify({'error': 'Failed to get crop growing steps'}), 500

@app.route('/predict-disease', methods=['POST'])
def predict_disease():
    """
    Disease detection pipeline:
      - Input: multipart/form-data with:
          - image: file (required)
          - crop (optional): corn/potato/rice/wheat/other crop name
      - Routing:
          * If crop is in SUPPORTED_CROPS  → ViT + text-only Gemma guidance
          * If crop is not provided       → ViT + text-only Gemma guidance (backward compatible)
          * If crop is provided but NOT in SUPPORTED_CROPS → Gemma Vision only (image sent)
    Response JSON keeps `prediction` for backward compatibility.
    """
    if 'image' not in request.files:
        return jsonify({'error': 'Image is required'}), 400

    image_file = request.files['image']
    if image_file.filename == '':
        return jsonify({'error': 'Empty filename'}), 400

    # crop hint can come from form or query
    crop_raw = (
        request.form.get('crop')
        or request.form.get('crop_name')
        or request.args.get('crop')
    )
    crop = crop_raw.strip().lower() if crop_raw else None

    # ---- load & save image ----
    try:
        os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
        filename = secure_filename(image_file.filename or "leaf.jpg")
        save_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)

        image_bytes = image_file.read()
        with open(save_path, "wb") as f:
            f.write(image_bytes)

        pil_img = Image.open(BytesIO(image_bytes)).convert("RGB")

    except UnidentifiedImageError:
        return jsonify({"error": "Uploaded file is not a valid image"}), 400
    except Exception as e:
        print("Error reading image:", e)
        return jsonify({"error": "Failed to read image"}), 500

    try:
        # ------------------------------------------------------------------
        # ROUTING DECISION
        # ------------------------------------------------------------------
        # 1) No crop provided  → ViT + text-only Gemma
        # 2) crop in SUPPORTED_CROPS → ViT + text-only Gemma
        # 3) crop provided but not in SUPPORTED_CROPS → Gemma Vision only
        # ------------------------------------------------------------------
        use_vit = False
        use_gemma_only = False

        if crop is None:
            use_vit = True
        elif crop in SUPPORTED_CROPS:
            use_vit = True
        else:
            use_gemma_only = True

        # ---------------- Route A: ViT + TEXT-ONLY GEMMA (no image) -------------
        if use_vit:
            vit, processor = load_vit_model()
            top1, topk_pairs = predict_vit(vit, processor, pil_img, TOPK)
            hf_label, conf = top1

            # Map HF label to (crop, pretty disease name) if possible
            crop_norm, disease_name = LABEL_PRETTY.get(hf_label, (None, hf_label))

            # Prefer mapped crop; fall back to user-provided crop
            effective_crop = crop_norm or crop

            # Generate practical guidance with text-only Gemma
            guidance = ""
            try:
                guidance = build_disease_guidance_text(
                    effective_crop,
                    disease_name,
                    conf,
                )
            except Exception as e:
                print("Disease guidance generation failed:", e)

            return jsonify({
                "source": "vit+gemma",
                "crop": effective_crop,
                "classifier_label": hf_label,
                "disease": disease_name,
                "classifier_confidence": round(conf, 4),
                "topk": [
                    {"label": lbl, "probability": round(p, 4)}
                    for (lbl, p) in topk_pairs
                ],
                # for old clients expecting a plain string
                "prediction": disease_name,
                # full narrative guidance
                "gemma_guidance": guidance,
            })

        # ---------------- Route B: Gemma Vision ONLY -------------------
        crop_name_for_prompt = crop.capitalize() if crop else "Unknown crop"
        gprompt = prompt_unsupported(crop_name_for_prompt)
        guidance = ollama_vision(
            gprompt,
            pil_img,
            system="You are a practical agronomy assistant.",
            temperature=0.2,
        )

        return jsonify({
            "source": "gemma-only",
            "crop": crop,
            # For compatibility, prediction is the full guidance text here
            "prediction": guidance,
            "gemma_guidance": guidance,
        })

    except requests.exceptions.RequestException as e:
        print("Ollama request failed:", e)
        return jsonify({"error": "Vision backend not available"}), 500
    except Exception as e:
        print("Error making disease prediction:", e)
        return jsonify({"error": "Failed to make prediction"}), 500


@app.route('/chatbot', methods=['POST'])
def chatbot():
    data = request.json
    lang = data.get('language', 'en')
    query = data.get('query')

    if lang != 'en' and query:
        query = translate_text(query, lang)

    try:
        if query:
            queries = [(
                "Give me the answer for this query "
                f"{query} in language {lang}. "
                "The query should be related to only farming related stuff. "
                "In case of any other irrelevant question just say something like "
                "I am KrishiSahayak and will answer to only farming related stuff. "
                "In case of queries like diseases of crops, ask them to go to "
                "CNN Enabled Plant Disease Identification of the same website KrishiVikas"
            )]

        response = model.generate_content(queries)
        recommendations_text = response.text

        recommendations_text = recommendations_text.replace("**", "")
        recommendations_text = recommendations_text.replace("*", "-")
        return recommendations_text

    except Exception as e:
        print(f"Error generating answers: {e}")
        return jsonify({'error': 'Failed to get answers'}), 500

def get_weather_forecast(lat, lon):
    url = f"http://api.weatherapi.com/v1/forecast.json?key={WEATHER_API_KEY}&q={lat},{lon}&days=7"
    try:
        response = requests.get(url)
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        print(f"Error fetching weather forecast: {e}")
        return None

@app.route("/weather-forecast", methods=["POST"])
def weather_forecast():
    data = request.json
    lat = data.get("lat")
    lon = data.get("lon")

    if not lat or not lon:
        return jsonify({"error": "Latitude and Longitude are required"}), 400

    forecast_data = get_weather_forecast(lat, lon)
    if forecast_data:
        return jsonify(forecast_data)
    else:
        return jsonify({"error": "Failed to fetch weather forecast"}), 500

@app.route('/crop-calendar', methods=['POST'])
def generate_crop_calendar():
    try:
        data = request.json
        if not data or "cropName" not in data or "latitude" not in data or "longitude" not in data:
            return jsonify({"message": "Invalid data provided"}), 400

        crop_name = data["cropName"].strip()
        latitude = data["latitude"]
        longitude = data["longitude"]
        lang = data["lang"]

        if not crop_name:
            return jsonify({"message": "Crop name is required"}), 400

        weather_data = get_weather_data(latitude, longitude)
        if not weather_data:
            return jsonify({'error': 'Failed to get weather data'}), 500

        prompt = (
            f"You are an expert agronomist. Generate a **detailed crop calendar** for {crop_name} based on {weather_data} in this language {lang}."
            f"based on the location (Latitude: {latitude}, Longitude: {longitude}) in India. "
            f"Ensure it follows best farming practices suited for the region. The response should "
            f"be **structured and formatted** so that each stage is clearly separated. Use double asterisks (**) "
            f"to highlight stage titles. Keep each section detailed and concise.\n\n"
            f"### Crop Calendar for {crop_name} ({latitude}, {longitude})\n"
            f"1. **Land Preparation**\n   - Time:\n   - Activities:\n"
            f"2. **Seed Treatment**\n   - Time:\n   - Method:\n   - Chemicals Used:\n"
            f"3. **Sowing Period**\n   - Best Months:\n   - Method:\n"
            f"4. **Irrigation Schedule**\n   - Frequency:\n   - Best Practices:\n"
            f"5. **Fertilization Schedule**\n   - Types of Fertilizers:\n   - Application Timing:\n"
            f"6. **Weed Management**\n   - Techniques:\n   - Chemicals Used:\n"
            f"7. **Pest & Disease Management**\n   - Common Pests & Diseases:\n   - Control Methods:\n"
            f"8. **Harvesting Time**\n   - Month:\n   - Harvesting Methods:\n"
            f"9. **Post-Harvest Handling**\n   - Storage & Processing Tips:\n\n"
            f"Make sure the response follows this structured format so that each stage is **clearly extractable**."
        )

        response = model.generate_content([prompt])
        crop_calendar_text = response.text
        crop_calendar_text = crop_calendar_text.replace("**", "")
        crop_calendar_text = crop_calendar_text.replace("###", "")

        return jsonify({"cropCalendar": crop_calendar_text}), 200

    except Exception as e:
        print(f"Error while generating crop calendar: {e}")
        return jsonify({"message": "Error generating crop calendar"}), 500

# ------------------------------------------------------------------
# Sales Data (forecast, best/worst sellers) – unchanged
# ------------------------------------------------------------------
df = pd.read_csv('../data/indian_crop_sales_data.csv', parse_dates=['Date'])

crop_sums = df.groupby('Crop')['Quantity Sold (kg)'].sum().sort_values(ascending=False)
top_3_crops = crop_sums.head(3)
bottom_3_crops = crop_sums.tail(3)

@app.route('/best_worst_sellers', methods=['GET'])
def best_worst_sellers():
    best_sellers_df = top_3_crops.reset_index().rename(
        columns={'Quantity Sold (kg)': 'TotalSales'}
    )
    worst_sellers_df = bottom_3_crops.reset_index().rename(
        columns={'Quantity Sold (kg)': 'TotalSales'}
    )
    response = {
        'best_sellers': best_sellers_df.to_dict(orient='records'),
        'worst_sellers': worst_sellers_df.to_dict(orient='records')
    }
    return jsonify(response)

@app.route('/forecast', methods=['GET'])
def forecast():
    crop_name = request.args.get('crop', 'Rice')
    periods = request.args.get('periods', 7, type=int)

    crop_data = df[df['Crop'] == crop_name].copy()
    crop_data.sort_values(by='Date', inplace=True)
    crop_data.set_index('Date', inplace=True)

    if len(crop_data) < 5:
        return jsonify({'error': 'Not enough data to forecast for this crop.'}), 400

    y = crop_data['Quantity Sold (kg)']

    model_sarima = SARIMAX(
        y,
        order=(1, 1, 1),
        seasonal_order=(1, 1, 1, 7),
        enforce_stationarity=False,
        enforce_invertibility=False
    )
    results = model_sarima.fit(disp=False)

    forecast_values = results.predict(
        start=len(y),
        end=len(y) + periods - 1,
        typ='levels'
    ).tolist()

    last_date = crop_data.index[-1]
    future_dates = [last_date + timedelta(days=i) for i in range(1, periods+1)]

    forecast_output = []
    for date, val in zip(future_dates, forecast_values):
        forecast_output.append({
            'Date': date.strftime('%Y-%m-%d'),
            'Forecast': round(val, 2)
        })

    historical_df = crop_data.tail(7).reset_index()
    historical_df['Date'] = historical_df['Date'].dt.strftime('%Y-%m-%d')
    historical_output = historical_df[['Date', 'Quantity Sold (kg)']] \
        .rename(columns={'Quantity Sold (kg)': 'Actual'})

    response = {
        'crop': crop_name,
        'historical': historical_output.to_dict(orient='records'),
        'forecast': forecast_output
    }
    return jsonify(response)

# ------------------------------------------------------------------
# Geocoding + Yield Prediction (unchanged)
# ------------------------------------------------------------------
def get_state_from_coordinates(latitude, longitude):
    geolocator = Nominatim(user_agent="geoapi")
    try:
        location = geolocator.reverse((latitude, longitude), exactly_one=True)
        if location and "state" in location.raw["address"]:
            return location.raw["address"]["state"]
    except Exception as e:
        print("Geolocation error:", e)
    return None

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.json
        if not data:
            return jsonify({"error": "No JSON payload received"}), 400

        def safe_float(val):
            try:
                return float(val)
            except:
                return 0.0

        crop_type = data.get("cropType", "")
        land_size = safe_float(data.get("landSize"))
        fertilizer = safe_float(data.get("fertilizer"))
        pesticide = safe_float(data.get("pesticide"))
        latitude = safe_float(data.get("latitude"))
        longitude = safe_float(data.get("longitude"))
        year = safe_float(data.get("year", 2024))

        state = get_state_from_coordinates(latitude, longitude)
        climate = states_climate.get(
            state,
            {"AvgTemperature(C)": 25, "AnnualRainfall(mm)": 700}
        )

        input_dict = {
            "Year": [year],
            "LandSize(ha)": [land_size],
            "FertilizerUsage(kg_ha)": [fertilizer],
            "PesticideUsage(kg_ha)": [pesticide],
            "AvgTemperature(C)": [climate["AvgTemperature(C)"]],
            "AnnualRainfall(mm)": [climate["AnnualRainfall(mm)"]],
            "State_Haryana": [1 if state == "Haryana" else 0],
            "State_Maharashtra": [1 if state == "Maharashtra" else 0],
            "State_Punjab": [1 if state == "Punjab" else 0],
            "State_Uttar Pradesh": [1 if state == "Uttar Pradesh" else 0],
            "State_West Bengal": [1 if state == "West Bengal" else 0],
            "CropType_Maize": [1 if crop_type == "Maize" else 0],
            "CropType_Rice": [1 if crop_type == "Rice" else 0],
            "CropType_Wheat": [1 if crop_type == "Wheat" else 0],
        }

        input_df = pd.DataFrame(input_dict)
        for col in model_columns:
            if col not in input_df.columns:
                input_df[col] = 0
        input_df = input_df[model_columns]

        X_s1 = stacked_pipeline["scaler_X1"].transform(input_df)
        X_poly = stacked_pipeline["poly"].transform(X_s1)
        X_final = stacked_pipeline["scaler_X2"].transform(X_poly)

        ridge_pred = stacked_pipeline["ridge"].predict(X_final)
        rf_pred = stacked_pipeline["rf"].predict(X_final)
        gb_pred = stacked_pipeline["gb"].predict(X_final)
        svr_pred = stacked_pipeline["svr"].predict(X_final)

        rf_pred_scaled = stacked_pipeline["scaler_y"].transform(
            np.log1p(rf_pred).reshape(-1, 1)
        ).ravel()
        gb_pred_scaled = stacked_pipeline["scaler_y"].transform(
            np.log1p(gb_pred).reshape(-1, 1)
        ).ravel()

        meta_input = np.column_stack([
            ridge_pred,
            rf_pred_scaled,
            gb_pred_scaled,
            svr_pred
        ])
        stack_pred_scaled = stacked_pipeline["meta_model"].predict(meta_input)
        stack_pred_raw = scaled_logy_to_raw(stack_pred_scaled)

        return jsonify({"predicted_yield": round(float(stack_pred_raw[0]), 2)})

    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500

# ------------------------------------------------------------------
# Complaints APIs (unchanged)
# ------------------------------------------------------------------
@app.route('/submit-complaint', methods=['POST'])
def submit_complaint():
    data = request.json
    if not data:
        return jsonify({"error": "No data provided"}), 400

    complaint_text = data.get("text", "")
    lat = data.get("latitude")
    lon = data.get("longitude")

    if not complaint_text or lat is None or lon is None:
        return jsonify({"error": "Please provide text, latitude, and longitude"}), 400

    random_embedding = [round(random.uniform(-1, 1), 3) for _ in range(5)]
    embedding_json = json.dumps(random_embedding)

    conn = get_db_connection()
    conn.execute("""
        INSERT INTO complaints (text, latitude, longitude, embedding)
        VALUES (?, ?, ?, ?)
    """, (complaint_text, lat, lon, embedding_json))
    conn.commit()
    conn.close()

    return jsonify({"message": "Complaint submitted successfully"}), 200

@app.route('/run-clustering', methods=['POST'])
def do_clustering():
    num_clusters = run_clustering()
    return jsonify({"message": f"Clustering complete. #clusters={num_clusters}"}), 200

@app.route('/complaints', methods=['GET'])
def get_complaints():
    conn = get_db_connection()
    rows = conn.execute("SELECT * FROM complaints").fetchall()
    conn.close()

    complaints = []
    for row in rows:
        complaints.append({
            "id": row["id"],
            "text": row["text"],
            "latitude": row["latitude"],
            "longitude": row["longitude"],
            "embedding": row["embedding"],
            "cluster_id": row["cluster_id"],
            "created_at": row["created_at"]
        })
    return jsonify(complaints)

# ------------------------------------------------------------------
# Main
# ------------------------------------------------------------------
if __name__ == '__main__':
    if not os.path.exists(app.config['UPLOAD_FOLDER']):
        os.makedirs(app.config['UPLOAD_FOLDER'])
    app.run(debug=True)
