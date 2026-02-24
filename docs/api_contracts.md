# API Contracts — Vitiscan

This document describes the exact request and response formats for all Vitiscan APIs.
It serves as the reference for any component that communicates with these APIs —
in particular the Streamlit frontend, which chains both API calls for each user request.

---

## Overview — How the APIs are chained

```
User uploads a leaf photo
        │
        ▼
POST /diagno  (Diagnostic API)
        │
        └── returns predicted disease class (e.g. "plasmopara_viticola")
                        │
                        ▼
                POST /solutions  (Treatment Plan API)
                        │
                        └── returns full treatment plan
```

The Streamlit app makes these two calls sequentially.
The `cnn_label` field from the Diagnostic API response is passed directly
as input to the Treatment Plan API.

---

## 1. Diagnostic API

**Base URL (production):** `https://mouniat-vitiscanpro-diagno-api.hf.space`
**Base URL (local):** `http://localhost:4000`

---

### GET /

Health check endpoint. Used to verify the API is running before sending requests.

**Request:** no body required.

**Response `200 OK`:**
```json
{
  "message": "Vitiscan Diagnostic API is running",
  "status": "ok"
}
```

---

### GET /diseases

Returns the full list of detectable disease classes and their human-readable labels.

**Request:** no body required.

**Response `200 OK`:**
```json
{
  "diseases": {
    "colomerus_vitis":             "Colomerus Vitis",
    "elsinoe_ampelina":            "Elsinoe Ampelina",
    "erysiphe_necator":            "Erysiphe Necator",
    "guignardia_bidwellii":        "Guignardia Bidwellii",
    "healthy":                     "Healthy",
    "phaeomoniella_chlamydospora": "Phaeomoniella Chlamydospora",
    "plasmopara_viticola":         "Plasmopara Viticola"
  },
  "dataset_name": "inrae"
}
```

---

### POST /diagno

Runs disease classification on an uploaded leaf photograph.

**Request:** `multipart/form-data`

| Field | Type | Required | Description |
|---|---|---|---|
| `file` | image file | ✅ | JPEG or PNG leaf photograph (recommended: 224×224 px minimum) |

**Example (curl):**
```bash
curl -X POST https://mouniat-vitiscanpro-diagno-api.hf.space/diagno \
  -H "accept: application/json" \
  -F "file=@leaf_photo.jpg"
```

**Response `200 OK`:**
```json
{
  "predictions": [
    {"disease": "plasmopara_viticola",         "confidence": 0.923},
    {"disease": "erysiphe_necator",             "confidence": 0.041},
    {"disease": "guignardia_bidwellii",         "confidence": 0.018},
    {"disease": "healthy",                      "confidence": 0.008},
    {"disease": "elsinoe_ampelina",             "confidence": 0.005},
    {"disease": "colomerus_vitis",              "confidence": 0.003},
    {"disease": "phaeomoniella_chlamydospora",  "confidence": 0.002}
  ],
  "model_version": "resnet18-vitiscan-v3"
}
```

**Response fields:**

| Field | Type | Description |
|---|---|---|
| `predictions` | list | All 7 classes ranked by confidence (highest first) |
| `predictions[].disease` | string | INRAE class name |
| `predictions[].confidence` | float | Probability score between 0.0 and 1.0 |
| `model_version` | string | Identifier of the model used for this prediction |

> **Note:** The sum of all confidence scores always equals 1.0 (softmax output).
> The first element of `predictions` is always the top prediction.

**Error responses:**

| Code | Cause | Response body |
|---|---|---|
| `400` | No file received | `{"message": "No file received"}` |
| `422` | Invalid request format | FastAPI validation error detail |
| `500` | Model inference error | `{"message": "Prediction failed. See server logs for details."}` |

---

## 2. Treatment Plan API

**Base URL (production):** `https://mouniat-vitiscanpro-solution-api.hf.space`
**Base URL (local):** `http://localhost:4001`

---

### GET /

Health check endpoint.

**Request:** no body required.

**Response `200 OK`:**
```json
{
  "message": "Vitiscan Treatment Plan API is running",
  "status": "ok"
}
```

---

### POST /solutions

Generates a complete treatment plan for a diagnosed disease.
Retrieves relevant agronomic knowledge from Weaviate and uses an LLM
to generate structured recommendations.

**Request:** `application/json`

| Field | Type | Required | Description |
|---|---|---|---|
| `cnn_label` | string | ✅ | INRAE disease class name from Diagnostic API |
| `mode` | string | ✅ | Farming mode: `"conventional"` or `"organic"` |
| `severity` | string | ✅ | Observed severity: `"low"`, `"moderate"`, or `"high"` |
| `area_m2` | float | ✅ | Vineyard surface area in square meters |
| `date_iso` | string | ✅ | Diagnosis date in ISO format `YYYY-MM-DD` (used to infer season) |
| `location` | string | ❌ | Vineyard location (optional, for context) |

**Example request body:**
```json
{
  "cnn_label": "plasmopara_viticola",
  "mode": "conventional",
  "severity": "moderate",
  "area_m2": 500,
  "date_iso": "2026-02-24",
  "location": "Bordeaux"
}
```

**Example (curl):**
```bash
curl -X POST https://mouniat-vitiscanpro-solution-api.hf.space/solutions \
  -H "Content-Type: application/json" \
  -d '{
    "cnn_label": "plasmopara_viticola",
    "mode": "conventional",
    "severity": "moderate",
    "area_m2": 500,
    "date_iso": "2026-02-24",
    "location": "Bordeaux"
  }'
```

**Response `200 OK`:**
```json
{
  "data": {
    "cnn_label":    "plasmopara_viticola",
    "disease_name": "Downy Mildew",
    "mode":         "conventional",
    "area_m2":      500,
    "severity":     "moderate",
    "season":       "winter",
    "treatment_plan": {
      "product":   "Mancozeb 80 WP",
      "dose_g":    400,
      "water_l":   100,
      "frequency": "every 10-14 days"
    },
    "diagnostic": "Downy mildew detected. Caused by Plasmopara viticola, a fungal pathogen that thrives in humid conditions. Immediate treatment is recommended.",
    "treatment_actions": [
      "Apply copper-based fungicide (Bordeaux mixture) immediately.",
      "Use mancozeb or cymoxanil as curative treatment.",
      "Remove and destroy severely infected leaves."
    ],
    "preventive_actions": [
      "Apply preventive fungicide treatments before forecasted rain periods.",
      "Ensure good air circulation through canopy management.",
      "Avoid leaf wetness by adjusting irrigation timing."
    ],
    "warnings": [
      "These recommendations are indicative only.",
      "Always verify local regulations and product labels before application.",
      "Copper applications are limited in organic farming — check local regulations."
    ],
    "raw_llm_output": "..."
  }
}
```

**Response fields:**

| Field | Type | Description |
|---|---|---|
| `cnn_label` | string | Input disease class, echoed back |
| `disease_name` | string | Human-readable disease name in English |
| `mode` | string | Input farming mode, echoed back |
| `area_m2` | float | Input surface area, echoed back |
| `severity` | string | Input severity, echoed back |
| `season` | string | Season inferred from `date_iso` |
| `treatment_plan` | object | Dosage rules (product, dose, frequency) |
| `diagnostic` | string | LLM-generated diagnostic description |
| `treatment_actions` | list[string] | Curative actions to take immediately |
| `preventive_actions` | list[string] | Long-term prevention measures |
| `warnings` | list[string] | Safety and regulatory warnings |
| `raw_llm_output` | string | Raw LLM output before parsing (for debugging) |

**Error responses:**

| Code | Cause |
|---|---|
| `422` | Missing required field or invalid type |
| `500` | LLM or pipeline error (fallback response returned instead when possible) |

> **Fallback mode:** When Weaviate Cloud is not configured, the API returns
> a static response based on the knowledge base instead of crashing.
> The `warnings` field will contain a note indicating fallback mode is active.

---

## 3. How Streamlit uses both APIs

This is the exact sequence implemented in the Streamlit `WebUI` repository:

```python
import requests

# Step 1 — Diagnose the leaf image
with open("leaf_photo.jpg", "rb") as f:
    diag_response = requests.post(
        "https://mouniat-vitiscanpro-diagno-api.hf.space/diagno",
        files={"file": f}
    )

diag_data    = diag_response.json()
top_disease  = diag_data["predictions"][0]["disease"]     # e.g. "plasmopara_viticola"
confidence   = diag_data["predictions"][0]["confidence"]  # e.g. 0.923
model_version = diag_data["model_version"]

# Step 2 — Get treatment plan for the diagnosed disease
treatment_response = requests.post(
    "https://mouniat-vitiscanpro-solution-api.hf.space/solutions",
    json={
        "cnn_label": top_disease,
        "mode":      "conventional",
        "severity":  "moderate",
        "area_m2":   100,
        "date_iso":  "2026-02-24",
        "location":  "Bordeaux"
    }
)

treatment_data = treatment_response.json()["data"]

# Step 3 — Display results in the UI
print(f"Diagnosed disease : {treatment_data['disease_name']} ({confidence:.0%} confidence)")
print(f"Diagnostic        : {treatment_data['diagnostic']}")
print(f"Treatment actions : {treatment_data['treatment_actions']}")
```