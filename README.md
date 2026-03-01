# Vitiscan — AI-Powered Grape Leaf Disease Diagnostic System

> End-to-end MLOps project for automated detection and treatment of grape vine diseases using computer vision and retrieval-augmented generation (RAG).

[![Python](https://img.shields.io/badge/Python-3.11-blue)](https://python.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.2-orange)](https://pytorch.org)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.115-green)](https://fastapi.tiangolo.com)
[![Airflow](https://img.shields.io/badge/Airflow-3.1.3-red)](https://airflow.apache.org)
[![MLflow](https://img.shields.io/badge/MLflow-2.22-purple)](https://mlflow.org)
[![Evidently](https://img.shields.io/badge/Evidently-0.4.33-teal)](https://evidentlyai.com)

---

## Table of Contents

- [Project Overview](#project-overview)
- [Live Demo](#live-demo)
- [Global Architecture](#global-architecture)
- [How It All Works Together](#how-it-all-works-together)
- [Repository Structure](#repository-structure)
- [Model Performance](#model-performance)
- [Technology Stack](#technology-stack)
- [Getting Started](#getting-started)
- [MLOps Pipeline](#mlops-pipeline)
- [Data Drift Detection](#data-drift-detection)
- [CI/CD Pipeline](#cicd-pipeline)
- [Disease Classes](#disease-classes)
- [Author](#author)

---

## Project Overview

Vitiscan is a production-ready MLOps system that allows vine growers to **diagnose grape leaf diseases from a photograph** and receive an **automated treatment plan** in seconds.

A grower takes a photo of a suspicious leaf → Vitiscan identifies the disease among 7 categories → A RAG-powered LLM generates a tailored treatment plan with dosage, prevention steps, and safety warnings.

Beyond the end-user application, Vitiscan demonstrates a complete MLOps lifecycle:
- Automated model retraining when new labeled data is available
- **Data drift detection** using Evidently to monitor image quality changes
- Continuous model performance monitoring
- Zero-downtime deployment with pre-production validation
- Full CI/CD pipeline for code quality and automated deployment

This project was built as a certification project in data science and MLOps engineering and serves as a professional portfolio demonstration.

---

## Live Demo

| Component | URL | 
|---|---|
| Web Application (Streamlit) | [vitiscan-streamlit.hf.space](https://mouniat-vitiscan-streamlit.hf.space) | 
| Diagnostic API | [vitiscanpro-diagno-api.hf.space/docs](https://mouniat-vitiscanpro-diagno-api.hf.space/docs) | 
| Treatment Plan API | [vitiscanpro-solution-api.hf.space/docs](https://mouniat-vitiscanpro-solution-api.hf.space/docs) | 
| MLflow Tracking | [vitiscanpro-hf.hf.space](https://mouniat-vitiscanpro-hf.hf.space) | 

---

## Global Architecture

```
┌────────────────────────────────────────────────────────────────────────┐
│                        USER INTERFACE                                  │
│                                                                        │
│                 Streamlit Web Application                              │
│         (photo upload → diagnosis display → treatment plan)            │
└────────────────────┬──────────────────────┬────────────────────────────┘
                     │                      │
                     ▼                      ▼
        ┌────────────────────┐   ┌──────────────────────────┐
        │   Diagnostic API   │   │   Treatment Plan API     │
        │   (FastAPI)        │   │   (FastAPI + RAG + LLM)  │
        │                    │   │                          │
        │ POST /diagno       │   │ POST /solutions          │
        │ → ResNet18 CNN     │   │ → Weaviate vector search │
        │ → 7 disease classes│   │ → LLM generation         │
        └────────┬───────────┘   └──────────┬───────────────┘
                 │                          │
                 ▼                          ▼
        ┌────────────────┐        ┌─────────────────────┐
        │  MLflow Server │        │  Weaviate Cloud     │
        │  (model store) │        │  (vector database)  │
        │  HF Spaces     │        │  knowledge base .md │
        └────────┬───────┘        └─────────────────────┘
                 │
                 │ model artifacts
                 ▼
┌───────────────────────────────────────────────────────────────────────┐
│                        MLOPS PIPELINE (Airflow)                       │
│                                                                       │
│   dag_monitoring (weekly)                                             │
│       │                                                               │
│       ├── Volume trigger (≥200 new images on S3) ──┐                  │
│       ├── Delay trigger (≥60 days since training) ─┤                  │
│       └── Performance check (F1 or Recall < 0.90) ─┼─ alert           │
│                                                    │                  │
│       Data Drift Detection (Evidently)             │                  │
│       └── Compare new images vs training set ──────┼─ drift alert    │
│           (brightness, contrast, colors, etc.)     │                  │
│                                                    ▼                  │
│   dag_data_ingestion                                                  │
│       │  validate → integrate → balance → update metadata             │
│       ▼                                                               │
│   dag_retraining                                                      │
│       │  train → compare → deploy preprod → test → deploy prod        │
│       └── or: keep current model if no improvement                    │
│                                                                       │
│                    AWS S3 (vitiscanpro-bucket)                        │
│          new-images/ → datasets/combined/ → mlflow-artifacts/         │
│                              └── monitoring/evidently/reports/        │
└───────────────────────────────────────────────────────────────────────┘

┌────────────────────────────────────────────────────────────────────────┐
│                      CI/CD PIPELINE (GitHub Actions)                   │
│                                                                        │
│   On push to main:                                                     │
│       unit-tests (Model-CNN) → integration-tests (API) → deploy (HF)   │
└────────────────────────────────────────────────────────────────────────┘
```

---

## How It All Works Together

Understanding how the 6 repositories interact is key to understanding Vitiscan.

### The user journey (front to back)

A grower opens the **Streamlit app** and uploads a photo of a suspicious leaf. Streamlit calls the **Diagnostic API** which loads the ResNet18 model from **MLflow** and returns a ranked list of disease predictions. Streamlit then calls the **Treatment Plan API** which queries the **Weaviate** vector database (built from agronomic knowledge base files), retrieves the most relevant chunks, and passes them to an LLM to generate a personalized treatment plan with dosage, prevention steps, and safety warnings.

### The automated model lifecycle (back end)

Every week, the **Airflow** `dag_monitoring` DAG performs a comprehensive health check:

1. **Volume & time triggers**: Are there enough new labeled images on S3 (≥200)? Has it been 60+ days since last training?

2. **Data drift detection**: Using **Evidently**, the pipeline extracts numerical features from new images (brightness, contrast, color distributions, aspect ratio) and compares them statistically against the training dataset reference. If more than 30% of features show significant drift, an alert is triggered — this could indicate camera changes, lighting conditions, or seasonal variations affecting image quality.

3. **Performance monitoring**: Has the production model's F1 score or recall dropped below 0.90 threshold?

If a retraining trigger is met, it automatically chains `dag_data_ingestion` (which validates and balances the new images) and then `dag_retraining` (which trains, compares, and deploys a new model to **MLflow** and **HuggingFace Spaces** — but only if it genuinely outperforms the current one).

### The code quality guardrail (CI/CD)

Every time a developer pushes code to GitHub, **GitHub Actions** automatically runs unit tests on the model scripts and integration tests on the Diagnostic API. Only if all tests pass, it automatically redeploy the API to HuggingFace Spaces. This guarantees that broken code never reaches production.

### The key architectural separation

```
GitHub Actions  →  protects CODE quality    (triggered by git push)
Airflow         →  manages MODEL lifecycle  (triggered by time + data)
Evidently       →  monitors DATA quality    (triggered by Airflow weekly)
```
These systems operate independently and only share HuggingFace Spaces and S3 as deployment/storage targets.

---

## Repository Structure

This project is organized across **6 specialized repositories**, each with a single responsibility:

| Repository | Role | Key Technologies |
|---|---|---|
| [Model_CNN](https://github.com/VITISCAN-PRO/Model_CNN) | ResNet18 model training, evaluation scripts, unit tests | PyTorch, MLflow, scikit-learn |
| [Diagnostic_API](https://github.com/VITISCAN-PRO/Diagnostic_API) | REST API for disease classification from leaf images | FastAPI, PyTorch, MLflow |
| [Treatment_Plan_API_RAG_LLM](https://github.com/VITISCAN-PRO/Treatment_Plan_API_RAG_LLM) | RAG pipeline for treatment plan generation | FastAPI, Weaviate, LLM |
| [WebUI_Streamlit](https://github.com/VITISCAN-PRO/WebUI_Streamlit) | User-facing web application | Streamlit, Python |
| [MLflow](https://github.com/VITISCAN-PRO/MLflow) | Experiment tracking server configuration | MLflow, HuggingFace Spaces |
| [Airflow](https://github.com/VITISCAN-PRO/Airflow) | Automated MLOps pipeline (3 DAGs) + drift detection | Apache Airflow, Evidently, AWS S3 |


### Detailed role of each repository

**Model_CNN** is the core machine learning component. It contains the ResNet18 fine-tuning scripts, the dataset loading utilities, the training configuration (`config.yml`), and all unit tests. Every model trained by the Airflow pipeline originates from this codebase.

**Diagnostic_API** exposes the trained model as a REST endpoint. It loads the production model from MLflow at startup and provides `/diagno` for image-based classification and `/diseases` for the list of detectable classes. It is automatically deployed to HuggingFace Spaces by GitHub Actions on every validated code push.

**Treatment_Plan_API_RAG_LLM** implements the RAG pipeline. It embeds the 7 agronomic knowledge base files into a Weaviate vector database, retrieves the most relevant chunks for a given disease, and uses an LLM to generate a structured treatment plan with diagnostic, treatment actions, preventive actions, and warnings.

**WebUI_Streamlit** is the end-user interface. It orchestrates the two API calls (diagnosis then treatment plan) and presents the results in a clean, accessible interface designed for vine growers with no technical background.

**MLflow** contains the configuration for the MLflow tracking server hosted on HuggingFace Spaces. It stores all experiment runs, hyperparameters, metrics, and model artifacts. It is the single source of truth for which model version is currently in production.

**Airflow** contains the 3 DAGs that automate the entire ML lifecycle, plus the **Evidently-based drift detection module**:
- `dag_monitoring` — runs weekly and decides if retraining is needed, includes data drift analysis
- `dag_data_ingestion` — validates, integrates, and balances new labeled images
- `dag_retraining` — trains, evaluates, and safely deploys improved models
- `utils/drift_detection.py` — Evidently integration for statistical drift analysis
- `scripts/generate_reference_features.py` — one-time script to create baseline feature dataset

---

## Model Performance

The ResNet18 model was fine-tuned on a combined dataset from INRAE scientific images and Kaggle healthy leaf images.

| Metric | Value |
|---|---|
| **Test Accuracy** | 98.3% |
| **Test F1 Score (macro)** | 0.982 |
| **Test Recall (macro)** | 0.983 |
| **Model architecture** | ResNet18 (fine-tuned) |
| **Input size** | 224 × 224 px |
| **Number of classes** | 7 |
| **Training images per class** | ~350 (balanced) |
| **Training approach** | Transfer learning from ImageNet |
| **Device** | MPS |

**Training strategy:** The ImageNet pre-trained weights were frozen for the first epochs to train only the classification head, then the full network was fine-tuned at a lower learning rate. Class balancing was applied via weighted sampling to handle minor imbalances in the dataset.

**Retraining triggers:** The production model is automatically retrained when F1 drops below 0.90, when recall drops below 0.90, when ≥ 200 new labeled images are available on S3, or when 60+ days have passed since the last training.

---

## Technology Stack

### Machine Learning
| Tool | Version | Purpose |
|---|---|---|
| PyTorch | 2.2 | Model training and inference |
| torchvision | 0.17 | ResNet18 architecture and image transforms |
| scikit-learn | 1.4 | Metrics computation (F1, recall, accuracy) |
| MLflow | 2.22 | Experiment tracking and model registry |

### APIs & Backend
| Tool | Version | Purpose |
|---|---|---|
| FastAPI | 0.115 | REST API framework |
| Uvicorn | 0.29 | ASGI server |
| Weaviate | 4.x | Vector database for RAG |
| sentence-transformers | 2.7 | Text embedding for RAG retrieval |
| Pydantic | 2.x | Data validation and schemas |

### MLOps & Infrastructure
| Tool | Version | Purpose |
|---|---|---|
| Apache Airflow | 3.1.3 | ML pipeline orchestration |
| **Evidently** | **0.4.33** | **Data drift detection and monitoring** |
| GitHub Actions | — | CI/CD for code quality and deployment |
| AWS S3 | — | Dataset and artifact storage |
| AWS EC2 | p3.2xlarge | GPU instance for model training |
| Docker / Docker Compose | — | Local Airflow environment |

### Deployment
| Tool | Purpose |
|---|---|
| HuggingFace Spaces | Hosting for all APIs, MLflow, and Streamlit app |
| HuggingFace Hub API | Automated deployment from GitHub Actions |

---

## Getting Started

> 📘 **For complete step-by-step instructions**, see [docs/setup_guide.md](docs/setup_guide.md)

This section provides a quick overview. The setup guide contains detailed instructions, troubleshooting, and production deployment steps.

### Prerequisites

- Python 3.11+
- Docker and Docker Compose
- AWS account (S3 + EC2 access)
- HuggingFace account
- Git

### Clone all repositories

```bash
# Create a project folder
mkdir vitiscan && cd vitiscan

# Clone each repository
git clone https://github.com/VITISCAN-PRO/Model_CNN.git
git clone https://github.com/VITISCAN-PRO/Diagnostic_API.git
git clone https://github.com/VITISCAN-PRO/Treatment_Plan_API_RAG_LLM.git
git clone https://github.com/VITISCAN-PRO/WebUI_Streamlit.git
git clone https://github.com/VITISCAN-PRO/MLflow.git
git clone https://github.com/VITISCAN-PRO/Airflow.git
```

### Run each component

**1. Diagnostic API (local)**
```bash
cd Diagnostic_API
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt
uvicorn app:app --reload --port 4000
# → http://localhost:4000/docs
```

**2. Treatment Plan API (local)**
```bash
cd Treatment_Plan_API_RAG_LLM
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt
# Start Weaviate first (requires Docker)
docker compose up weaviate -d
uvicorn app.main:app --reload --port 4001
# → http://localhost:4001/docs
```

**3. Streamlit app (local)**
```bash
cd WebUI_Streamlit
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt
streamlit run app.py
# → http://localhost:8501
```

**4. Airflow pipeline (local via Docker)**
```bash
cd Airflow
cp .env.template .env
# Edit .env with your AWS credentials and tokens
docker compose build
docker compose up -d
# → http://localhost:8081  (login: airflow / airflow)
```

**5. Initialize drift detection reference (one-time setup)**
```bash
# Enter the Airflow scheduler container
docker compose exec airflow-scheduler bash

# Generate reference features from training dataset
cd /opt/airflow/scripts
python generate_reference_features.py

# This creates s3://vitiscanpro-bucket/monitoring/reference_features.csv
```

### Required environment variables

Each component requires a `.env` file. Refer to the `.env.template` file in each repository for the full list. Key variables across the project:

| Variable | Used by | Description |
|---|---|---|
| `MLFLOW_TRACKING_URI` | Diagnostic API, Airflow | MLflow server URL |
| `MLFLOW_MODEL_URI` | Diagnostic API | URI of the production model |
| `AWS_ACCESS_KEY_ID` | Airflow, Diagnostic API | AWS credentials |
| `AWS_SECRET_ACCESS_KEY` | Airflow, Diagnostic API | AWS credentials |
| `S3_BUCKET_NAME` | Airflow, Diagnostic API | S3 bucket name |
| `WEAVIATE_URL` | Treatment Plan API | Weaviate Cloud cluster URL |
| `WEAVIATE_API_KEY` | Treatment Plan API | Weaviate Cloud API key |
| `HF_TOKEN` | GitHub Actions, Airflow | HuggingFace deployment token |
| `VITISCAN_DRIFT_DETECTION_ENABLED` | Airflow | Enable/disable drift detection (default: true) |
| `VITISCAN_DRIFT_THRESHOLD` | Airflow | Drift alert threshold (default: 0.3 = 30%) |
| `VITISCAN_MIN_IMAGES_FOR_DRIFT` | Airflow | Minimum images for drift analysis (default: 50) |

---

## MLOps Pipeline

The automated ML pipeline runs entirely inside Apache Airflow and consists of 3 DAGs that trigger each other in cascade.

```
Every Monday (scheduled)
        │
        ▼
dag_monitoring
        │
        ├── ≥200 new images on S3?  ──┐
        ├── ≥60 days since training? ─┤ → dag_data_ingestion
        └── Otherwise                        │
               │                             ▼
               ▼                       dag_retraining
       check_data_drift                      │
          (Evidently)                        │
               │                             │
        ┌──────┴──────┐              ┌───────┴───────┐
        ▼             ▼              ▼               ▼
   drift_alert   check_performance   New model    Not better?
                       │             better?           │
                ┌──────┴──────┐         │        keep current
                ▼             ▼         ▼
          perf_alert     no_action   deploy preprod
                                         │
                                    tests pass?
                                    ┌────┴────┐
                                   YES       NO
                                    │         │
                               deploy prod  rollback
```

For the full pipeline documentation, see the [Airflow repository README](https://github.com/VITISCAN-PRO/Airflow).

---

## Data Drift Detection

Vitiscan uses **Evidently** to detect when incoming images differ statistically from the training dataset. This is crucial for maintaining model reliability in production.

### Why drift detection matters

In agricultural applications, image characteristics can change due to:
- **Equipment changes**: New smartphone or camera with different color calibration
- **Seasonal variations**: Different lighting conditions across seasons
- **Geographic expansion**: Images from new regions with different soil/climate
- **User behavior**: Different photo angles or distances

If these changes are significant, the model may perform poorly on the new data distribution — even if test metrics looked good at training time.

### How it works

1. **Reference dataset**: A one-time script extracts numerical features from all training images and saves them to S3 as `reference_features.csv`

2. **Feature extraction**: For each image, Evidently computes:
   - `brightness` — average pixel intensity
   - `contrast` — standard deviation of pixel values
   - `aspect_ratio` — width/height ratio
   - `red_mean`, `green_mean`, `blue_mean` — average color channel values
   - `saturation` — color intensity
   - `file_size_kb` — compressed file size (proxy for image complexity)

3. **Statistical comparison**: Evidently applies statistical tests (Kolmogorov-Smirnov, chi-squared) to compare the distribution of each feature between the reference and current datasets

4. **Drift decision**: If more than 30% of features show statistically significant drift (configurable via `VITISCAN_DRIFT_THRESHOLD`), an alert is triggered

### Generated reports

Each drift analysis produces:
- **HTML report** — Interactive visualization archived on S3 (`monitoring/evidently/reports/`)
- **JSON results** — Machine-readable output for pipeline decisions

Example JSON output:
```json
{
  "dataset_drift": true,
  "drift_share": 0.43,
  "drifted_features": ["brightness", "contrast", "blue_mean"],
  "feature_drift_scores": {
    "brightness": {"drift_detected": true, "drift_score": 0.002},
    "contrast": {"drift_detected": true, "drift_score": 0.015},
    "aspect_ratio": {"drift_detected": false, "drift_score": 0.234}
  }
}
```

### Configuration

| Variable | Default | Description |
|----------|---------|-------------|
| `VITISCAN_DRIFT_DETECTION_ENABLED` | `true` | Enable/disable drift detection |
| `VITISCAN_DRIFT_THRESHOLD` | `0.3` | Alert threshold (30% of features drifted) |
| `VITISCAN_MIN_IMAGES_FOR_DRIFT` | `50` | Minimum images required for analysis |

### Recommended actions when drift is detected

1. **Investigate the cause**: Check recent image uploads for quality issues or equipment changes
2. **Evaluate impact**: Review model predictions on recent images for accuracy
3. **Decide on action**:
   - If temporary (e.g., one-time batch issue): ignore and monitor
   - If permanent (e.g., new camera standard): update reference dataset
   - If quality problem: fix at source before retraining

---

## CI/CD Pipeline

Every push to the `main` branch of the `Diagnostic_API` repository triggers a GitHub Actions workflow with 3 sequential jobs:

```
Push to main
      │
      ├── Job 1: Unit Tests (Model-CNN scripts)
      │         pytest Model-CNN/tests/ --cov-fail-under=70
      │
      ├── Job 2: Integration Tests (Diagnostic API endpoints)
      │         pytest Diagnostic-API/tests/test_api_integration.py
      │         (uses mock model — no GPU or cloud credentials needed)
      │
      └── Job 3: Deploy to HuggingFace Spaces
                (only if Jobs 1 AND 2 pass, only on push to main)
                huggingface_hub.upload_folder(...)
```

**Key principle:** broken code never reaches production. If any test fails, the deployment is blocked automatically.

---

## Disease Classes

The model detects **7 grape vine disease classes** from the INRAE scientific nomenclature:

| Class name | Common name | Type |
|---|---|---|
| `colomerus_vitis` | Grape erineum mite | Parasite |
| `elsinoe_ampelina` | Anthracnose | Fungal |
| `erysiphe_necator` | Powdery mildew | Fungal |
| `guignardia_bidwellii` | Black rot | Fungal |
| `healthy` | Healthy leaf | — |
| `phaeomoniella_chlamydospora` | Esca disease | Wood pathogen |
| `plasmopara_viticola` | Downy mildew | Fungal |

---

## Author

**Mounia Tonazzini** — Agronomist Engineer & Data Scientist and Data Engineer

This project was developed as a certification project in MLOps engineering and data science.

- HuggingFace: [huggingface.co/MouniaT](https://huggingface.co/MouniaT)
- LinkedIn: [linkedin.com/in/mounia-tonazzini](https://www.linkedin.com/in/mounia-tonazzini)
- GitHub: [github.com/VITISCAN-PRO](https://github.com/VITISCAN-PRO)
- Email: mounia.tonazzini@gmail.com