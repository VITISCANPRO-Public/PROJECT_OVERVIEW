# Vitiscan — AI-Powered Grape Leaf Disease Diagnostic System

> End-to-end MLOps project for automated detection and treatment of grape vine diseases using computer vision and retrieval-augmented generation (RAG).

[![Python](https://img.shields.io/badge/Python-3.11-blue)](https://python.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.2-orange)](https://pytorch.org)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.110-green)](https://fastapi.tiangolo.com)
[![Airflow](https://img.shields.io/badge/Airflow-3.1-red)](https://airflow.apache.org)
[![MLflow](https://img.shields.io/badge/MLflow-2.13-purple)](https://mlflow.org)

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
- [CI/CD Pipeline](#cicd-pipeline)
- [Disease Classes](#disease-classes)
- [Author](#author)

---

## Project Overview

Vitiscan is a production-ready MLOps system that allows vine growers to **diagnose grape leaf diseases from a photograph** and receive an **automated treatment plan** in seconds.

A grower takes a photo of a suspicious leaf → Vitiscan identifies the disease among 7 categories → A RAG-powered LLM generates a tailored treatment plan with dosage, prevention steps, and safety warnings.

Beyond the end-user application, Vitiscan demonstrates a complete MLOps lifecycle:
- Automated model retraining when new labeled data is available
- Continuous model performance monitoring
- Zero-downtime deployment with pre-production validation
- Full CI/CD pipeline for code quality and automated deployment

This project was built as a certification project in MLOps engineering and serves as a professional portfolio demonstration.

---

## Live Demo

| Component | URL | 
|---|---|
| Web Application (Streamlit) | [vitiscanpro-webui.hf.space](https://mouniat-vitiscanpro-webui.hf.space) | 
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

Every week, the **Airflow** `dag_monitoring` DAG checks two things: are there enough new labeled images on S3 to justify retraining? Has the production model's F1 score dropped below threshold? If a retraining trigger is met, it automatically chains `dag_data_ingestion` (which validates and balances the new images) and then `dag_retraining` (which trains, compares, and deploys a new model to **MLflow** and **HuggingFace Spaces** — but only if it genuinely outperforms the current one).

### The code quality guardrail (CI/CD)

Every time a developer pushes code to GitHub, **GitHub Actions** automatically runs unit tests on the model scripts and integration tests on the Diagnostic API. Only if all tests pass does it automatically redeploy the API to HuggingFace Spaces. This guarantees that broken code never reaches production.

### The key architectural separation

```
GitHub Actions  →  protects CODE quality    (triggered by git push)
Airflow         →  manages MODEL lifecycle  (triggered by time + data)
```
These two systems operate independently and only share HuggingFace Spaces as a deployment target — GitHub Actions deploys the API code, Airflow deploys the model weights.

---

## Repository Structure

This project is organized across **6 specialized repositories**, each with a single responsibility:

| Repository | Role | Key Technologies |
|---|---|---|
| [Model_CNN](https://github.com/VITISCAN-PRO/Model_CNN.git) | ResNet18 model training, evaluation scripts, unit tests | PyTorch, MLflow, scikit-learn |
| [Diagnostic_API](https://github.com/VITISCAN-PRO/Diagnostic_API.git) | REST API for disease classification from leaf images | FastAPI, PyTorch, MLflow |
| [Treatment_Plan_API_RAG_LLM](https://github.com/VITISCAN-PRO/Treatment_Plan_API_RAG_LLM.git) | RAG pipeline for treatment plan generation | FastAPI, Weaviate, LLM |
| [WebUI_Streamlit](https://github.com/VITISCAN-PRO/WebUI_Streamlit.git) | User-facing web application | Streamlit, Python |
| [MLflow](https://github.com/VITISCAN-PRO/MLflow.git) | Experiment tracking server configuration | MLflow, HuggingFace Spaces |
| [Airflow](https://github.com/VITISCAN-PRO/Airflow.git) | Automated MLOps pipeline (3 DAGs) | Apache Airflow, AWS S3, boto3 |


### Detailed role of each repository

**Model_CNN** is the core machine learning component. It contains the ResNet18 fine-tuning scripts, the dataset loading utilities, the training configuration (`config.yml`), and all unit tests. Every model trained by the Airflow pipeline originates from this codebase.

**Diagnostic_API** exposes the trained model as a REST endpoint. It loads the production model from MLflow at startup and provides `/diagno` for image-based classification and `/diseases` for the list of detectable classes. It is automatically deployed to HuggingFace Spaces by GitHub Actions on every validated code push.

**Treatment_Plan_API_RAG_LLM** implements the RAG pipeline. It embeds the 7 agronomic knowledge base files into a Weaviate vector database, retrieves the most relevant chunks for a given disease, and uses an LLM to generate a structured treatment plan with diagnostic, treatment actions, preventive actions, and warnings.

**WebUI_Streamlit** is the end-user interface. It orchestrates the two API calls (diagnosis then treatment plan) and presents the results in a clean, accessible interface designed for vine growers with no technical background.

**MLflow** contains the configuration for the MLflow tracking server hosted on HuggingFace Spaces. It stores all experiment runs, hyperparameters, metrics, and model artifacts. It is the single source of truth for which model version is currently in production.

**Airflow** contains the 3 DAGs that automate the entire ML lifecycle. `dag_monitoring` runs weekly and decides if retraining is needed. `dag_data_ingestion` validates, integrates, and balances new labeled images. `dag_retraining` trains, evaluates, and safely deploys improved models — with pre-production testing and automatic rollback.

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
| MLflow | 2.13 | Experiment tracking and model registry |

### APIs & Backend
| Tool | Version | Purpose |
|---|---|---|
| FastAPI | 0.110 | REST API framework |
| Uvicorn | 0.29 | ASGI server |
| Weaviate | 4.x | Vector database for RAG |
| sentence-transformers | 2.7 | Text embedding for RAG retrieval |
| Pydantic | 2.x | Data validation and schemas |

### MLOps & Infrastructure
| Tool | Version | Purpose |
|---|---|---|
| Apache Airflow | 3.1 | ML pipeline orchestration |
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
git clone https://github.com/mouniat/Model_CNN.git
git clone https://github.com/mouniat/Diagnostic_API.git
git clone https://github.com/mouniat/Treatment_Plan_API_RAG_LLM.git
git clone https://github.com/mouniat/WebUI_Streamlit.git
git clone https://github.com/mouniat/MLflow.git
git clone https://github.com/mouniat/Airflow.git
```

### Run each component

**1. Diagnostic API (local)**
```bash
cd Diagnostic_API
conda activate vitiscan_api_diagno
pip install -r requirements.txt
uvicorn app:app --reload --port 4000
# → http://localhost:4000/docs
```

**2. Treatment Plan API (local)**
```bash
cd Treatment_Plan_API_RAG_LLM
pip install -r requirements.txt
# Start Weaviate first (requires Docker)
docker compose up weaviate -d
uvicorn app.main:app --reload --port 4001
# → http://localhost:4001/docs
```

**3. Streamlit app (local)**
```bash
cd WebUI_Streamlit
pip install -r requirements.txt
streamlit run app.py
# → http://localhost:8501
```

**4. Airflow pipeline (local via Docker)**
```bash
cd Airflow
docker compose build
docker compose up -d
# → http://localhost:8080  (login: airflow / airflow)
```

### Required environment variables

Each component requires a `.env` file. Refer to the `.env.example` file in each repository for the full list. Key variables across the project:

| Variable | Used by | Description |
|---|---|---|
| `MLFLOW_TRACKING_URI` | Diagnostic API, Airflow | MLflow server URL |
| `MLFLOW_MODEL_URI` | Diagnostic API | URI of the production model |
| `AWS_ACCESS_KEY_ID` | Airflow, Diagnostic API | AWS credentials |
| `AWS_SECRET_ACCESS_KEY` | Airflow, Diagnostic API | AWS credentials |
| `S3_BUCKET_NAME` | Airflow, Diagnostic API | S3 bucket name |
| `WEAVIATE_URL` | Treatment Plan API | Weaviate Cloud cluster URL |
| `WEAVIATE_API_KEY` | Treatment Plan API | Weaviate Cloud API key |
| `HF_TOKEN` | GitHub Actions | HuggingFace deployment token |

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
        └── F1 or Recall < 0.90?    ──┘        │
                                               ▼
                                      dag_retraining
                                             │
                              ┌──────────────┴──────────────┐
                              │                             │
                      New model better?              Not better?
                              │                             │
                    deploy to pre-prod            keep current model
                              │
                       tests pass?
                    ┌─────────┴──────────┐
                   YES                   NO
                    │                    │
              deploy to prod          rollback
```

For the full pipeline documentation, see the [Airflow repository README](https://github.com/mouniat/Airflow).

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
- LinkedIn: [www.linkedin.com/in/mounia-tonazzini](www.linkedin.com/in/mounia-tonazzini)
- GitHub: [github/Mounia-Agronomist-Datascientist](https://github.com/Mounia-Agronomist-Datascientist)
- Email : mounia.tonazzini@gmail.com