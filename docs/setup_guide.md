# Setup Guide — Vitiscan

Complete step-by-step guide to run the full Vitiscan stack locally
or deploy it to production on HuggingFace Spaces.

---

## Prerequisites

Before starting, make sure you have the following installed and configured:

| Tool | Version | Check |
|---|---|---|
| Python | 3.11+ | `python3 --version` |
| Conda | any | `conda --version` |
| Docker Desktop | 4.x+ | `docker --version` |
| Docker Compose | 2.x+ | `docker compose version` |
| Git | any | `git --version` |
| AWS CLI | 2.x | `aws --version` |

You also need accounts on:
- **AWS** — for S3 (dataset storage) and EC2 (model training)
- **HuggingFace** — for hosting the APIs, MLflow, and Streamlit app
- **GitHub** — for CI/CD with GitHub Actions

---

## Step 1 — Clone all repositories

```bash
# Create a root project folder
mkdir vitiscan && cd vitiscan

# Clone all 6 repositories
git clone https://github.com/VITISCAN-PRO/Model_CNN.git
git clone https://github.com/VITISCAN-PRO/Diagnostic_API.git
git clone https://github.com/VITISCAN-PRO/Treatment_Plan_API_RAG_LLM.git
git clone https://github.com/VITISCAN-PRO/WebUI_Streamlit.git
git clone https://github.com/VITISCAN-PRO/MLflow.git
git clone https://github.com/VITISCAN-PRO/Airflow.git
```

Your folder structure should look like:
```
vitiscan/
├── Model_CNN/
├── Diagnostic_API/
├── Treatment_Plan_API_RAG_LLM/
├── WebUI_Streamlit/
├── MLflow/
└── Airflow/
```

---

## Step 2 — Configure AWS credentials

All components that access S3 (Airflow DAGs, Diagnostic API) need AWS credentials.

```bash
aws configure
```

Enter when prompted:
```
AWS Access Key ID     : your_access_key
AWS Secret Access Key : your_secret_key
Default region name  : eu-west-3
Default output format : json
```

Verify access to your S3 bucket:
```bash
aws s3 ls s3://vitiscanpro-bucket/
```

You should see the bucket contents without any error.

---

## Step 3 — Create a Conda environment

One shared environment works for all Python components:

```bash
conda create -n vitiscan python=3.11 -y
conda activate vitiscan
```

Install dependencies for each component:
```bash
# Model CNN
pip install -r vitiscan/Model_CNN/requirements.txt

# Diagnostic API
pip install -r vitiscan/Diagnostic_API/requirements.txt

# Treatment Plan API
pip install -r vitiscan/Treatment_Plan_API_RAG_LLM/requirements.txt

# Streamlit
pip install -r vitiscan/WebUI_Streamlit/requirements.txt
```

---

## Step 4 — Configure environment variables

Each component needs a `.env` file. Copy the example file and fill in your values.

### Diagnostic API

```bash
cd vitiscan/Diagnostic_API
cp .env.template .env
```

Edit `.env`:
```bash
# --- MLFlow ---
MLFLOW_TRACKING_URI="https://******.hf.space"
MLFLOW_MODEL_URI="s3://your_bucket/mlflow-artifacts/models/******/artifacts"
S3_BUCKET_NAME="your_bucket"
MLFLOW_ARTIFACT_ROOT="s3://your-bucket/mlflow-artifacts/"

# --- Dataset ---

DATASET_NAME="inrae"

# --- AWS and S3 bucket CREDENTIALS ---
AWS_ACCESS_KEY_ID="your_key"
AWS_SECRET_ACCESS_KEY="your_access_key"
AWS_DEFAULT_REGION="your_region"
AWS_REGION="your_region"
```

### Treatment Plan API

```bash
cd vitiscan/Treatment_Plan_API_RAG_LLM
cp .env.template .env
```

Edit `.env`:
```bash
WEAVIATE_URL=          # leave empty for local dev, fill with cloud URL for production
WEAVIATE_API_KEY=      # leave empty for local dev
HF_API_TOKEN="hf**********************"
HF_MODEL_ID=meta-llama/Meta-Llama-3-8B-Instruct
```

### Airflow

```bash
cd vitiscan/Airflow
cp .env.template .env
```

Edit `.env`:
```bash
AIRFLOW_UID=50000
AWS_ACCESS_KEY_ID=your_access_key
AWS_SECRET_ACCESS_KEY=your_secret_key
AWS_DEFAULT_REGION=your_region
MLFLOW_TRACKING_URI=https://mouniat-vitiscanpro-hf.hf.space
HF_TOKEN=hf_your_token_here
NGROK_AUTHTOKEN=your_ngrok_token   # optional, for external access
```

---

## Step 5 — Run the Diagnostic API locally

```bash
conda activate vitiscan_api_diagno
cd vitiscan/Diagnostic_API
uvicorn app:app --reload --port 4000
```

Open `http://localhost:4000/docs` — you should see the FastAPI documentation page.

Test the health check:
```bash
curl http://localhost:4000/
# Expected: {"message": "Vitiscan Diagnostic API is running", "status": "ok"}
```

> **Note:** On first startup the API loads the ResNet18 model from MLflow.
> This takes 20-40 seconds. If MLflow is not reachable, set `TESTING=true`
> in your `.env` to use the mock model instead.

---

## Step 6 — Run the Treatment Plan API locally

Weaviate is required for full RAG functionality. Start it with Docker:

```bash
cd vitiscan/Treatment_Plan_API_RAG_LLM
docker compose up weaviate -d
```

Wait 20 seconds for Weaviate to initialize, then start the API:

```bash
conda activate vitiscan-rag
uvicorn app.main:app --reload --port 4001
```

Open `http://localhost:4001/docs`.

> **Without Weaviate:** If you skip the Docker step, the API will still start
> and return static fallback responses. This is the expected behavior in
> local development without Docker.

---

## Step 7 — Run Streamlit locally

```bash
conda activate vitiscan_streamlit
cd vitiscan/WebUI_Streamlit
streamlit run app.py
```

Open `http://localhost:8501`.

Make sure both APIs are running (steps 5 and 6) before testing Streamlit,
otherwise it will show connection errors.

---

## Step 8 — Run Airflow locally

```bash
cd vitiscan/Airflow

# Build the custom Docker image (includes mlflow, boto3, etc.)
docker compose build

# Start all services
docker compose up -d

# Follow startup logs
docker compose logs airflow-scheduler -f
```

Wait until you see:
```
scheduler | INFO - Starting the scheduler...
```

Open `http://localhost:8080` and log in with:
- Username: `airflow`
- Password: `airflow`

You should see 3 DAGs in the list: `dag_monitoring`, `dag_data_ingestion`, `dag_retraining`.

To test the pipeline manually:
```bash
# Upload a test image to S3
aws s3 cp vitiscan/Model_CNN/data-inrae/plasmopara_viticola/any_image.jpg \
  s3://vitiscanpro-bucket/new-images/plasmopara_viticola/test_001.jpg

# Then trigger dag_data_ingestion from the Airflow UI
```

---

## Step 9 — Deploy to HuggingFace Spaces (production)

Each API and the Streamlit app are deployed as separate HuggingFace Spaces.

### Diagnostic API

```bash
pip install huggingface_hub

python3 - <<EOF
from huggingface_hub import HfApi
api = HfApi()
api.upload_folder(
    folder_path="vitiscan/Diagnostic_API",
    repo_id="mouniat/vitiscanpro-diagno-api",
    repo_type="space",
    token="hf_your_token_here"
)
EOF
```

After deployment, configure these secrets in HuggingFace Space settings
(Settings → Repository secrets):

| Secret name | Value |
|---|---|
| `MLFLOW_TRACKING_URI` | `https://mouniat-vitiscanpro-hf.hf.space` |
| `MLFLOW_MODEL_URI` | `models:/resnet18-vitiscan/Production` |
| `AWS_ACCESS_KEY_ID` | your AWS key |
| `AWS_SECRET_ACCESS_KEY` | your AWS secret |
| `S3_BUCKET_NAME` | `vitiscanpro-bucket` |

### Treatment Plan API

Same process — upload the `Treatment_Plan_API_RAG_LLM` folder to its Space,
then configure:

| Secret name | Value |
|---|---|
| `WEAVIATE_URL` | your Weaviate Cloud cluster URL |
| `WEAVIATE_API_KEY` | your Weaviate Cloud API key |

### Streamlit app

Upload the `WebUI_Streamlit` folder to its Space, then configure:

| Secret name | Value |
|---|---|
| `DIAGNOSTIC_API_URL` | `https://mouniat-vitiscanpro-diagno-api.hf.space` |
| `TREATMENT_API_URL` | `https://mouniat-vitiscanpro-solution-api.hf.space` |

---

## Step 10 — Configure GitHub Actions (CI/CD)

Add these secrets to your `Diagnostic_API` GitHub repository
(Settings → Secrets and variables → Actions → New repository secret):

| Secret name | Value |
|---|---|
| `HF_TOKEN` | your HuggingFace token with write access |

From that point on, every push to `main` will automatically:
1. Run unit tests on `Model_CNN/tests/`
2. Run integration tests on `Diagnostic_API/tests/`
3. Deploy to HuggingFace Spaces if all tests pass

---

## Common issues and fixes

**`ModuleNotFoundError: No module named 'mlflow'` in Airflow**

The Docker image was not rebuilt after adding mlflow to `requirements.txt`. Run:
```bash
docker compose down
docker compose build
docker compose up -d
```

**`WEAVIATE_URL is missing in deployed environment`**

You have a `WEAVIATE_URL` secret set to `localhost` in HuggingFace.
Delete it from Settings → Repository secrets — the API will use the fallback mode instead.

**`Connection refused` when calling the Diagnostic API locally**

The API is not running. Start it with:
```bash
conda activate vitiscan
cd Diagnostic_API
uvicorn app:app --reload --port 4000
```

**HuggingFace Space shows `This Space is sleeping`**

Free Spaces sleep after inactivity. Click "Wake up this Space" and wait 1-2 minutes.

**`aws: command not found`**

Install the AWS CLI:
```bash
brew install awscli    # Mac
```
Then run `aws configure` with your credentials.

**GitHub Actions deploy job skipped**

The deploy job only runs on push to `main`, not on Pull Requests.
Make sure you are pushing directly to `main` and that both test jobs passed.