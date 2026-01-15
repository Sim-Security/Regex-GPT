# RegexGPT Deployment Guide

This document covers deployment to GCP Cloud Run and MLflow experiment tracking setup.

## Table of Contents
- [Prerequisites](#prerequisites)
- [Local Development](#local-development)
- [MLflow Experiment Tracking](#mlflow-experiment-tracking)
- [GCP Cloud Run Deployment](#gcp-cloud-run-deployment)
- [CI/CD Pipeline](#cicd-pipeline)
- [Troubleshooting](#troubleshooting)

---

## Prerequisites

### Local Requirements
- Python 3.11+
- Docker (for container testing)
- NVIDIA GPU with CUDA support (for training)

### GCP Requirements
- GCP Project with billing enabled
- Cloud Run API enabled
- Cloud Build API enabled
- Container Registry or Artifact Registry enabled
- Service account with appropriate permissions

### GitHub Requirements (for CI/CD)
- Repository secrets configured:
  - `GCP_PROJECT_ID`: Your GCP project ID
  - `GCP_SA_KEY`: Service account JSON key with Cloud Run Admin permissions

---

## Local Development

### Install Dependencies
```bash
pip install -r requirements.txt
```

### Run Training (with MLflow tracking)
```bash
python train_hf.py
```

### Run the Gradio App Locally
```bash
python app.py
# App available at http://localhost:7860
```

### Test with Docker
```bash
# Build image
docker build -t regex-gpt .

# Run container
docker run -p 8080:8080 regex-gpt

# Access at http://localhost:8080
```

---

## MLflow Experiment Tracking

MLflow is integrated into the training pipeline to track:
- Hyperparameters (LoRA config, learning rate, batch size, etc.)
- Training metrics (loss, accuracy, runtime)
- Model artifacts (LoRA adapters, tokenizer)

### Local MLflow Tracking

By default, experiments are logged to `./mlruns` directory.

**View experiments:**
```bash
# Start MLflow UI
mlflow ui --backend-store-uri ./mlruns

# Open in browser: http://localhost:5000
```

### Remote MLflow Tracking Server

For team collaboration or production, use a remote MLflow server.

**Set up remote tracking:**
```bash
# Set environment variable before training
export MLFLOW_TRACKING_URI="http://your-mlflow-server:5000"

# Or use a cloud-hosted solution
export MLFLOW_TRACKING_URI="databricks://your-workspace"
```

**Deploy MLflow Server on GCP:**
```bash
# Using Cloud Run (simple setup)
gcloud run deploy mlflow-server \
  --image ghcr.io/mlflow/mlflow:latest \
  --port 5000 \
  --memory 2Gi \
  --allow-unauthenticated \
  --set-env-vars "BACKEND_STORE_URI=sqlite:///mlflow.db,ARTIFACT_ROOT=gs://your-bucket/mlflow-artifacts"
```

### Tracked Metrics

Each training run logs:

| Category | Metrics |
|----------|---------|
| Hyperparameters | model_name, lora_r, lora_alpha, lora_dropout, batch_size, learning_rate, num_epochs |
| Model Info | trainable_params, total_params, trainable_pct |
| Dataset | train_examples, val_examples |
| Training | train_loss, train_runtime, samples_per_second |
| Evaluation | eval_loss, eval_runtime |

### Compare Experiments

```bash
# List experiments
mlflow experiments search

# Compare runs
mlflow runs list --experiment-name "RegexGPT-FineTuning"
```

---

## GCP Cloud Run Deployment

### Manual Deployment

1. **Authenticate with GCP:**
```bash
gcloud auth login
gcloud config set project YOUR_PROJECT_ID
```

2. **Build and push the image:**
```bash
# Build
docker build -t gcr.io/YOUR_PROJECT_ID/regex-gpt .

# Push
docker push gcr.io/YOUR_PROJECT_ID/regex-gpt
```

3. **Deploy to Cloud Run:**
```bash
gcloud run deploy regex-gpt \
  --image gcr.io/YOUR_PROJECT_ID/regex-gpt \
  --region us-central1 \
  --platform managed \
  --memory 4Gi \
  --cpu 2 \
  --port 8080 \
  --allow-unauthenticated
```

### Using Cloud Build

Deploy using the included `cloudbuild.yaml`:

```bash
# Submit build
gcloud builds submit --config cloudbuild.yaml

# With substitutions
gcloud builds submit --config cloudbuild.yaml \
  --substitutions _SERVICE_NAME=regex-gpt,_REGION=us-central1
```

### Environment Variables

Configure these for production:

| Variable | Description | Default |
|----------|-------------|---------|
| `PORT` | Server port | 8080 |
| `MLFLOW_TRACKING_URI` | MLflow server URL | ./mlruns |
| `GRADIO_SERVER_NAME` | Server bind address | 0.0.0.0 |

---

## CI/CD Pipeline

The GitHub Actions workflow (`.github/workflows/deploy.yml`) automates:
1. Running tests and linting
2. Building Docker image
3. Pushing to GCR
4. Deploying to Cloud Run

### Setup

1. **Create service account:**
```bash
gcloud iam service-accounts create github-actions \
  --display-name "GitHub Actions"

# Grant permissions
gcloud projects add-iam-policy-binding YOUR_PROJECT_ID \
  --member="serviceAccount:github-actions@YOUR_PROJECT_ID.iam.gserviceaccount.com" \
  --role="roles/run.admin"

gcloud projects add-iam-policy-binding YOUR_PROJECT_ID \
  --member="serviceAccount:github-actions@YOUR_PROJECT_ID.iam.gserviceaccount.com" \
  --role="roles/storage.admin"

gcloud projects add-iam-policy-binding YOUR_PROJECT_ID \
  --member="serviceAccount:github-actions@YOUR_PROJECT_ID.iam.gserviceaccount.com" \
  --role="roles/iam.serviceAccountUser"
```

2. **Create and download key:**
```bash
gcloud iam service-accounts keys create key.json \
  --iam-account=github-actions@YOUR_PROJECT_ID.iam.gserviceaccount.com
```

3. **Add GitHub secrets:**
- Go to Repository Settings > Secrets and variables > Actions
- Add `GCP_PROJECT_ID`: Your project ID
- Add `GCP_SA_KEY`: Contents of `key.json`

### Workflow Triggers

- **Push to main**: Full build, test, and deploy
- **Pull request**: Run tests only (no deployment)

---

## Troubleshooting

### Common Issues

**1. Out of memory during training:**
```bash
# Reduce batch size
BATCH_SIZE=1
GRADIENT_ACCUMULATION=16
```

**2. Cloud Run cold start timeout:**
- Increase memory allocation to 4Gi+
- Set min-instances to 1 for warm start

**3. MLflow connection errors:**
```bash
# Check tracking URI
echo $MLFLOW_TRACKING_URI

# Test connection
mlflow experiments list
```

**4. Docker build fails:**
```bash
# Clear Docker cache
docker system prune -a

# Build with no cache
docker build --no-cache -t regex-gpt .
```

### Logs

**Cloud Run logs:**
```bash
gcloud logging read "resource.type=cloud_run_revision AND resource.labels.service_name=regex-gpt" --limit 50
```

**MLflow logs:**
```bash
# Check run details
mlflow runs describe --run-id YOUR_RUN_ID
```

---

## Architecture

```
+------------------+     +------------------+     +------------------+
|   GitHub Repo    | --> |   Cloud Build    | --> |   Cloud Run      |
|  (push to main)  |     |  (build & push)  |     |  (serve app)     |
+------------------+     +------------------+     +------------------+
                                                          |
                                                          v
                                                  +------------------+
                                                  |   Users/API      |
                                                  +------------------+

Training Pipeline:
+------------------+     +------------------+     +------------------+
|   train_hf.py    | --> |   MLflow Server  | --> |   Experiments    |
|  (run locally)   |     |  (track metrics) |     |  (compare runs)  |
+------------------+     +------------------+     +------------------+
```

---

## Contact

For issues or questions, open a GitHub issue or contact the maintainers.
