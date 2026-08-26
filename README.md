# Model Monitoring Dashboard

A comprehensive ML model monitoring solution with drift detection, latency tracking, and real-time visualization.

## Features

- **Drift Detection**: PSI, Kolmogorov-Smirnov, Jensen-Shannon, and Wasserstein distance metrics
- **Latency Tracking**: P50/P90/P95/P99 percentile monitoring with configurable alerts
- **Real-time Dashboard**: Streamlit-based interactive visualization
- **Grafana Integration**: Pre-configured dashboards for Prometheus metrics
- **Alerting**: Slack and email notifications for drift and latency anomalies
- **REST API**: FastAPI-based endpoints for all monitoring operations

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    Model Monitoring System                   │
├─────────────────────────────────────────────────────────────┤
│  ┌─────────────┐  ┌──────────────┐  ┌─────────────────┐   │
│  │   Drift     │  │   Latency    │  │    Metrics      │   │
│  │  Detector   │  │   Tracker    │  │   Collector     │   │
│  └─────────────┘  └──────────────┘  └─────────────────┘   │
│           │                │                │               │
│           └────────────────┼────────────────┘               │
│                            │                                │
│                    ┌───────▼───────┐                        │
│                    │ Model Monitor │                        │
│                    └───────┬───────┘                        │
│                            │                                │
│           ┌────────────────┼────────────────┐               │
│           │                │                │               │
│  ┌────────▼─────┐  ┌──────▼──────┐  ┌─────▼──────────┐   │
│  │   FastAPI    │  │  Streamlit  │  │    Grafana     │   │
│  │     API      │  │  Dashboard  │  │   Dashboard    │   │
│  └──────────────┘  └─────────────┘  └────────────────┘   │
└─────────────────────────────────────────────────────────────┘
```

## Quick Start

### Using Docker Compose (Recommended)

```bash
# Clone the repository
git clone https://github.com/KhalidKhan541/Model-Monitoring-Dashboard.git
cd Model-Monitoring-Dashboard

# Start all services
docker-compose up -d

# Access the services:
# - API: http://localhost:8001
# - Dashboard: http://localhost:8501
# - Grafana: http://localhost:3000 (admin/admin)
# - Prometheus: http://localhost:9090
```

### Manual Installation

```bash
# Install dependencies
pip install -r requirements.txt

# Set up PostgreSQL database
psql -U postgres -c "CREATE DATABASE model_monitoring;"
psql -U postgres -d model_monitoring -f migrations/init_db.sql

# Start the API server
uvicorn src.api:app --host 0.0.0.0 --port 8001

# Start the dashboard (in a separate terminal)
streamlit run dashboard/app.py --server.port 8501
```

## API Endpoints

### Models
- `POST /models/register` - Register a new model
- `GET /models` - List all models
- `GET /models/{model_name}/summary` - Get model summary
- `GET /models/{model_name}/health` - Get model health status

### Predictions
- `POST /predictions/log` - Log a prediction
- `POST /predictions/log/batch` - Log batch predictions
- `GET /predictions/{model_name}/statistics` - Get prediction statistics

### Drift Detection
- `POST /drift/check` - Check for data drift
- `GET /drift/{model_name}/history` - Get drift history

### Latency
- `GET /latency/{model_name}/{endpoint}` - Get latency statistics
- `GET /latency/{model_name}/{endpoint}/history` - Get latency history
- `GET /latency/{model_name}/{endpoint}/alerts` - Check latency alerts
- `GET /latency/{model_name}/{endpoint}/throughput` - Get throughput

### Health
- `GET /health/status` - Get system health status
- `GET /dashboard/{model_name}` - Get HTML dashboard

## Usage Example

### Register a Model

```python
import requests

response = requests.post("http://localhost:8001/models/register", 
    json={"model_name": "my_classifier", "config": {"type": "classification"}})
```

### Log Predictions

```python
response = requests.post("http://localhost:8001/predictions/log",
    json={
        "model_name": "my_classifier",
        "endpoint": "/predict",
        "prediction_value": 0.85,
        "prediction_class": 1,
        "confidence": 0.92,
        "latency_ms": 45.2,
        "features": {"feature1": 1.2, "feature2": 3.4}
    })
```

### Check for Drift

```python
import numpy as np
import pandas as pd

# Generate sample data
reference_data = pd.DataFrame({
    'feature1': np.random.normal(0, 1, 1000),
    'feature2': np.random.uniform(0, 10, 1000)
})

current_data = pd.DataFrame({
    'feature1': np.random.normal(0.5, 1.2, 500),  # Shifted mean
    'feature2': np.random.uniform(2, 12, 500)
})

response = requests.post("http://localhost:8001/drift/check",
    json={
        "model_name": "my_classifier",
        "data": current_data.to_dict('records')
    })
print(response.json())
```

## Configuration

Edit `config/monitoring_config.yaml` to customize:

```yaml
drift_detection:
  thresholds:
    psi_threshold: 0.2
    ks_threshold: 0.05

latency_tracking:
  alert_thresholds:
    p99_ms: 1000
    p95_ms: 500

alerting:
  slack_webhook: ${SLACK_WEBHOOK_URL}
```

## Environment Variables

- `POSTGRES_HOST` - PostgreSQL host
- `POSTGRES_PORT` - PostgreSQL port
- `POSTGRES_DB` - Database name
- `POSTGRES_USER` - Database user
- `POSTGRES_PASSWORD` - Database password
- `REDIS_HOST` - Redis host
- `SLACK_WEBHOOK_URL` - Slack webhook for alerts

## License

MIT License