from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
from typing import Dict, List, Any, Optional
from datetime import datetime, timedelta
import pandas as pd
import numpy as np

from .monitor import ModelMonitor

app = FastAPI(title="Model Monitoring Dashboard API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

monitor = None

class PredictionLogRequest(BaseModel):
    model_name: str
    endpoint: str
    prediction_value: float
    prediction_class: Optional[int] = None
    confidence: Optional[float] = None
    latency_ms: Optional[float] = None
    batch_size: Optional[int] = None
    features: Optional[Dict[str, Any]] = None

class BatchPredictionRequest(BaseModel):
    model_name: str
    endpoint: str
    predictions: List[Dict[str, Any]]

class DriftCheckRequest(BaseModel):
    model_name: str
    data: List[Dict[str, Any]]
    predictions: Optional[List[float]] = None

class ModelRegistrationRequest(BaseModel):
    model_name: str
    config: Optional[Dict[str, Any]] = {}

@app.on_event("startup")
def startup():
    global monitor
    from .metrics_collector import MetricsCollector
    monitor = ModelMonitor({
        'drift_detection': {
            'thresholds': {
                'psi_threshold': 0.2,
                'ks_threshold': 0.05,
                'js_threshold': 0.1
            }
        },
        'latency_tracking': {
            'window_size': 1000,
            'percentiles': [50, 90, 95, 99],
            'alert_thresholds': {
                'p99_ms': 1000,
                'p95_ms': 500,
                'p50_ms': 100,
                'mean_ms': 200
            }
        },
        'metrics_collector': {
            'database': {
                'host': 'localhost',
                'port': 5432,
                'database': 'model_monitoring',
                'user': 'monitoring_user',
                'password': 'monitoring_password'
            }
        }
    })

@app.get("/")
def root():
    return {"message": "Model Monitoring Dashboard API"}

@app.get("/health")
def health():
    return {
        "status": "healthy",
        "timestamp": datetime.utcnow().isoformat(),
        "monitor_status": monitor.get_health_status() if monitor else {}
    }

@app.post("/models/register")
def register_model(data: ModelRegistrationRequest):
    try:
        monitor.register_model(data.model_name, data.config)
        return {"status": "registered", "model_name": data.model_name}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.get("/models")
def list_models():
    return {"models": list(monitor.models.keys()) if monitor else []}

@app.get("/models/{model_name}/summary")
def get_model_summary(model_name: str):
    try:
        return monitor.get_model_summary(model_name)
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.get("/models/{model_name}/health")
def get_model_health(model_name: str):
    try:
        summary = monitor.get_model_summary(model_name)
        return {
            "model_name": model_name,
            "status": summary.get('status', 'unknown'),
            "details": summary
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.post("/predictions/log")
def log_prediction(data: PredictionLogRequest):
    try:
        monitor.log_prediction(
            model_name=data.model_name,
            endpoint=data.endpoint,
            prediction_value=data.prediction_value,
            prediction_class=data.prediction_class,
            confidence=data.confidence,
            latency_ms=data.latency_ms,
            batch_size=data.batch_size,
            features=data.features
        )
        return {"status": "logged"}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.post("/predictions/log/batch")
def log_batch_predictions(data: BatchPredictionRequest):
    try:
        monitor.log_batch_predictions(
            model_name=data.model_name,
            endpoint=data.endpoint,
            predictions=data.predictions
        )
        return {"status": "logged", "count": len(data.predictions)}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.get("/predictions/{model_name}/statistics")
def get_prediction_statistics(model_name: str, window_minutes: int = Query(60, ge=1, le=1440)):
    try:
        return monitor.get_prediction_statistics(model_name, window_minutes)
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.post("/drift/check")
def check_drift(data: DriftCheckRequest):
    try:
        df = pd.DataFrame(data.data)
        predictions = np.array(data.predictions) if data.predictions else None
        
        result = monitor.check_drift(data.model_name, df, predictions)
        
        return {
            "model_name": data.model_name,
            "drift_type": result.drift_type.value,
            "has_drift": result.drift_type.value != "none",
            "drifted_features": result.drifted_features,
            "drift_scores": result.drift_scores,
            "timestamp": result.timestamp.isoformat(),
            "details": result.details
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.get("/drift/{model_name}/history")
def get_drift_history(model_name: str, limit: int = Query(50, ge=1, le=500)):
    try:
        return {"drift_alerts": monitor.get_drift_history(model_name, limit)}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.get("/latency/{model_name}/{endpoint}")
def get_latency_statistics(model_name: str, endpoint: str, 
                          window_minutes: int = Query(60, ge=1, le=1440)):
    try:
        return monitor.get_latency_statistics(endpoint, model_name, window_minutes)
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.get("/latency/{model_name}/{endpoint}/history")
def get_latency_history(model_name: str, endpoint: str,
                       limit: int = Query(100, ge=1, le=1000)):
    try:
        return {"history": monitor.get_latency_history(endpoint, model_name, limit)}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.get("/latency/{model_name}/{endpoint}/alerts")
def check_latency_alerts(model_name: str, endpoint: str):
    try:
        return {"alerts": monitor.check_latency_alerts(endpoint, model_name)}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.get("/latency/{model_name}/{endpoint}/throughput")
def get_throughput(model_name: str, endpoint: str,
                  window_seconds: int = Query(60, ge=1, le=3600)):
    try:
        return monitor.get_throughput(endpoint, model_name, window_seconds)
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.get("/health/status")
def get_health_status():
    try:
        return monitor.get_health_status()
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.get("/dashboard/{model_name}", response_class=HTMLResponse)
def get_dashboard(model_name: str):
    html_content = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>Model Monitoring Dashboard - {model_name}</title>
        <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
        <style>
            body {{ font-family: Arial, sans-serif; margin: 20px; background: #f5f5f5; }}
            .container {{ max-width: 1400px; margin: 0 auto; }}
            .card {{ background: white; border-radius: 8px; padding: 20px; margin: 10px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }}
            .grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(300px, 1fr)); gap: 20px; }}
            h1 {{ color: #333; }}
            h2 {{ color: #666; font-size: 18px; }}
            .metric {{ font-size: 24px; font-weight: bold; color: #007bff; }}
            .status-healthy {{ color: #28a745; }}
            .status-warning {{ color: #ffc107; }}
            .status-critical {{ color: #dc3545; }}
            canvas {{ max-height: 300px; }}
        </style>
    </head>
    <body>
        <div class="container">
            <h1>Model Monitoring Dashboard</h1>
            <h2>Model: {model_name}</h2>
            
            <div class="grid">
                <div class="card">
                    <h2>Health Status</h2>
                    <div id="health-status" class="metric">Loading...</div>
                </div>
                <div class="card">
                    <h2>Total Predictions (24h)</h2>
                    <div id="total-predictions" class="metric">-</div>
                </div>
                <div class="card">
                    <h2>Average Latency</h2>
                    <div id="avg-latency" class="metric">-</div>
                </div>
                <div class="card">
                    <h2>Drift Alerts</h2>
                    <div id="drift-alerts" class="metric">-</div>
                </div>
            </div>
            
            <div class="grid">
                <div class="card">
                    <h2>Prediction Distribution</h2>
                    <canvas id="predictionChart"></canvas>
                </div>
                <div class="card">
                    <h2>Latency Distribution</h2>
                    <canvas id="latencyChart"></canvas>
                </div>
            </div>
            
            <div class="card">
                <h2>Recent Drift Alerts</h2>
                <div id="drift-table">Loading...</div>
            </div>
        </div>
        
        <script>
            async function loadDashboard() {{
                try {{
                    const summaryRes = await fetch('/models/{model_name}/summary');
                    const summary = await summaryRes.json();
                    
                    document.getElementById('health-status').textContent = summary.status || 'Unknown';
                    document.getElementById('health-status').className = 'metric status-' + (summary.status || 'unknown');
                    document.getElementById('total-predictions').textContent = summary.prediction_summary?.total_predictions || 0;
                    document.getElementById('avg-latency').textContent = (summary.prediction_summary?.statistics?.mean_latency || 0).toFixed(2) + 'ms';
                    document.getElementById('drift-alerts').textContent = summary.recent_drift_alerts?.length || 0;
                    
                    const driftRes = await fetch('/drift/{model_name}/history?limit=10');
                    const driftData = await driftRes.json();
                    
                    let tableHtml = '<table style="width:100%;border-collapse:collapse;"><tr><th style="border:1px solid #ddd;padding:8px;background:#f2f2f2;">Time</th><th style="border:1px solid #ddd;padding:8px;background:#f2f2f2;">Type</th><th style="border:1px solid #ddd;padding:8px;background:#f2f2f2;">Severity</th><th style="border:1px solid #ddd;padding:8px;background:#f2f2f2;">Features</th></tr>';
                    (driftData.drift_alerts || []).forEach(alert => {{
                        tableHtml += `<tr>
                            <td style="border:1px solid #ddd;padding:8px;">${{new Date(alert.created_at).toLocaleString()}}</td>
                            <td style="border:1px solid #ddd;padding:8px;">${{alert.drift_type}}</td>
                            <td style="border:1px solid #ddd;padding:8px;">${{alert.severity}}</td>
                            <td style="border:1px solid #ddd;padding:8px;">${{JSON.parse(alert.drifted_features || '[]').join(', ')}}</td>
                        </tr>`;
                    }});
                    tableHtml += '</table>';
                    document.getElementById('drift-table').innerHTML = tableHtml;
                    
                }} catch (error) {{
                    console.error('Error loading dashboard:', error);
                }}
            }}
            
            loadDashboard();
            setInterval(loadDashboard, 30000);
        </script>
    </body>
    </html>
    """
    return HTMLResponse(content=html_content)