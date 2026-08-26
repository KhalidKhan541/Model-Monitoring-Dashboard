import json
import logging
from typing import Dict, List, Any, Optional
from datetime import datetime, timedelta
from sqlalchemy import create_engine, text
from sqlalchemy.orm import declarative_base
from sqlalchemy import Column, Integer, String, Float, DateTime, JSON
import pandas as pd

logger = logging.getLogger(__name__)

Base = declarative_base()

class PredictionLog(Base):
    __tablename__ = 'prediction_logs'
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    model_name = Column(String, nullable=False)
    endpoint = Column(String, nullable=False)
    prediction_value = Column(Float)
    prediction_class = Column(Integer)
    confidence = Column(Float)
    latency_ms = Column(Float)
    batch_size = Column(Integer)
    features = Column(JSON)
    metadata = Column(JSON)
    created_at = Column(DateTime, default=datetime.utcnow)

class DriftAlert(Base):
    __tablename__ = 'drift_alerts'
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    model_name = Column(String, nullable=False)
    drift_type = Column(String, nullable=False)
    severity = Column(String, nullable=False)
    drifted_features = Column(JSON)
    drift_scores = Column(JSON)
    details = Column(JSON)
    created_at = Column(DateTime, default=datetime.utcnow)
    resolved_at = Column(DateTime, nullable=True)

class ModelMetric(Base):
    __tablename__ = 'model_metrics'
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    model_name = Column(String, nullable=False)
    metric_name = Column(String, nullable=False)
    metric_value = Column(Float)
    dimensions = Column(JSON)
    created_at = Column(DateTime, default=datetime.utcnow)

class MetricsCollector:
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.db_config = config.get('database', {})
        self.engine = self._create_engine()
        self._init_db()
    
    def _create_engine(self):
        conn_str = (
            f"postgresql://{self.db_config.get('user', 'monitoring_user')}:{self.db_config.get('password', 'monitoring_password')}"
            f"@{self.db_config.get('host', 'localhost')}:{self.db_config.get('port', 5432)}/{self.db_config.get('database', 'model_monitoring')}"
        )
        return create_engine(conn_str)
    
    def _init_db(self):
        Base.metadata.create_all(self.engine)
        logger.info("Metrics collector database initialized")
    
    def log_prediction(self, model_name: str, endpoint: str,
                      prediction_value: float, prediction_class: int = None,
                      confidence: float = None, latency_ms: float = None,
                      batch_size: int = None, features: Dict[str, Any] = None,
                      metadata: Dict[str, Any] = None):
        with self.engine.connect() as conn:
            conn.execute(
                text("""
                    INSERT INTO prediction_logs 
                    (model_name, endpoint, prediction_value, prediction_class, 
                     confidence, latency_ms, batch_size, features, metadata)
                    VALUES (:model_name, :endpoint, :prediction_value, :prediction_class,
                            :confidence, :latency_ms, :batch_size, :features, :metadata)
                """),
                {
                    'model_name': model_name,
                    'endpoint': endpoint,
                    'prediction_value': prediction_value,
                    'prediction_class': prediction_class,
                    'confidence': confidence,
                    'latency_ms': latency_ms,
                    'batch_size': batch_size,
                    'features': json.dumps(features) if features else None,
                    'metadata': json.dumps(metadata) if metadata else None
                }
            )
            conn.commit()
    
    def log_batch_predictions(self, model_name: str, endpoint: str,
                            predictions: List[Dict[str, Any]]):
        with self.engine.connect() as conn:
            for pred in predictions:
                conn.execute(
                    text("""
                        INSERT INTO prediction_logs 
                        (model_name, endpoint, prediction_value, prediction_class,
                         confidence, latency_ms, batch_size, features, metadata)
                        VALUES (:model_name, :endpoint, :prediction_value, :prediction_class,
                                :confidence, :latency_ms, :batch_size, :features, :metadata)
                    """),
                    {
                        'model_name': model_name,
                        'endpoint': endpoint,
                        'prediction_value': pred.get('prediction_value'),
                        'prediction_class': pred.get('prediction_class'),
                        'confidence': pred.get('confidence'),
                        'latency_ms': pred.get('latency_ms'),
                        'batch_size': pred.get('batch_size'),
                        'features': json.dumps(pred.get('features')) if pred.get('features') else None,
                        'metadata': json.dumps(pred.get('metadata')) if pred.get('metadata') else None
                    }
                )
            conn.commit()
    
    def log_drift_alert(self, model_name: str, drift_type: str, severity: str,
                       drifted_features: List[str], drift_scores: Dict[str, Any],
                       details: Dict[str, Any] = None):
        with self.engine.connect() as conn:
            conn.execute(
                text("""
                    INSERT INTO drift_alerts 
                    (model_name, drift_type, severity, drifted_features, drift_scores, details)
                    VALUES (:model_name, :drift_type, :severity, :drifted_features, :drift_scores, :details)
                """),
                {
                    'model_name': model_name,
                    'drift_type': drift_type,
                    'severity': severity,
                    'drifted_features': json.dumps(drifted_features),
                    'drift_scores': json.dumps(drift_scores),
                    'details': json.dumps(details) if details else None
                }
            )
            conn.commit()
    
    def log_metric(self, model_name: str, metric_name: str, metric_value: float,
                  dimensions: Dict[str, Any] = None):
        with self.engine.connect() as conn:
            conn.execute(
                text("""
                    INSERT INTO model_metrics (model_name, metric_name, metric_value, dimensions)
                    VALUES (:model_name, :metric_name, :metric_value, :dimensions)
                """),
                {
                    'model_name': model_name,
                    'metric_name': metric_name,
                    'metric_value': metric_value,
                    'dimensions': json.dumps(dimensions) if dimensions else None
                }
            )
            conn.commit()
    
    def get_prediction_logs(self, model_name: str = None, endpoint: str = None,
                           start_time: datetime = None, end_time: datetime = None,
                           limit: int = 1000) -> pd.DataFrame:
        query = "SELECT * FROM prediction_logs WHERE 1=1"
        params = {}
        
        if model_name:
            query += " AND model_name = :model_name"
            params['model_name'] = model_name
        
        if endpoint:
            query += " AND endpoint = :endpoint"
            params['endpoint'] = endpoint
        
        if start_time:
            query += " AND created_at >= :start_time"
            params['start_time'] = start_time
        
        if end_time:
            query += " AND created_at <= :end_time"
            params['end_time'] = end_time
        
        query += f" ORDER BY created_at DESC LIMIT {limit}"
        
        return pd.read_sql(query, self.engine, params=params)
    
    def get_prediction_statistics(self, model_name: str, 
                                 window_minutes: int = 60) -> Dict[str, Any]:
        cutoff = datetime.utcnow() - timedelta(minutes=window_minutes)
        
        df = self.get_prediction_logs(model_name=model_name, start_time=cutoff)
        
        if df.empty:
            return {
                'model_name': model_name,
                'window_minutes': window_minutes,
                'total_predictions': 0,
                'statistics': {}
            }
        
        return {
            'model_name': model_name,
            'window_minutes': window_minutes,
            'total_predictions': len(df),
            'statistics': {
                'mean_prediction': float(df['prediction_value'].mean()),
                'std_prediction': float(df['prediction_value'].std()),
                'mean_confidence': float(df['confidence'].mean()) if 'confidence' in df.columns else None,
                'mean_latency': float(df['latency_ms'].mean()) if 'latency_ms' in df.columns else None,
                'p95_latency': float(df['latency_ms'].quantile(0.95)) if 'latency_ms' in df.columns else None,
                'p99_latency': float(df['latency_ms'].quantile(0.99)) if 'latency_ms' in df.columns else None,
            },
            'prediction_distribution': {
                'min': float(df['prediction_value'].min()),
                'max': float(df['prediction_value'].max()),
                'histogram': np.histogram(df['prediction_value'].values, bins=10)[0].tolist()
            } if len(df) > 0 else {}
        }
    
    def get_drift_alerts(self, model_name: str = None, 
                        start_time: datetime = None,
                        limit: int = 100) -> List[Dict[str, Any]]:
        query = "SELECT * FROM drift_alerts WHERE 1=1"
        params = {}
        
        if model_name:
            query += " AND model_name = :model_name"
            params['model_name'] = model_name
        
        if start_time:
            query += " AND created_at >= :start_time"
            params['start_time'] = start_time
        
        query += f" ORDER BY created_at DESC LIMIT {limit}"
        
        df = pd.read_sql(query, self.engine, params=params)
        
        return df.to_dict('records') if not df.empty else []
    
    def get_model_metrics(self, model_name: str, metric_name: str = None,
                         start_time: datetime = None,
                         limit: int = 1000) -> pd.DataFrame:
        query = "SELECT * FROM model_metrics WHERE model_name = :model_name"
        params = {'model_name': model_name}
        
        if metric_name:
            query += " AND metric_name = :metric_name"
            params['metric_name'] = metric_name
        
        if start_time:
            query += " AND created_at >= :start_time"
            params['start_time'] = start_time
        
        query += f" ORDER BY created_at DESC LIMIT {limit}"
        
        return pd.read_sql(query, self.engine, params=params)
    
    def get_dashboard_summary(self, model_name: str) -> Dict[str, Any]:
        prediction_stats = self.get_prediction_statistics(model_name, window_minutes=60)
        
        drift_alerts = self.get_drift_alerts(model_name=model_name, limit=10)
        
        recent_metrics = self.get_model_metrics(model_name, limit=100)
        
        return {
            'model_name': model_name,
            'timestamp': datetime.utcnow().isoformat(),
            'prediction_summary': prediction_stats,
            'recent_drift_alerts': drift_alerts,
            'metrics_count': len(recent_metrics),
            'status': 'healthy' if not any(a.get('severity') == 'critical' for a in drift_alerts) else 'critical'
        }