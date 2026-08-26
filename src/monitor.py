import logging
from typing import Dict, List, Any, Optional
from datetime import datetime
import pandas as pd
import numpy as np

from .drift_detector import DriftDetector, DriftResult, DriftType
from .latency_tracker import LatencyTracker
from .metrics_collector import MetricsCollector

logger = logging.getLogger(__name__)

class ModelMonitor:
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        
        self.drift_detector = DriftDetector(config.get('drift_detection', {}))
        self.latency_tracker = LatencyTracker(config.get('latency_tracking', {}))
        self.metrics_collector = MetricsCollector(config.get('metrics_collector', {}))
        
        self.models: Dict[str, Dict[str, Any]] = {}
        
        logger.info("Model Monitor initialized")
    
    def register_model(self, model_name: str, model_config: Dict[str, Any]):
        self.models[model_name] = {
            'config': model_config,
            'registered_at': datetime.utcnow(),
            'reference_data': None,
            'reference_predictions': None
        }
        
        logger.info(f"Registered model: {model_name}")
    
    def set_reference_data(self, model_name: str, data: pd.DataFrame,
                          predictions: np.ndarray = None):
        if model_name not in self.models:
            self.register_model(model_name, {})
        
        self.drift_detector.set_reference_data(data, predictions)
        self.models[model_name]['reference_data'] = data
        self.models[model_name]['reference_predictions'] = predictions
        
        logger.info(f"Set reference data for {model_name}: {len(data)} rows")
    
    def log_prediction(self, model_name: str, endpoint: str,
                      prediction_value: float, prediction_class: int = None,
                      confidence: float = None, latency_ms: float = None,
                      batch_size: int = None, features: Dict[str, Any] = None):
        self.metrics_collector.log_prediction(
            model_name=model_name,
            endpoint=endpoint,
            prediction_value=prediction_value,
            prediction_class=prediction_class,
            confidence=confidence,
            latency_ms=latency_ms,
            batch_size=batch_size,
            features=features
        )
        
        if latency_ms is not None:
            self.latency_tracker.record_latency(
                latency_ms=latency_ms,
                endpoint=endpoint,
                model_name=model_name,
                batch_size=batch_size
            )
    
    def log_batch_predictions(self, model_name: str, endpoint: str,
                            predictions: List[Dict[str, Any]]):
        self.metrics_collector.log_batch_predictions(model_name, endpoint, predictions)
        
        for pred in predictions:
            if pred.get('latency_ms') is not None:
                self.latency_tracker.record_latency(
                    latency_ms=pred['latency_ms'],
                    endpoint=endpoint,
                    model_name=model_name,
                    batch_size=pred.get('batch_size')
                )
    
    def check_drift(self, model_name: str, current_data: pd.DataFrame,
                   current_predictions: np.ndarray = None) -> DriftResult:
        result = self.drift_detector.detect_all(current_data, current_predictions)
        
        if result.drift_type != DriftType.NONE:
            severity = 'critical' if result.drift_type == DriftType.PREDICTION_DRIFT else 'warning'
            
            self.metrics_collector.log_drift_alert(
                model_name=model_name,
                drift_type=result.drift_type.value,
                severity=severity,
                drifted_features=result.drifted_features,
                drift_scores=result.drift_scores,
                details=result.details
            )
            
            logger.warning(f"Drift detected for {model_name}: {result.drift_type.value}")
        
        return result
    
    def get_latency_statistics(self, endpoint: str, model_name: str = None,
                              window_minutes: int = 60) -> Dict[str, Any]:
        return self.latency_tracker.get_statistics(endpoint, model_name, window_minutes)
    
    def get_latency_history(self, endpoint: str, model_name: str = None,
                           limit: int = 100) -> List[Dict[str, Any]]:
        return self.latency_tracker.get_latency_history(endpoint, model_name, limit)
    
    def check_latency_alerts(self, endpoint: str, model_name: str = None) -> List[Dict[str, Any]]:
        return self.latency_tracker.check_alerts(endpoint, model_name)
    
    def get_model_summary(self, model_name: str) -> Dict[str, Any]:
        summary = self.metrics_collector.get_dashboard_summary(model_name)
        
        model_info = self.models.get(model_name, {})
        summary['model_info'] = {
            'registered_at': model_info.get('registered_at', '').isoformat() if model_info.get('registered_at') else None,
            'has_reference_data': model_info.get('reference_data') is not None,
            'has_reference_predictions': model_info.get('reference_predictions') is not None
        }
        
        return summary
    
    def get_prediction_statistics(self, model_name: str, 
                                 window_minutes: int = 60) -> Dict[str, Any]:
        return self.metrics_collector.get_prediction_statistics(model_name, window_minutes)
    
    def get_drift_history(self, model_name: str, limit: int = 50) -> List[Dict[str, Any]]:
        return self.metrics_collector.get_drift_alerts(model_name=model_name, limit=limit)
    
    def get_throughput(self, endpoint: str, model_name: str = None,
                      window_seconds: int = 60) -> Dict[str, Any]:
        return self.latency_tracker.get_throughput(endpoint, model_name, window_seconds)
    
    def get_health_status(self) -> Dict[str, Any]:
        models_status = {}
        
        for model_name in self.models:
            recent_drift = self.metrics_collector.get_drift_alerts(
                model_name=model_name, 
                limit=5
            )
            
            has_critical = any(a.get('severity') == 'critical' for a in recent_drift)
            
            models_status[model_name] = {
                'status': 'critical' if has_critical else 'healthy',
                'recent_drift_alerts': len(recent_drift),
                'has_reference_data': self.models[model_name].get('reference_data') is not None
            }
        
        return {
            'timestamp': datetime.utcnow().isoformat(),
            'models': models_status,
            'overall_status': 'critical' if any(
                s['status'] == 'critical' for s in models_status.values()
            ) else 'healthy'
        }