import time
import numpy as np
from typing import Dict, List, Any, Optional
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from collections import deque
import threading
import logging

logger = logging.getLogger(__name__)

@dataclass
class LatencyMetric:
    timestamp: datetime
    latency_ms: float
    endpoint: str
    model_name: Optional[str] = None
    batch_size: Optional[int] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

class LatencyTracker:
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.window_size = config.get('window_size', 1000)
        self.percentiles = config.get('percentiles', [50, 90, 95, 99])
        
        self.latency_history: Dict[str, deque] = {}
        self.raw_latencies: Dict[str, deque] = {}
        
        self._lock = threading.Lock()
        
        self.alert_thresholds = config.get('alert_thresholds', {
            'p99_ms': 1000,
            'p95_ms': 500,
            'p50_ms': 100,
            'mean_ms': 200
        })
    
    def _get_history_key(self, endpoint: str, model_name: Optional[str] = None) -> str:
        if model_name:
            return f"{endpoint}:{model_name}"
        return endpoint
    
    def record_latency(self, latency_ms: float, endpoint: str, 
                      model_name: Optional[str] = None,
                      batch_size: Optional[int] = None,
                      metadata: Optional[Dict[str, Any]] = None):
        key = self._get_history_key(endpoint, model_name)
        
        metric = LatencyMetric(
            timestamp=datetime.utcnow(),
            latency_ms=latency_ms,
            endpoint=endpoint,
            model_name=model_name,
            batch_size=batch_size,
            metadata=metadata or {}
        )
        
        with self._lock:
            if key not in self.latency_history:
                self.latency_history[key] = deque(maxlen=self.window_size)
                self.raw_latencies[key] = deque(maxlen=self.window_size * 10)
            
            self.latency_history[key].append(metric)
            self.raw_latencies[key].append(latency_ms)
    
    def start_timer(self) -> float:
        return time.time()
    
    def stop_timer(self, start_time: float, endpoint: str,
                  model_name: Optional[str] = None,
                  batch_size: Optional[int] = None) -> float:
        latency_ms = (time.time() - start_time) * 1000
        self.record_latency(latency_ms, endpoint, model_name, batch_size)
        return latency_ms
    
    def get_statistics(self, endpoint: str, model_name: Optional[str] = None,
                      window_minutes: Optional[int] = None) -> Dict[str, Any]:
        key = self._get_history_key(endpoint, model_name)
        
        with self._lock:
            if key not in self.raw_latencies or not self.raw_latencies[key]:
                return {
                    'endpoint': endpoint,
                    'model_name': model_name,
                    'count': 0,
                    'statistics': {}
                }
            
            latencies = list(self.raw_latencies[key])
            
            if window_minutes:
                cutoff = datetime.utcnow() - timedelta(minutes=window_minutes)
                history = self.latency_history.get(key, deque())
                latencies = [
                    m.latency_ms for m in history 
                    if m.timestamp >= cutoff
                ]
            
            if not latencies:
                return {
                    'endpoint': endpoint,
                    'model_name': model_name,
                    'count': 0,
                    'statistics': {}
                }
            
            latency_array = np.array(latencies)
            
            statistics = {
                'count': len(latencies),
                'mean': float(np.mean(latency_array)),
                'std': float(np.std(latency_array)),
                'min': float(np.min(latency_array)),
                'max': float(np.max(latency_array)),
                'median': float(np.median(latency_array)),
            }
            
            for p in self.percentiles:
                statistics[f'p{p}'] = float(np.percentile(latency_array, p))
            
            return {
                'endpoint': endpoint,
                'model_name': model_name,
                'count': len(latencies),
                'statistics': statistics,
                'window_minutes': window_minutes
            }
    
    def get_latency_history(self, endpoint: str, model_name: Optional[str] = None,
                           limit: int = 100) -> List[Dict[str, Any]]:
        key = self._get_history_key(endpoint, model_name)
        
        with self._lock:
            if key not in self.latency_history:
                return []
            
            history = list(self.latency_history[key])[-limit:]
            
            return [
                {
                    'timestamp': m.timestamp.isoformat(),
                    'latency_ms': m.latency_ms,
                    'endpoint': m.endpoint,
                    'model_name': m.model_name,
                    'batch_size': m.batch_size
                }
                for m in history
            ]
    
    def check_alerts(self, endpoint: str, model_name: Optional[str] = None) -> List[Dict[str, Any]]:
        stats = self.get_statistics(endpoint, model_name)
        
        if stats['count'] == 0:
            return []
        
        alerts = []
        statistics = stats['statistics']
        
        if statistics.get('p99', 0) > self.alert_thresholds['p99_ms']:
            alerts.append({
                'type': 'high_latency_p99',
                'severity': 'critical',
                'message': f"P99 latency ({statistics['p99']:.2f}ms) exceeds threshold ({self.alert_thresholds['p99_ms']}ms)",
                'value': statistics['p99'],
                'threshold': self.alert_thresholds['p99_ms']
            })
        
        if statistics.get('p95', 0) > self.alert_thresholds['p95_ms']:
            alerts.append({
                'type': 'high_latency_p95',
                'severity': 'warning',
                'message': f"P95 latency ({statistics['p95']:.2f}ms) exceeds threshold ({self.alert_thresholds['p95_ms']}ms)",
                'value': statistics['p95'],
                'threshold': self.alert_thresholds['p95_ms']
            })
        
        if statistics.get('p50', 0) > self.alert_thresholds['p50_ms']:
            alerts.append({
                'type': 'high_latency_p50',
                'severity': 'warning',
                'message': f"P50 latency ({statistics['p50']:.2f}ms) exceeds threshold ({self.alert_thresholds['p50_ms']}ms)",
                'value': statistics['p50'],
                'threshold': self.alert_thresholds['p50_ms']
            })
        
        if statistics.get('mean', 0) > self.alert_thresholds['mean_ms']:
            alerts.append({
                'type': 'high_latency_mean',
                'severity': 'info',
                'message': f"Mean latency ({statistics['mean']:.2f}ms) exceeds threshold ({self.alert_thresholds['mean_ms']}ms)",
                'value': statistics['mean'],
                'threshold': self.alert_thresholds['mean_ms']
            })
        
        return alerts
    
    def get_throughput(self, endpoint: str, model_name: Optional[str] = None,
                      window_seconds: int = 60) -> Dict[str, Any]:
        key = self._get_history_key(endpoint, model_name)
        
        with self._lock:
            if key not in self.latency_history:
                return {'requests_per_second': 0, 'total_requests': 0}
            
            cutoff = datetime.utcnow() - timedelta(seconds=window_seconds)
            recent = [
                m for m in self.latency_history[key]
                if m.timestamp >= cutoff
            ]
            
            rps = len(recent) / window_seconds if window_seconds > 0 else 0
            
            return {
                'requests_per_second': rps,
                'total_requests': len(recent),
                'window_seconds': window_seconds
            }
    
    def clear_history(self, endpoint: Optional[str] = None):
        with self._lock:
            if endpoint:
                keys_to_clear = [k for k in self.latency_history if k.startswith(endpoint)]
                for key in keys_to_clear:
                    del self.latency_history[key]
                    if key in self.raw_latencies:
                        del self.raw_latencies[key]
            else:
                self.latency_history.clear()
                self.raw_latencies.clear()
        
        logger.info(f"Cleared latency history for {endpoint or 'all endpoints'}")