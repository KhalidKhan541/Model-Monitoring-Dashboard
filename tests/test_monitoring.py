import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from unittest.mock import Mock, patch

from src.drift_detector import DriftDetector, DriftType
from src.latency_tracker import LatencyTracker
from src.monitor import ModelMonitor

@pytest.fixture
def sample_reference_data():
    np.random.seed(42)
    n_samples = 1000
    return pd.DataFrame({
        'feature1': np.random.normal(0, 1, n_samples),
        'feature2': np.random.uniform(0, 10, n_samples),
        'feature3': np.random.choice(['A', 'B', 'C'], n_samples),
        'feature4': np.random.exponential(2, n_samples)
    })

@pytest.fixture
def sample_current_data():
    np.random.seed(123)
    n_samples = 500
    return pd.DataFrame({
        'feature1': np.random.normal(0.5, 1.2, n_samples),
        'feature2': np.random.uniform(2, 12, n_samples),
        'feature3': np.random.choice(['A', 'B', 'C', 'D'], n_samples, p=[0.3, 0.3, 0.3, 0.1]),
        'feature4': np.random.exponential(3, n_samples)
    })

@pytest.fixture
def sample_predictions():
    np.random.seed(42)
    return np.random.normal(0.5, 0.1, 1000)

@pytest.fixture
def drift_detector():
    config = {
        'thresholds': {
            'psi_threshold': 0.2,
            'ks_threshold': 0.05,
            'js_threshold': 0.1,
            'feature_drift_threshold': 0.1
        }
    }
    return DriftDetector(config)

@pytest.fixture
def latency_tracker():
    config = {
        'window_size': 1000,
        'percentiles': [50, 90, 95, 99],
        'alert_thresholds': {
            'p99_ms': 1000,
            'p95_ms': 500,
            'p50_ms': 100,
            'mean_ms': 200
        }
    }
    return LatencyTracker(config)

class TestDriftDetector:
    def test_set_reference_data(self, drift_detector, sample_reference_data, sample_predictions):
        drift_detector.set_reference_data(sample_reference_data, sample_predictions)
        assert drift_detector.reference_data is not None
        assert len(drift_detector.reference_data) == len(sample_reference_data)
    
    def test_detect_no_drift_same_data(self, drift_detector, sample_reference_data):
        drift_detector.set_reference_data(sample_reference_data)
        result = drift_detector.detect_all(sample_reference_data)
        assert result.drift_type == DriftType.NONE
    
    def test_detect_feature_drift(self, drift_detector, sample_reference_data, sample_current_data):
        drift_detector.set_reference_data(sample_reference_data)
        result = drift_detector.detect_all(sample_current_data)
        assert result.drift_type != DriftType.NONE
        assert len(result.drifted_features) > 0
    
    def test_psi_calculation(self, drift_detector):
        ref = np.random.normal(0, 1, 1000)
        current = np.random.normal(0.5, 1, 1000)
        psi = drift_detector._calculate_psi(ref, current)
        assert 0 <= psi <= 1
    
    def test_kolmogorov_smirnov_test(self, drift_detector):
        ref = np.random.normal(0, 1, 1000)
        current = np.random.normal(0.5, 1, 1000)
        stat, p_value = drift_detector._kolmogorov_smirnov_test(ref, current)
        assert 0 <= stat <= 1
        assert 0 <= p_value <= 1
    
    def test_jensen_shannon_divergence(self, drift_detector):
        ref = np.random.normal(0, 1, 1000)
        current = np.random.normal(0.5, 1, 1000)
        js = drift_detector._jensen_shannon_divergence(ref, current)
        assert 0 <= js <= 1
    
    def test_wasserstein_distance(self, drift_detector):
        ref = np.random.normal(0, 1, 1000)
        current = np.random.normal(0.5, 1, 1000)
        wd = drift_detector._wasserstein_distance(ref, current)
        assert wd >= 0

class TestLatencyTracker:
    def test_record_latency(self, latency_tracker):
        latency_tracker.record_latency(100.0, "/predict")
        stats = latency_tracker.get_statistics("/predict")
        assert stats['count'] == 1
        assert stats['statistics']['mean'] == 100.0
    
    def test_multiple_latencies(self, latency_tracker):
        latencies = [50, 100, 150, 200, 250]
        for lat in latencies:
            latency_tracker.record_latency(lat, "/predict")
        
        stats = latency_tracker.get_statistics("/predict")
        assert stats['count'] == 5
        assert stats['statistics']['min'] == 50
        assert stats['statistics']['max'] == 250
    
    def test_percentiles(self, latency_tracker):
        for i in range(100):
            latency_tracker.record_latency(float(i), "/predict")
        
        stats = latency_tracker.get_statistics("/predict")
        assert 'p50' in stats['statistics']
        assert 'p90' in stats['statistics']
        assert 'p95' in stats['statistics']
        assert 'p99' in stats['statistics']
    
    def test_latency_history(self, latency_tracker):
        for i in range(10):
            latency_tracker.record_latency(float(i * 10), "/predict")
        
        history = latency_tracker.get_latency_history("/predict", limit=5)
        assert len(history) == 5
    
    def test_check_alerts(self, latency_tracker):
        latency_tracker.record_latency(1500.0, "/predict")
        alerts = latency_tracker.check_alerts("/predict")
        assert len(alerts) > 0
        assert alerts[0]['type'] == 'high_latency_p99'
    
    def test_clear_history(self, latency_tracker):
        latency_tracker.record_latency(100.0, "/predict")
        latency_tracker.clear_history("/predict")
        stats = latency_tracker.get_statistics("/predict")
        assert stats['count'] == 0
    
    def test_throughput(self, latency_tracker):
        for i in range(10):
            latency_tracker.record_latency(float(i), "/predict")
        throughput = latency_tracker.get_throughput("/predict", window_seconds=60)
        assert throughput['total_requests'] == 10

class TestModelMonitor:
    @patch('src.monitor.MetricsCollector')
    def test_register_model(self, mock_metrics_collector):
        monitor = ModelMonitor({})
        monitor.register_model("test_model", {"type": "classifier"})
        assert "test_model" in monitor.models
    
    @patch('src.monitor.MetricsCollector')
    def test_log_prediction(self, mock_metrics_collector):
        monitor = ModelMonitor({})
        monitor.register_model("test_model", {})
        monitor.log_prediction(
            model_name="test_model",
            endpoint="/predict",
            prediction_value=0.85,
            latency_ms=100.0
        )
        mock_metrics_collector.return_value.log_prediction.assert_called_once()
    
    @patch('src.monitor.MetricsCollector')
    def test_set_reference_data(self, mock_metrics_collector):
        monitor = ModelMonitor({})
        data = pd.DataFrame({'feature1': [1, 2, 3]})
        monitor.set_reference_data("test_model", data)
        assert monitor.models["test_model"]['reference_data'] is not None
    
    @patch('src.monitor.MetricsCollector')
    def test_get_health_status(self, mock_metrics_collector):
        monitor = ModelMonitor({})
        monitor.register_model("test_model", {})
        status = monitor.get_health_status()
        assert 'models' in status
        assert 'overall_status' in status

if __name__ == "__main__":
    pytest.main([__file__, "-v"])