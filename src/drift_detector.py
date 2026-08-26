import numpy as np
import pandas as pd
from scipy import stats
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime, timedelta
import logging
from dataclasses import dataclass
from enum import Enum

logger = logging.getLogger(__name__)

class DriftType(Enum):
    NONE = "none"
    FEATURE_DRIFT = "feature_drift"
    PREDICTION_DRIFT = "prediction_drift"
    CONCEPT_DRIFT = "concept_drift"

@dataclass
class DriftResult:
    drift_type: DriftType
    drifted_features: List[str]
    drift_scores: Dict[str, float]
    p_values: Dict[str, float]
    thresholds: Dict[str, float]
    timestamp: datetime
    details: Dict[str, Any]

class DriftDetector:
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.thresholds = config.get('thresholds', {
            'psi_threshold': 0.2,
            'ks_threshold': 0.05,
            'js_threshold': 0.1,
            'feature_drift_threshold': 0.1,
            'prediction_drift_threshold': 0.05
        })
        self.reference_data: Optional[pd.DataFrame] = None
        self.reference_predictions: Optional[np.ndarray] = None
    
    def set_reference_data(self, data: pd.DataFrame, predictions: np.ndarray = None):
        self.reference_data = data.copy()
        self.reference_predictions = predictions.copy() if predictions is not None else None
        logger.info(f"Reference data set: {len(data)} rows, {len(data.columns)} features")
    
    def calculate_psi(self, expected: np.ndarray, actual: np.ndarray, buckets: int = 10) -> float:
        expected_pct = np.histogram(expected, bins=buckets)[0] / len(expected)
        actual_pct = np.histogram(actual, bins=buckets)[0] / len(actual)
        
        expected_pct = np.clip(expected_pct, 1e-6, None)
        actual_pct = np.clip(actual_pct, 1e-6, None)
        
        psi = np.sum((actual_pct - expected_pct) * np.log(actual_pct / expected_pct))
        return float(psi)
    
    def calculate_ks_test(self, reference: np.ndarray, current: np.ndarray) -> Tuple[float, float]:
        statistic, p_value = stats.ks_2samp(reference, current)
        return float(statistic), float(p_value)
    
    def calculate_jensen_shannon(self, reference: np.ndarray, current: np.ndarray, buckets: int = 50) -> float:
        ref_hist = np.histogram(reference, bins=buckets, density=True)[0]
        cur_hist = np.histogram(current, bins=buckets, density=True)[0]
        
        ref_hist = ref_hist / (ref_hist.sum() + 1e-10)
        cur_hist = cur_hist / (cur_hist.sum() + 1e-10)
        
        m = 0.5 * (ref_hist + cur_hist)
        
        m = np.clip(m, 1e-10, None)
        ref_hist = np.clip(ref_hist, 1e-10, None)
        cur_hist = np.clip(cur_hist, 1e-10, None)
        
        js_divergence = 0.5 * (stats.entropy(ref_hist, m) + stats.entropy(cur_hist, m))
        return float(js_divergence)
    
    def calculate_wasserstein(self, reference: np.ndarray, current: np.ndarray) -> float:
        return float(stats.wasserstein_distance(reference, current))
    
    def detect_feature_drift(self, current_data: pd.DataFrame) -> Dict[str, Any]:
        if self.reference_data is None:
            raise ValueError("Reference data not set")
        
        common_features = list(set(self.reference_data.columns) & set(current_data.columns))
        
        drift_scores = {}
        p_values = {}
        drifted_features = []
        
        for feature in common_features:
            ref_values = self.reference_data[feature].dropna().values
            cur_values = current_data[feature].dropna().values
            
            if len(ref_values) == 0 or len(cur_values) == 0:
                continue
            
            if np.issubdtype(ref_values.dtype, np.number):
                psi = self.calculate_psi(ref_values, cur_values)
                ks_stat, p_val = self.calculate_ks_test(ref_values, cur_values)
                js_div = self.calculate_jensen_shannon(ref_values, cur_values)
                
                drift_scores[feature] = {
                    'psi': psi,
                    'ks_statistic': ks_stat,
                    'js_divergence': js_div,
                    'wasserstein': self.calculate_wasserstein(ref_values, cur_values)
                }
                p_values[feature] = p_val
                
                if psi > self.thresholds['psi_threshold'] or p_val < self.thresholds['ks_threshold']:
                    drifted_features.append(feature)
        
        has_drift = len(drifted_features) > 0
        
        return {
            'has_drift': has_drift,
            'drift_type': DriftType.FEATURE_DRIFT if has_drift else DriftType.NONE,
            'drifted_features': drifted_features,
            'drift_scores': drift_scores,
            'p_values': p_values,
            'thresholds': self.thresholds,
            'timestamp': datetime.utcnow(),
            'details': {
                'total_features': len(common_features),
                'drifted_count': len(drifted_features),
                'drift_percentage': len(drifted_features) / max(len(common_features), 1) * 100
            }
        }
    
    def detect_prediction_drift(self, current_predictions: np.ndarray) -> Dict[str, Any]:
        if self.reference_predictions is None:
            raise ValueError("Reference predictions not set")
        
        psi = self.calculate_psi(self.reference_predictions, current_predictions)
        ks_stat, p_val = self.calculate_ks_test(self.reference_predictions, current_predictions)
        js_div = self.calculate_jensen_shannon(self.reference_predictions, current_predictions)
        
        has_drift = psi > self.thresholds['prediction_drift_threshold']
        
        return {
            'has_drift': has_drift,
            'drift_type': DriftType.PREDICTION_DRIFT if has_drift else DriftType.NONE,
            'drift_scores': {
                'psi': psi,
                'ks_statistic': ks_stat,
                'js_divergence': js_div,
                'wasserstein': self.calculate_wasserstein(self.reference_predictions, current_predictions)
            },
            'p_value': p_val,
            'thresholds': self.thresholds,
            'timestamp': datetime.utcnow(),
            'details': {
                'reference_mean': float(np.mean(self.reference_predictions)),
                'current_mean': float(np.mean(current_predictions)),
                'reference_std': float(np.std(self.reference_predictions)),
                'current_std': float(np.std(current_predictions))
            }
        }
    
    def detect_concept_drift(self, features: pd.DataFrame, predictions: np.ndarray, 
                           true_labels: np.ndarray) -> Dict[str, Any]:
        from sklearn.metrics import accuracy_score, f1_score
        
        current_accuracy = accuracy_score(true_labels, (predictions > 0.5).astype(int))
        current_f1 = f1_score(true_labels, (predictions > 0.5).astype(int), average='weighted')
        
        return {
            'has_drift': False,
            'drift_type': DriftType.CONCEPT_DRIFT,
            'metrics': {
                'accuracy': current_accuracy,
                'f1_score': current_f1
            },
            'timestamp': datetime.utcnow(),
            'details': {
                'true_labels_distribution': {
                    'positive': int(np.sum(true_labels)),
                    'negative': int(len(true_labels) - np.sum(true_labels))
                }
            }
        }
    
    def detect_all(self, current_data: pd.DataFrame, current_predictions: np.ndarray = None) -> DriftResult:
        feature_drift = self.detect_feature_drift(current_data)
        
        prediction_drift = None
        if current_predictions is not None and self.reference_predictions is not None:
            prediction_drift = self.detect_prediction_drift(current_predictions)
        
        drifted_features = feature_drift['drifted_features']
        drift_type = feature_drift['drift_type']
        
        if prediction_drift and prediction_drift['has_drift']:
            drift_type = DriftType.PREDICTION_DRIFT
        
        return DriftResult(
            drift_type=drift_type,
            drifted_features=drifted_features,
            drift_scores=feature_drift['drift_scores'],
            p_values=feature_drift['p_values'],
            thresholds=self.thresholds,
            timestamp=datetime.utcnow(),
            details={
                'feature_drift': feature_drift,
                'prediction_drift': prediction_drift
            }
        )