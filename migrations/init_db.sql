-- Model Monitoring Database Schema
-- ================================

-- Create database (run separately)
-- CREATE DATABASE model_monitoring;

-- Prediction Logs Table
CREATE TABLE IF NOT EXISTS prediction_logs (
    id SERIAL PRIMARY KEY,
    model_name VARCHAR(255) NOT NULL,
    endpoint VARCHAR(255) NOT NULL,
    prediction_value FLOAT,
    prediction_class INTEGER,
    confidence FLOAT,
    latency_ms FLOAT,
    batch_size INTEGER,
    features JSONB,
    metadata JSONB,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Create indexes for prediction_logs
CREATE INDEX IF NOT EXISTS idx_prediction_logs_model_name ON prediction_logs(model_name);
CREATE INDEX IF NOT EXISTS idx_prediction_logs_endpoint ON prediction_logs(endpoint);
CREATE INDEX IF NOT EXISTS idx_prediction_logs_created_at ON prediction_logs(created_at);
CREATE INDEX IF NOT EXISTS idx_prediction_logs_model_created ON prediction_logs(model_name, created_at);

-- Drift Alerts Table
CREATE TABLE IF NOT EXISTS drift_alerts (
    id SERIAL PRIMARY KEY,
    model_name VARCHAR(255) NOT NULL,
    drift_type VARCHAR(100) NOT NULL,
    severity VARCHAR(50) NOT NULL,
    drifted_features JSONB,
    drift_scores JSONB,
    details JSONB,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    resolved_at TIMESTAMP
);

-- Create indexes for drift_alerts
CREATE INDEX IF NOT EXISTS idx_drift_alerts_model_name ON drift_alerts(model_name);
CREATE INDEX IF NOT EXISTS idx_drift_alerts_created_at ON drift_alerts(created_at);
CREATE INDEX IF NOT EXISTS idx_drift_alerts_severity ON drift_alerts(severity);
CREATE INDEX IF NOT EXISTS idx_drift_alerts_model_created ON drift_alerts(model_name, created_at);

-- Model Metrics Table
CREATE TABLE IF NOT EXISTS model_metrics (
    id SERIAL PRIMARY KEY,
    model_name VARCHAR(255) NOT NULL,
    metric_name VARCHAR(255) NOT NULL,
    metric_value FLOAT,
    dimensions JSONB,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Create indexes for model_metrics
CREATE INDEX IF NOT EXISTS idx_model_metrics_model_name ON model_metrics(model_name);
CREATE INDEX IF NOT EXISTS idx_model_metrics_metric_name ON model_metrics(metric_name);
CREATE INDEX IF NOT EXISTS idx_model_metrics_created_at ON model_metrics(created_at);
CREATE INDEX IF NOT EXISTS idx_model_metrics_model_metric ON model_metrics(model_name, metric_name);

-- Latency Tracking Table
CREATE TABLE IF NOT EXISTS latency_logs (
    id SERIAL PRIMARY KEY,
    model_name VARCHAR(255) NOT NULL,
    endpoint VARCHAR(255) NOT NULL,
    latency_ms FLOAT NOT NULL,
    batch_size INTEGER,
    metadata JSONB,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Create indexes for latency_logs
CREATE INDEX IF NOT EXISTS idx_latency_logs_model_name ON latency_logs(model_name);
CREATE INDEX IF NOT EXISTS idx_latency_logs_endpoint ON latency_logs(endpoint);
CREATE INDEX IF NOT EXISTS idx_latency_logs_created_at ON latency_logs(created_at);
CREATE INDEX IF NOT EXISTS idx_latency_logs_model_endpoint ON latency_logs(model_name, endpoint);

-- Model Registry Table
CREATE TABLE IF NOT EXISTS model_registry (
    id SERIAL PRIMARY KEY,
    model_name VARCHAR(255) UNIQUE NOT NULL,
    model_config JSONB,
    reference_data_path VARCHAR(500),
    status VARCHAR(50) DEFAULT 'active',
    registered_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Create index for model_registry
CREATE INDEX IF NOT EXISTS idx_model_registry_model_name ON model_registry(model_name);

-- Create view for recent drift alerts
CREATE OR REPLACE VIEW recent_drift_alerts AS
SELECT 
    model_name,
    drift_type,
    severity,
    drifted_features,
    drift_scores,
    created_at
FROM drift_alerts
WHERE created_at >= CURRENT_TIMESTAMP - INTERVAL '7 days'
ORDER BY created_at DESC;

-- Create view for model health summary
CREATE OR REPLACE VIEW model_health_summary AS
SELECT 
    mr.model_name,
    mr.status,
    COALESCE(alert_counts.critical_alerts, 0) as critical_alerts,
    COALESCE(alert_counts.warning_alerts, 0) as warning_alerts,
    COALESCE(pred_counts.total_predictions, 0) as predictions_24h,
    COALESCE(lat.avg_latency_ms, 0) as avg_latency_ms,
    mr.registered_at,
    mr.updated_at
FROM model_registry mr
LEFT JOIN (
    SELECT 
        model_name,
        SUM(CASE WHEN severity = 'critical' THEN 1 ELSE 0 END) as critical_alerts,
        SUM(CASE WHEN severity = 'warning' THEN 1 ELSE 0 END) as warning_alerts
    FROM drift_alerts
    WHERE created_at >= CURRENT_TIMESTAMP - INTERVAL '24 hours'
    GROUP BY model_name
) alert_counts ON mr.model_name = alert_counts.model_name
LEFT JOIN (
    SELECT 
        model_name,
        COUNT(*) as total_predictions
    FROM prediction_logs
    WHERE created_at >= CURRENT_TIMESTAMP - INTERVAL '24 hours'
    GROUP BY model_name
) pred_counts ON mr.model_name = pred_counts.model_name
LEFT JOIN (
    SELECT 
        model_name,
        AVG(latency_ms) as avg_latency_ms
    FROM latency_logs
    WHERE created_at >= CURRENT_TIMESTAMP - INTERVAL '1 hour'
    GROUP BY model_name
) lat ON mr.model_name = lat.model_name
ORDER BY mr.model_name;