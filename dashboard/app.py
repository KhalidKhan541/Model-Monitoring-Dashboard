import streamlit as st
import pandas as pd
import numpy as np
import requests
from datetime import datetime, timedelta
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

st.set_page_config(
    page_title="Model Monitoring Dashboard",
    page_icon="📊",
    layout="wide"
)

st.title("📊 Model Monitoring Dashboard")
st.sidebar.header("Configuration")

API_BASE_URL = st.sidebar.text_input("API Base URL", "http://localhost:8001")

@st.cache_data(ttl=30)
def get_health_status():
    try:
        response = requests.get(f"{API_BASE_URL}/health/status", timeout=5)
        return response.json()
    except Exception as e:
        return {"status": "error", "error": str(e)}

@st.cache_data(ttl=30)
def get_model_list():
    try:
        response = requests.get(f"{API_BASE_URL}/models", timeout=5)
        return response.json().get("models", [])
    except Exception as e:
        return []

@st.cache_data(ttl=60)
def get_model_summary(model_name):
    try:
        response = requests.get(f"{API_BASE_URL}/models/{model_name}/summary", timeout=5)
        return response.json()
    except Exception as e:
        return {}

@st.cache_data(ttl=60)
def get_prediction_statistics(model_name, window_minutes):
    try:
        response = requests.get(
            f"{API_BASE_URL}/predictions/{model_name}/statistics",
            params={"window_minutes": window_minutes},
            timeout=5
        )
        return response.json()
    except Exception as e:
        return {}

@st.cache_data(ttl=60)
def get_drift_history(model_name, limit=50):
    try:
        response = requests.get(
            f"{API_BASE_URL}/drift/{model_name}/history",
            params={"limit": limit},
            timeout=5
        )
        return response.json().get("drift_alerts", [])
    except Exception as e:
        return []

@st.cache_data(ttl=60)
def get_latency_history(model_name, endpoint, limit=100):
    try:
        response = requests.get(
            f"{API_BASE_URL}/latency/{model_name}/{endpoint}/history",
            params={"limit": limit},
            timeout=5
        )
        return response.json().get("history", [])
    except Exception as e:
        return []

def main():
    health = get_health_status()
    
    st.subheader("System Status")
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        status = health.get("overall_status", "unknown")
        st.metric("System Status", status.upper())
    
    with col2:
        models = health.get("models", {})
        st.metric("Registered Models", len(models))
    
    with col3:
        critical_count = sum(1 for m in models.values() if m.get("status") == "critical")
        st.metric("Critical Models", critical_count)
    
    with col4:
        healthy_count = sum(1 for m in models.values() if m.get("status") == "healthy")
        st.metric("Healthy Models", healthy_count)
    
    st.divider()
    
    model_list = get_model_list()
    
    if not model_list:
        st.warning("No models registered. Please register a model first.")
        st.code("POST /models/register with {'model_name': 'your_model_name'}")
        return
    
    st.subheader("Model Overview")
    
    selected_model = st.selectbox("Select Model", model_list)
    
    window_hours = st.slider("Time Window (hours)", 1, 168, 24)
    window_minutes = window_hours * 60
    
    if selected_model:
        col1, col2, col3, col4 = st.columns(4)
        
        summary = get_model_summary(selected_model)
        pred_stats = get_prediction_statistics(selected_model, window_minutes)
        
        with col1:
            total_preds = pred_stats.get("total_predictions", 0)
            st.metric("Total Predictions", total_preds)
        
        with col2:
            mean_latency = pred_stats.get("statistics", {}).get("mean_latency", 0)
            st.metric("Avg Latency", f"{mean_latency:.2f}ms" if mean_latency else "N/A")
        
        with col3:
            p95_latency = pred_stats.get("statistics", {}).get("p95_latency", 0)
            st.metric("P95 Latency", f"{p95_latency:.2f}ms" if p95_latency else "N/A")
        
        with col4:
            drift_alerts = get_drift_history(selected_model, limit=10)
            st.metric("Recent Drift Alerts", len(drift_alerts))
        
        st.divider()
        
        tab1, tab2, tab3, tab4 = st.tabs(["📈 Prediction Metrics", "⏱️ Latency", "🔍 Drift Analysis", "📋 Raw Data"])
        
        with tab1:
            st.subheader("Prediction Distribution")
            
            if pred_stats and "prediction_distribution" in pred_stats:
                pred_dist = pred_stats["prediction_distribution"]
                
                col1, col2 = st.columns(2)
                
                with col1:
                    if "min" in pred_dist and "max" in pred_dist:
                        st.write(f"**Prediction Range:** {pred_dist['min']:.4f} - {pred_dist['max']:.4f}")
                        st.write(f"**Mean:** {pred_stats.get('statistics', {}).get('mean_prediction', 0):.4f}")
                        st.write(f"**Std:** {pred_stats.get('statistics', {}).get('std_prediction', 0):.4f}")
                
                with col2:
                    if "histogram" in pred_dist:
                        hist_data = pred_dist["histogram"]
                        fig = px.bar(
                            x=list(range(len(hist_data))),
                            y=hist_data,
                            title="Prediction Histogram",
                            labels={"x": "Bin", "y": "Count"}
                        )
                        st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("No prediction data available for this time window.")
        
        with tab2:
            st.subheader("Latency Metrics")
            
            endpoint = st.text_input("Endpoint", "/predict")
            
            latency_stats = get_latency_history(selected_model, endpoint)
            
            if latency_stats:
                latencies = [h.get("latency_ms", 0) for h in latency_stats]
                
                if latencies:
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.metric("Min Latency", f"{min(latencies):.2f}ms")
                        st.metric("Max Latency", f"{max(latencies):.2f}ms")
                        st.metric("Mean Latency", f"{np.mean(latencies):.2f}ms")
                    
                    with col2:
                        percentiles = [50, 90, 95, 99]
                        p_values = np.percentile(latencies, percentiles)
                        for p, val in zip(percentiles, p_values):
                            st.metric(f"P{p} Latency", f"{val:.2f}ms")
                    
                    fig = px.line(
                        x=list(range(len(latencies))),
                        y=latencies,
                        title="Latency Over Time",
                        labels={"x": "Request Index", "y": "Latency (ms)"}
                    )
                    st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("No latency data available.")
        
        with tab3:
            st.subheader("Drift Analysis")
            
            drift_alerts = get_drift_history(selected_model, limit=20)
            
            if drift_alerts:
                df_drift = pd.DataFrame(drift_alerts)
                
                if "drift_type" in df_drift.columns:
                    fig = px.histogram(
                        df_drift,
                        x="drift_type",
                        color="severity",
                        title="Drift Alerts by Type",
                        barmode="group"
                    )
                    st.plotly_chart(fig, use_container_width=True)
                
                if "created_at" in df_drift.columns:
                    df_drift["created_at"] = pd.to_datetime(df_drift["created_at"])
                    fig = px.scatter(
                        df_drift,
                        x="created_at",
                        y="drift_type",
                        color="severity",
                        title="Drift Alerts Timeline"
                    )
                    st.plotly_chart(fig, use_container_width=True)
                
                st.dataframe(df_drift)
            else:
                st.success("No drift alerts detected.")
        
        with tab4:
            st.subheader("Raw Statistics")
            
            st.json(pred_stats)
            
            with st.expander("Model Summary"):
                st.json(summary)

if __name__ == "__main__":
    main()