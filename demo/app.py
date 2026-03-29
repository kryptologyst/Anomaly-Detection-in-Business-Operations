"""Streamlit demo application for anomaly detection in operations.

This interactive demo allows users to explore anomaly detection
algorithms and visualize results.
"""

import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import sys
from pathlib import Path

# Add src to path
src_path = Path(__file__).parent.parent / "src"
sys.path.insert(0, str(src_path))

from data.synthetic_data import generate_operational_data
from models.anomaly_detector import AnomalyDetector
from evaluation.metrics import AnomalyEvaluator
from visualization.plots import AnomalyVisualizer

# Page configuration
st.set_page_config(
    page_title="Anomaly Detection in Operations",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .disclaimer {
        background-color: #ffebee;
        border-left: 5px solid #f44336;
        padding: 1rem;
        margin: 1rem 0;
        border-radius: 5px;
    }
    .metric-card {
        background-color: #f8f9fa;
        padding: 1rem;
        border-radius: 5px;
        border: 1px solid #dee2e6;
        margin: 0.5rem 0;
    }
</style>
""", unsafe_allow_html=True)

# Header
st.markdown('<h1 class="main-header">🔍 Anomaly Detection in Operations</h1>', unsafe_allow_html=True)

# Disclaimer
st.markdown("""
<div class="disclaimer">
    <h4>⚠️ IMPORTANT DISCLAIMER</h4>
    <p><strong>This is a research and educational tool. Do NOT use for automated decision-making without human review.</strong></p>
    <p>This software is provided for educational and research purposes only. It is not intended for production use in automated decision-making systems without proper human oversight, validation, and compliance with applicable regulations.</p>
</div>
""", unsafe_allow_html=True)

# Sidebar
st.sidebar.header("Configuration")

# Data generation parameters
st.sidebar.subheader("Data Parameters")
n_samples = st.sidebar.slider("Number of Samples", 100, 2000, 1000)
anomaly_rate = st.sidebar.slider("Anomaly Rate", 0.01, 0.20, 0.05)
include_equipment = st.sidebar.checkbox("Include Equipment IDs", value=True)
include_seasonality = st.sidebar.checkbox("Include Seasonality", value=True)
include_trend = st.sidebar.checkbox("Include Trend", value=True)

# Model parameters
st.sidebar.subheader("Model Parameters")
algorithm = st.sidebar.selectbox(
    "Algorithm",
    ["isolation_forest", "one_class_svm", "lof"],
    index=0
)

if algorithm == "isolation_forest":
    contamination = st.sidebar.slider("Contamination", 0.01, 0.20, 0.05)
    n_estimators = st.sidebar.slider("Number of Estimators", 50, 300, 100)
elif algorithm == "one_class_svm":
    nu = st.sidebar.slider("Nu (Upper bound on training errors)", 0.01, 0.50, 0.1)
    kernel = st.sidebar.selectbox("Kernel", ["rbf", "linear", "poly"], index=0)
elif algorithm == "lof":
    n_neighbors = st.sidebar.slider("Number of Neighbors", 5, 50, 20)
    contamination = st.sidebar.slider("Contamination", 0.01, 0.20, 0.05)

# Cost parameters
st.sidebar.subheader("Cost Parameters")
cost_fp = st.sidebar.number_input("False Positive Cost", value=10.0, min_value=0.0)
cost_fn = st.sidebar.number_input("False Negative Cost", value=100.0, min_value=0.0)

# Main content
if st.button("Generate Data and Train Model", type="primary"):
    
    with st.spinner("Generating data and training model..."):
        
        # Generate synthetic data
        data = generate_operational_data(
            n_samples=n_samples,
            anomaly_rate=anomaly_rate,
            include_equipment=include_equipment,
            include_seasonality=include_seasonality,
            include_trend=include_trend,
            random_state=42
        )
        
        # Prepare model parameters
        model_params = {"random_state": 42}
        
        if algorithm == "isolation_forest":
            model_params.update({
                "contamination": contamination,
                "n_estimators": n_estimators
            })
        elif algorithm == "one_class_svm":
            model_params.update({
                "nu": nu,
                "kernel": kernel
            })
        elif algorithm == "lof":
            model_params.update({
                "n_neighbors": n_neighbors,
                "contamination": contamination
            })
        
        # Train model
        detector = AnomalyDetector(algorithm=algorithm, **model_params)
        detector.fit(data['features'])
        
        # Make predictions
        predictions = detector.predict(data['features'])
        scores = detector.decision_function(data['features'])
        
        # Evaluate model
        evaluator = AnomalyEvaluator(cost_false_positive=cost_fp, cost_false_negative=cost_fn)
        metrics = evaluator.evaluate(data['labels'], predictions, scores)
        
        # Store results in session state
        st.session_state.data = data
        st.session_state.predictions = predictions
        st.session_state.scores = scores
        st.session_state.metrics = metrics
        st.session_state.algorithm = algorithm

# Display results if available
if 'data' in st.session_state:
    
    data = st.session_state.data
    predictions = st.session_state.predictions
    scores = st.session_state.scores
    metrics = st.session_state.metrics
    algorithm = st.session_state.algorithm
    
    # Metrics summary
    st.header("📊 Performance Metrics")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("F1-Score", f"{metrics['f1_score']:.3f}")
    with col2:
        st.metric("Precision", f"{metrics['precision']:.3f}")
    with col3:
        st.metric("Recall", f"{metrics['recall']:.3f}")
    with col4:
        st.metric("Alert Rate", f"{metrics['alert_rate']:.3f}")
    
    # Cost analysis
    st.subheader("💰 Cost Analysis")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Total Cost", f"${metrics['total_cost']:.2f}")
    with col2:
        st.metric("Cost per Sample", f"${metrics['cost_per_sample']:.3f}")
    with col3:
        st.metric("Alert Efficiency", f"{metrics['alert_efficiency']:.3f}")
    
    # Visualizations
    st.header("📈 Visualizations")
    
    # Time series plot
    st.subheader("Time Series with Anomalies")
    
    df = data['data'].copy()
    df['predicted_anomaly'] = predictions
    
    # Create interactive plot
    fig = go.Figure()
    
    # Normal points
    normal_mask = predictions == 0
    fig.add_trace(go.Scatter(
        x=df.index[normal_mask],
        y=df['metric_value'][normal_mask],
        mode='lines+markers',
        name='Normal',
        line=dict(color='blue', width=2),
        marker=dict(size=4),
        hovertemplate='<b>Normal</b><br>Time: %{x}<br>Value: %{y}<extra></extra>'
    ))
    
    # Predicted anomalies
    anomaly_mask = predictions == 1
    fig.add_trace(go.Scatter(
        x=df.index[anomaly_mask],
        y=df['metric_value'][anomaly_mask],
        mode='markers',
        name='Predicted Anomaly',
        marker=dict(color='red', size=12, symbol='x'),
        hovertemplate='<b>Predicted Anomaly</b><br>Time: %{x}<br>Value: %{y}<extra></extra>'
    ))
    
    # True anomalies (if available)
    true_anomaly_mask = data['labels'] == 1
    fig.add_trace(go.Scatter(
        x=df.index[true_anomaly_mask],
        y=df['metric_value'][true_anomaly_mask],
        mode='markers',
        name='True Anomaly',
        marker=dict(color='orange', size=10, symbol='circle'),
        hovertemplate='<b>True Anomaly</b><br>Time: %{x}<br>Value: %{y}<extra></extra>'
    ))
    
    fig.update_layout(
        title=f"Operational Metrics with Anomaly Detection - {algorithm.title()}",
        xaxis_title="Time",
        yaxis_title="Metric Value",
        hovermode='x unified',
        template='plotly_white',
        height=500
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Score distribution
    st.subheader("Anomaly Score Distribution")
    
    fig = go.Figure()
    
    normal_scores = scores[data['labels'] == 0]
    anomaly_scores = scores[data['labels'] == 1]
    
    fig.add_trace(go.Histogram(
        x=normal_scores,
        name='Normal',
        opacity=0.7,
        nbinsx=50,
        marker_color='blue'
    ))
    
    fig.add_trace(go.Histogram(
        x=anomaly_scores,
        name='Anomaly',
        opacity=0.7,
        nbinsx=50,
        marker_color='red'
    ))
    
    fig.update_layout(
        title="Distribution of Anomaly Scores",
        xaxis_title="Anomaly Score",
        yaxis_title="Frequency",
        template='plotly_white',
        height=400
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Confusion matrix
    st.subheader("Confusion Matrix")
    
    from sklearn.metrics import confusion_matrix
    
    cm = confusion_matrix(data['labels'], predictions)
    
    fig = go.Figure(data=go.Heatmap(
        z=cm,
        x=['Predicted Normal', 'Predicted Anomaly'],
        y=['Actual Normal', 'Actual Anomaly'],
        colorscale='Blues',
        text=cm,
        texttemplate="%{text}",
        textfont={"size": 20},
        hoverongaps=False
    ))
    
    fig.update_layout(
        title="Confusion Matrix",
        template='plotly_white',
        height=400
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Detailed metrics
    st.subheader("📋 Detailed Evaluation Report")
    
    report = evaluator.create_evaluation_report(
        data['labels'],
        predictions,
        scores,
        model_name=algorithm.title()
    )
    
    st.text(report)
    
    # Data summary
    st.subheader("📊 Data Summary")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.write("**Dataset Statistics:**")
        st.write(f"- Total samples: {len(data['features'])}")
        st.write(f"- True anomalies: {data['labels'].sum()}")
        st.write(f"- Predicted anomalies: {predictions.sum()}")
        st.write(f"- Anomaly rate: {data['labels'].mean():.3f}")
    
    with col2:
        st.write("**Feature Information:**")
        st.write(f"- Number of features: {data['features'].shape[1]}")
        st.write(f"- Feature names: {', '.join(data['feature_names'])}")

# Footer
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #666; font-size: 0.8rem;">
    <p>Anomaly Detection in Operations - Research & Educational Tool</p>
    <p>⚠️ Not for automated decision-making without human review</p>
</div>
""", unsafe_allow_html=True)
