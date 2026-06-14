import streamlit as st

def render_shap_chart(shap_output: dict):
    """
    Renders a visual horizontal bar chart using Streamlit components to explain risk.
    """
    risk_level = shap_output.get("risk_level", "UNKNOWN")
    risk_score = shap_output.get("risk_score", 0.0)
    top_factors = shap_output.get("top_risk_factors", [])
    explanation = shap_output.get("explanation_text", "")
    
    # Define colors based on risk level
    color_map = {
        "LOW": "🟢",
        "MEDIUM": "🟡",
        "HIGH": "🟠",
        "CRITICAL": "🔴",
        "UNKNOWN": "⚪"
    }
    
    icon = color_map.get(risk_level, "⚪")
    
    st.markdown("### Risk Explainability (SHAP)")
    
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.metric(label="Predicted Risk Level", value=f"{icon} {risk_level}")
        st.metric(label="Risk Probability", value=f"{risk_score * 100:.1f}%")
        
    with col2:
        st.info(explanation)
        
    st.markdown("#### Top Contributing Factors")
    
    if not top_factors:
        st.write("No significant risk factors identified.")
        return
        
    for factor in top_factors:
        name = factor.get("factor", "Unknown")
        contribution = factor.get("contribution", 0.0)
        direction = factor.get("direction", "increases_risk")
        
        # Calculate percentage for progress bar (cap at 100%)
        # Normalizing contribution for visual display. Max realistic SHAP is around 2.0
        normalized_val = min(abs(contribution) / 2.0, 1.0)
        
        # We use custom HTML to create colored progress bars because st.progress doesn't 
        # let us change colors easily without custom CSS hacking that might break.
        
        color = "#ff4b4b" if direction == "increases_risk" else "#00cc66"
        sign = "+" if direction == "increases_risk" else "-"
        
        st.markdown(f"**{name}** ({sign}{abs(contribution):.2f})")
        
        progress_html = f"""
        <div style="width: 100%; background-color: #f0f2f6; border-radius: 5px; margin-bottom: 15px;">
            <div style="width: {normalized_val * 100}%; background-color: {color}; height: 10px; border-radius: 5px;"></div>
        </div>
        """
        st.markdown(progress_html, unsafe_allow_html=True)
