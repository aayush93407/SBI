import streamlit as st
import os
import subprocess

# Define paths to notebooks
notebooks = {
    "Health Insurance": "healt_from.ipynb",
    "Vehicle Insurance": "v.ipynb",
    "Cyber Fraud Insurance": "cyber.ipynb"
}

# Function to open notebook in VS Code
def open_in_vscode(notebook_name):
    try:
        file_path = notebooks[notebook_name]
        if os.path.exists(file_path):
            subprocess.run(["code", file_path], shell=True)
            st.success(f"Opening {notebook_name} notebook in VS Code...")
        else:
            st.error("File not found!")
    except Exception as e:
        st.error(f"Error opening file: {e}")

# Configure page settings
st.set_page_config(layout="wide")

# Custom CSS for styling
st.markdown("""
<style>
    html, body, [class*="css"] {
        font-family: 'Roboto', sans-serif;
        background-color: white !important;
        color: black !important;
    }

    h1 {
        text-align: center;
        color: black !important;
        font-size: 2.5rem;
        font-weight: 700;
    }

    .subtitle {
        text-align: center;
        color: grey !important;
        font-size: 1.1rem;
        margin-bottom: 5rem;
    }

    .pricing-card {
        background-color: black !important;
        border-radius: 1rem;
        padding: 2rem;
        box-shadow: 0 4px 6px rgba(255, 255, 255, 0.1);
        display: flex;
        flex-direction: column;
        text-align: center;
        transition: transform 0.3s ease, box-shadow 0.3s ease;
        margin-bottom: 2rem;
        border: 1px solid grey;
    }
    
    .pricing-card:hover {
        transform: translateY(-3px);
        box-shadow: 0 8px 15px rgba(255, 255, 255, 0.15);
    }

    .plan-name {
        font-size: 1.4rem;
        font-weight: 600;
        color: white !important;
        margin-bottom: 0.5rem;
    }

    .feature-list {
        list-style: none;
        padding: 0;
        margin: 0 0 2rem 0;
        text-align: left;
    }

    .feature-item {
        display: flex;
        align-items: center;
        margin-bottom: 0.75rem;
        font-size: 0.95rem;
        color: white !important;
    }

    .feature-item:before {
        content: "✓";
        color: #10B981 !important;
        font-weight: bold;
        margin-right: 0.5rem;
    }

    .stButton > button {
        width: 100%;
        background-color: #3B82F6;
        color: white;
        padding: 0.75rem 1.5rem;
        border-radius: 9999px;
        border: none;
        font-weight: 600;
        transition: background-color 0.3s ease;
        cursor: pointer;
        font-size: 1rem;
    }

    .stButton > button:hover {
        background-color: #2563EB;
    }
</style>
""", unsafe_allow_html=True)

# Page title and subtitle
st.markdown("<h1>Insurance Fraud Detection for SBI</h1>", unsafe_allow_html=True)
st.markdown("<h4 class='subtitle'>Pure Banking, Nothing Else</h4>", unsafe_allow_html=True)

# Create three columns for the insurance cards
col1, col2, col3 = st.columns(3)

# Define the updated insurance fraud detection plans
plans = {
    "Health Insurance": {
        "features": [
            "Detect identity fraud in insurance claims",
            "Handle large hospital datasets",
            "Identify mismatches in bills & insurance claims",
            "Basic fraud detection model"
        ]
    },
    "Vehicle Insurance": {
        "features": [
            "Analyze accident claim fraud",
            "Cross-check vehicle image",
            "Detect staged accidents & fake claims",
            "Advanced fraud detection algorithms"
        ]
    },
    "Cyber Fraud Insurance": {
        "features": [
            "Identify phishing & cyber insurance fraud",
            "Detect fraudulent transactions & identity theft",
            "Analyze large datasets of cyber attacks",
            "Real-time AI-driven fraud prevention"
        ]
    }
}

# Function to create an insurance card
def create_insurance_card(column, plan_name, plan_data):
    with column:
        st.markdown(f"""
        <div class="pricing-card">
            <div class="plan-name">{plan_name}</div>
            <ul class="feature-list">
                {"".join([f'<li class="feature-item">{feature}</li>' for feature in plan_data["features"]])}
            </ul>
        </div>
        """, unsafe_allow_html=True)
        
        if st.button(f"Open {plan_name}", key=f"btn_{plan_name.lower()}"):
            open_in_vscode(plan_name)

# Create the insurance cards in the three columns
create_insurance_card(col1, "Health Insurance", plans["Health Insurance"])
create_insurance_card(col2, "Vehicle Insurance", plans["Vehicle Insurance"])
create_insurance_card(col3, "Cyber Fraud Insurance", plans["Cyber Fraud Insurance"])
