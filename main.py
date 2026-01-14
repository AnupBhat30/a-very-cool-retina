import streamlit as st
import torch
import torch.nn as nn
import timm
import cv2
import numpy as np
from PIL import Image
import albumentations as A
from albumentations.pytorch import ToTensorV2
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import os
from pathlib import Path
from huggingface_hub import hf_hub_download

# Page config
st.set_page_config(
    page_title="Retinal Disease Detection",
    page_icon="👁️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS - Dark Theme
st.markdown("""
    <style>
    .main {
        background-color: #0e1117;
        color: #fafafa;
    }
    .stApp {
        background-color: #0e1117;
    }
    .stAlert {
        background-color: #1e2530;
        color: #fafafa;
    }
    h1 {
        color: #64b5f6;
        font-weight: 700;
    }
    h2, h3 {
        color: #90caf9;
    }
    p, div, span, label {
        color: #fafafa;
    }
    .disease-card {
        padding: 1rem;
        border-radius: 10px;
        margin: 0.5rem 0;
        box-shadow: 0 4px 6px rgba(0,0,0,0.3);
        color: #fafafa;
    }
    .high-risk {
        background-color: #5d1f1f;
        border-left: 5px solid #f44336;
        color: #fafafa;
    }
    .medium-risk {
        background-color: #5d4a1f;
        border-left: 5px solid #ff9800;
        color: #fafafa;
    }
    .low-risk {
        background-color: #1f5d2f;
        border-left: 5px solid #4caf50;
        color: #fafafa;
    }
    </style>
""", unsafe_allow_html=True)

# Disease information
DISEASE_INFO = {
    "DR": {
        "name": "Diabetic Retinopathy",
        "description": "Damage to blood vessels in retina due to diabetes",
        "severity": "High",
        "emoji": "🔴"
    },
    "ARMD": {
        "name": "Age-Related Macular Degeneration",
        "description": "Deterioration of central vision area",
        "severity": "High",
        "emoji": "🟠"
    },
    "MH": {
        "name": "Media Haze",
        "description": "Opacity in lens or vitreous reducing image clarity",
        "severity": "Medium",
        "emoji": "🟡"
    },
    "DN": {
        "name": "Drusen",
        "description": "Yellow deposits under retina",
        "severity": "Medium",
        "emoji": "🟡"
    },
    "MYA": {
        "name": "Myopia",
        "description": "Nearsightedness retinal changes",
        "severity": "Low",
        "emoji": "🟢"
    },
    "BRVO": {
        "name": "Branch Retinal Vein Occlusion",
        "description": "Blockage in retinal vein branches",
        "severity": "High",
        "emoji": "🔴"
    },
    "TSLN": {
        "name": "Tessellation",
        "description": "Visible choroidal vessels pattern",
        "severity": "Low",
        "emoji": "🟢"
    },
    "ODC": {
        "name": "Optic Disc Cupping",
        "description": "Enlarged optic nerve cup",
        "severity": "Medium",
        "emoji": "🟡"
    },
    "ODP": {
        "name": "Optic Disc Pallor",
        "description": "Pale optic disc indicating nerve damage",
        "severity": "High",
        "emoji": "🔴"
    },
    "ODE": {
        "name": "Optic Disc Edema",
        "description": "Swelling of optic nerve head",
        "severity": "High",
        "emoji": "🔴"
    }
}

DISEASE_LABELS = ["DR", "ARMD", "MH", "DN", "MYA", "BRVO", "TSLN", "ODC", "ODP", "ODE"]

@st.cache_resource
def load_model(model_name="efficientnet_b3_epoch6_best_odp(2).pth"):
    """Load the trained EfficientNet-B3 model from Kaggle or local storage"""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    model_path = Path(model_name)
    
    # If model doesn't exist locally, download from Kaggle
    if not model_path.exists():
        st.info("📥 Downloading model from Kaggle...")
        download_model_from_kaggle(model_name)
    
    # Create model architecture
    model = timm.create_model(
        "efficientnet_b3",
        pretrained=False,
        num_classes=10,
        in_chans=3
    )

    # Load weights
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()

    return model, device

def download_model_from_kaggle(model_filename):
    """Download model from Hugging Face Hub"""
    try:
        st.info(f"📥 Downloading {model_filename} from Hugging Face (this may take a few minutes)...")
        
        model_path = hf_hub_download(
            repo_id="anupbhat/effnet-Retinal-Fundus",
            filename=model_filename,
            cache_dir="."
        )
        
        # Copy to current directory
        import shutil
        if model_path != model_filename:
            shutil.copy(model_path, model_filename)
        
        st.success(f"✅ Model downloaded successfully!")
        
    except Exception as e:
        st.error(f"❌ Download failed: {str(e)}")
        st.info("Make sure huggingface_hub is installed: pip install huggingface_hub")
        raise

def apply_clahe(image):
    """Apply CLAHE preprocessing (MUST MATCH TRAINING)"""
    # Convert to LAB color space
    lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
    l, a, b = cv2.split(lab)
    
    # Apply CLAHE to L-channel
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    l = clahe.apply(l)
    
    # Merge and convert back
    lab = cv2.merge([l, a, b])
    image = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)
    return image

def preprocess_image(image, img_size=300, apply_clahe_preprocessing=False):
    """Preprocess image for model input"""
    # Convert PIL to numpy
    if isinstance(image, Image.Image):
        image = np.array(image)

    # Convert to RGB if needed
    if len(image.shape) == 2:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
    elif image.shape[2] == 4:
        image = cv2.cvtColor(image, cv2.COLOR_RGBA2RGB)

    # Apply CLAHE if enabled
    if apply_clahe_preprocessing:
        image = apply_clahe(image)
    
    # Resize to 300×300 for EfficientNet-B3
    image = cv2.resize(image, (img_size, img_size), interpolation=cv2.INTER_AREA)

    # Apply transforms
    transform = A.Compose([
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2()
    ])

    transformed = transform(image=image)
    image_tensor = transformed["image"].unsqueeze(0)

    return image_tensor

def predict(model, image_tensor, device):
    """Run inference with multi-label output"""
    with torch.no_grad():
        image_tensor = image_tensor.to(device)
        logits = model(image_tensor)
        probs = torch.sigmoid(logits).cpu().numpy()[0]

    return probs

def get_detected_diseases(predictions, threshold=0.5):
    """Get all diseases above threshold (multi-label)"""
    detected = []
    for idx, prob in enumerate(predictions):
        if prob >= threshold:
            detected.append((DISEASE_LABELS[idx], prob))
    return sorted(detected, key=lambda x: x[1], reverse=True)

def create_prediction_chart(predictions):
    """Create interactive bar chart for predictions"""
    df = pd.DataFrame({
        'Disease': [DISEASE_INFO[d]["name"] for d in DISEASE_LABELS],
        'Probability': predictions * 100,
        'Code': DISEASE_LABELS
    })

    df = df.sort_values('Probability', ascending=True)

    # Color based on probability
    colors = ['#f44336' if p > 50 else '#ff9800' if p > 30 else '#4caf50' 
              for p in df['Probability']]

    fig = go.Figure(go.Bar(
        x=df['Probability'],
        y=df['Disease'],
        orientation='h',
        marker=dict(color=colors),
        text=[f"{p:.1f}%" for p in df['Probability']],
        textposition='outside',
        hovertemplate='<b>%{y}</b><br>Probability: %{x:.2f}%<extra></extra>'
    ))

    fig.update_layout(
        title="Disease Detection Probabilities",
        xaxis_title="Probability (%)",
        yaxis_title="",
        height=500,
        showlegend=False,
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        font=dict(size=12),
        margin=dict(l=20, r=100, t=60, b=40)
    )

    fig.update_xaxes(range=[0, 105], gridcolor='lightgray')

    return fig

def create_risk_gauge(max_prob):
    """Create risk level gauge"""
    if max_prob > 0.5:
        color = "red"
        risk = "High Risk"
    elif max_prob > 0.3:
        color = "orange"
        risk = "Medium Risk"
    else:
        color = "green"
        risk = "Low Risk"

    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=max_prob * 100,
        title={'text': f"<b>{risk}</b>", 'font': {'size': 24}},
        gauge={
            'axis': {'range': [0, 100], 'tickwidth': 1},
            'bar': {'color': color},
            'steps': [
                {'range': [0, 30], 'color': '#e8f5e9'},
                {'range': [30, 50], 'color': '#fff3e0'},
                {'range': [50, 100], 'color': '#ffebee'}
            ],
            'threshold': {
                'line': {'color': "black", 'width': 4},
                'thickness': 0.75,
                'value': max_prob * 100
            }
        }
    ))

    fig.update_layout(
        height=300,
        margin=dict(l=20, r=20, t=60, b=20),
        paper_bgcolor='rgba(0,0,0,0)'
    )

    return fig

# Main app
def main():
    st.title("👁️ AI Retinal Disease Detection System")
    st.markdown("### EfficientNet-B3 Multi-Label Classifier")
    st.markdown("---")

    # Sidebar
    with st.sidebar:
        st.header("ℹ️ About")
        st.info(
            "This AI system detects 10 retinal diseases from fundus images using "
            "InceptionV3 with class-balanced loss achieving **93.5% validation AUC**."
        )

        st.header("🎯 Detected Diseases")
        for code, info in DISEASE_INFO.items():
            with st.expander(f"{info['emoji']} {info['name']}"):
                st.write(f"**Code:** {code}")
                st.write(f"**Severity:** {info['severity']}")
                st.write(info['description'])

        st.markdown("---")
        st.markdown("**Model:** EfficientNet-B3")
        st.markdown("**Input Size:** 300×300")
        st.markdown("**Architecture:** Deep CNN")

        # Model info
        st.header("📂 Model")
        st.info("✅ Model: EfficientNet-B3 from Hugging Face (anupbhat/effnet-Retinal-Fundus)")

    # Main content
    col1, col2 = st.columns([1, 1])

    with col1:
        st.header("📤 Upload Image")
        uploaded_file = st.file_uploader(
            "Choose a retinal fundus image",
            type=['jpg', 'jpeg', 'png'],
            help="Upload a clear fundus photograph of the retina"
        )

        if uploaded_file:
            image = Image.open(uploaded_file)
            st.image(image, caption="Uploaded Image", use_column_width=True)

            # Image info
            st.caption(f"Size: {image.size[0]}×{image.size[1]} | Format: {image.format}")

    with col2:
        st.header("🔍 Analysis")
        
        # Add CLAHE toggle
        use_clahe = st.checkbox("🔬 Apply CLAHE Enhancement", value=False, help="Enhance image contrast for better visibility")

        if uploaded_file:
            with st.spinner("🧠 Analyzing image..."):
                try:
                    # Load model (from Kaggle if not local)
                    model, device = load_model()

                    # Preprocess
                    image_tensor = preprocess_image(image, apply_clahe_preprocessing=use_clahe)

                    # Predict
                    predictions = predict(model, image_tensor, device)

                    # Get top prediction
                    max_idx = np.argmax(predictions)
                    max_prob = predictions[max_idx]
                    max_disease = DISEASE_LABELS[max_idx]

                    # Success message
                    st.success("✅ Analysis complete!")

                    # Risk gauge - show highest probability
                    highest_prob = np.max(predictions)
                    st.plotly_chart(create_risk_gauge(highest_prob), use_column_width=True)

                except Exception as e:
                    st.error(f"❌ Error during analysis: {str(e)}")
                    st.stop()

        elif not uploaded_file:
            st.info("👆 Upload an image to begin analysis")

    # Results section
    if uploaded_file:
        st.markdown("---")
        st.header("📊 Detailed Results")
        
        # Add threshold slider for multi-label detection
        st.subheader("⚙️ Detection Sensitivity")
        threshold = st.slider(
            "Adjust detection threshold:",
            min_value=0.0,
            max_value=1.0,
            value=0.5,
            step=0.05,
            help="Lower threshold = more diseases detected (higher sensitivity but more false positives). Higher = only confident predictions."
        )
        
        detected = get_detected_diseases(predictions, threshold=threshold)
        
        if detected:
            st.success(f"✅ Detected {len(detected)} disease(s) above {threshold:.0%} threshold")
        else:
            st.info(f"ℹ️ No diseases detected above {threshold:.0%} threshold")

        # Prediction chart
        fig = create_prediction_chart(predictions)
        st.plotly_chart(fig, use_column_width=True)

        # Detected diseases (multi-label)
        st.subheader("🎯 Detected Diseases")
        
        if detected:
            cols = st.columns(min(3, len(detected)))
            for idx, (disease_code, prob) in enumerate(detected[:3]):
                disease_info = DISEASE_INFO[disease_code]
                with cols[idx % 3]:
                    risk_class = "high-risk" if prob > 0.5 else "medium-risk" if prob > 0.3 else "low-risk"
                    st.markdown(f"""
                    <div class="disease-card {risk_class}">
                        <h3>{disease_info['emoji']} {disease_info['name']}</h3>
                        <h2>{prob*100:.1f}%</h2>
                        <p><b>Severity:</b> {disease_info['severity']}</p>
                        <p>{disease_info['description']}</p>
                    </div>
                    """, unsafe_allow_html=True)
        else:
            st.info("No diseases detected at current threshold")
        
        # All predictions ranked
        st.subheader("📈 All Disease Predictions")
        all_ranked = sorted(enumerate(predictions), key=lambda x: x[1], reverse=True)
        
        confidence_df = pd.DataFrame([
            {
                'Rank': i+1,
                'Disease': DISEASE_INFO[DISEASE_LABELS[idx]]["name"],
                'Code': DISEASE_LABELS[idx],
                'Probability': f"{prob*100:.1f}%",
                'Status': '✅ DETECTED' if prob >= threshold else '❌ Not detected'
            }
            for i, (idx, prob) in enumerate(all_ranked)
        ])
        st.dataframe(confidence_df, use_container_width=False, hide_index=True)

        # Download results
        st.markdown("---")
        st.subheader("💾 Export Results")

        export_df = pd.DataFrame({
            'Disease': [DISEASE_INFO[d]["name"] for d in DISEASE_LABELS],
            'Code': DISEASE_LABELS,
            'Probability (%)': (predictions * 100).round(2),
            'Detected': ['Yes' if d in [x[0] for x in detected] else 'No' for d in DISEASE_LABELS],
            'Severity': [DISEASE_INFO[d]["severity"] for d in DISEASE_LABELS]
        }).sort_values('Probability (%)', ascending=False)

        csv = export_df.to_csv(index=False)
        st.download_button(
            label="📥 Download Detailed Results (CSV)",
            data=csv,
            file_name="retinal_disease_predictions.csv",
            mime="text/csv"
        )

        # Medical disclaimer
        st.markdown("---")
        st.warning("""
        ⚠️ **Medical Disclaimer:** This is an AI research tool and should not replace professional 
        medical diagnosis. Always consult qualified healthcare professionals for medical decisions.
        """)

if __name__ == "__main__":
    main()