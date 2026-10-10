import os
import streamlit as st
import torch
import torchvision.transforms as transforms
from torchvision.models import resnet50
from PIL import Image

# Page Config
st.set_page_config(page_title="ADAS Road Hazard Detector", page_icon="🚗", layout="wide")

st.title("🚗 ADAS Hazard & Obstacle Classification System")
st.write("Upload a dashcam image to evaluate real-time road hazard alerts.")

# 1. Define Class Names and Hazard Logic
CIFAR10_CLASSES = (
    'plane', 'car', 'bird', 'cat', 'deer',
    'dog', 'frog', 'horse', 'ship', 'truck'
)

ADAS_HAZARD_MAP = {
    'plane': ('Low-Risk / Contextual Asset', 'info'),
    'car': ('Vehicle Hazard (Proximity Tracking)', 'warning'),
    'bird': ('Living Threat (Emergency Warning)', 'error'),
    'cat': ('Living Threat (Emergency Warning)', 'error'),
    'deer': ('Living Threat (Emergency Warning)', 'error'),
    'dog': ('Living Threat (Emergency Warning)', 'error'),
    'frog': ('Living Threat (Emergency Warning)', 'error'),
    'horse': ('Living Threat (Emergency Warning)', 'error'),
    'ship': ('Low-Risk / Contextual Asset', 'info'),
    'truck': ('Vehicle Hazard (Proximity Tracking)', 'warning')
}

# 2. Load Model Function
MODEL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "resnet50_cifar10.pth")

@st.cache_resource
def load_model():
    model = resnet50(weights=None)
    model.fc = torch.nn.Linear(model.fc.in_features, 10)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=torch.device('cpu')))
    model.eval()
    return model

model = load_model()

# 3. Image Preprocessing Transformation (Center-Crop Fix for Wide-Angle Images)
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

# 4. Upload & Classification UI
# 4. Upload & Classification UI
CONF_THRESHOLD = st.sidebar.slider("Minimum confidence to accept a class (%)", 50, 99, 85) / 100
MARGIN_THRESHOLD = 0.30  # top-1 must beat top-2 by at least this much

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert('RGB')

    col1, col2 = st.columns(2)

    with col1:
        st.image(image, caption="Uploaded Image", width="stretch")

    with col2:
        st.subheader("Inference & Hazard Result")

        img_tensor = transform(image).unsqueeze(0)
        with torch.no_grad():
            outputs = model(img_tensor)
            probabilities = torch.nn.functional.softmax(outputs, dim=1)[0]

        top_probs, top_idx = torch.topk(probabilities, 3)
        confidence = top_probs[0].item()
        margin = (top_probs[0] - top_probs[1]).item()
        pred_class = CIFAR10_CLASSES[top_idx[0].item()]

        is_unknown = confidence < CONF_THRESHOLD or margin < MARGIN_THRESHOLD

        if is_unknown:
            st.metric("Predicted Target", "UNKNOWN", f"Best guess {pred_class} at {confidence*100:.1f}% (rejected)")
            st.warning("⚠️ **UNRECOGNIZED OBJECT:** Not one of the 10 trained classes. Treat as an unclassified obstacle (caution).")
        else:
            status_text, alert_type = ADAS_HAZARD_MAP[pred_class]
            st.metric("Predicted Target", pred_class.upper(), f"{confidence*100:.2f}% Confidence")
            if alert_type == 'error':
                st.error(f"🚨 **RED ALERT:** {status_text}")
            elif alert_type == 'warning':
                st.warning(f"⚠️ **YELLOW ALERT:** {status_text}")
            else:
                st.info(f"ℹ️ **NOTICE:** {status_text}")

        st.write("**Top 3 probabilities**")
        for p, i in zip(top_probs, top_idx):
            st.write(f"{CIFAR10_CLASSES[i.item()]}: {p.item()*100:.1f}%")
