import streamlit as st
import torch
import torchvision.transforms as transforms
from torchvision.models import resnet18
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
@st.cache_resource
def load_model():
    model = resnet18()
    num_ftrs = model.fc.in_features
    model.fc = torch.nn.Linear(num_ftrs, 10)
    # Load fine-tuned weights
    model.load_state_dict(torch.load("cifar10_resnet18_adas.pth", map_location=torch.device('cpu')))
    model.eval()
    return model

model = load_model()

# 3. Image Preprocessing Transformation
transform = transforms.Compose([
    transforms.Resize((32, 32)),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
])

# 4. Upload & Classification UI
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert('RGB')
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.image(image, caption="Uploaded Image", use_column_width=True)
        
    with col2:
        st.subheader("Inference & Hazard Result")
        
        # Preprocess and Predict
        img_tensor = transform(image).unsqueeze(0)
        with torch.no_grad():
            outputs = model(img_tensor)
            probabilities = torch.nn.functional.softmax(outputs, dim=1)
            confidence, predicted = torch.max(probabilities, 1)
            
        pred_class = CIFAR10_CLASSES[predicted.item()]
        conf_percentage = confidence.item() * 100
        status_text, alert_type = ADAS_HAZARD_MAP[pred_class]
        
        st.metric("Predicted Target", pred_class.upper(), f"{conf_percentage:.2f}% Confidence")
        
        # Display Color-Coded Hazard Status
        if alert_type == 'error':
            st.error(f"🚨 **RED ALERT:** {status_text}")
        elif alert_type == 'warning':
            st.warning(f"⚠️ **YELLOW ALERT:** {status_text}")
        else:
            st.info(f"ℹ️ **NOTICE:** {status_text}")
