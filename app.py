
import streamlit as st
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import json

# -----------------------------
# Page Settings
# -----------------------------
st.set_page_config(
    page_title="Animal Classifier",
    page_icon="🐾",
    layout="centered"
)

st.title("🐾 Animal Image Classifier")
st.write("Upload an animal image and the AI model will predict the animal.")

# -----------------------------
# Load Class Names
# -----------------------------
with open("class_names.json", "r") as f:
    class_names = json.load(f)

# -----------------------------
# Load Model
# -----------------------------
device = torch.device("cpu")

model = models.resnet18(weights=None)

model.fc = nn.Linear(
    model.fc.in_features,
    len(class_names)
)

model.load_state_dict(
    torch.load(
        "animal_model.pth",
        map_location=device
    )
)

model = model.to(device)
model.eval()

# -----------------------------
# Image Transformation
# -----------------------------
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

# -----------------------------
# Upload Image
# -----------------------------
uploaded_file = st.file_uploader(
    "Upload an animal image",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:

    image = Image.open(uploaded_file).convert("RGB")

    st.image(
        image,
        caption="Uploaded Image",
        use_container_width=True
    )

    # Prepare image
    image_tensor = transform(image).unsqueeze(0)

    # Prediction
    with torch.no_grad():
        outputs = model(image_tensor)

        probabilities = torch.softmax(outputs, dim=1)

        confidence, predicted = torch.max(
            probabilities, 1
        )

    predicted_class = class_names[predicted.item()]
    confidence_value = confidence.item() * 100

    # -----------------------------
    # Result
    # -----------------------------
    st.success(
        f"Predicted Animal: {predicted_class}"
    )

    st.write(
        f"Confidence: {confidence_value:.2f}%"
    )
