import streamlit as st
import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image

# -------------------------
# CNN Model (MATCH TRAINING)
# -------------------------
class HandwrittenModel(nn.Module):
    def __init__(self):
        super().__init__()

        self.conv_block_1 = nn.Sequential(
            nn.Conv2d(1, 10, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(10, 10, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )

        self.conv_block_2 = nn.Sequential(
            nn.Conv2d(10, 10, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(10, 10, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(10 * 7 * 7, 10)
        )

    def forward(self, x):
        x = self.conv_block_1(x)
        x = self.conv_block_2(x)
        x = self.classifier(x)
        return x

# -------------------------
# Load Model
# -------------------------
model = HandwrittenModel()
model.load_state_dict(torch.load("Handwritten.pth", map_location="cpu"))
model.eval()

# -------------------------
# Streamlit UI
# -------------------------
st.title("✍️ Handwritten Digit Recognition")

uploaded_file = st.file_uploader("Upload Digit Image", type=["png", "jpg", "jpeg"])

# -------------------------
# Image Transform
# -------------------------
transform = transforms.Compose([
    transforms.Grayscale(),
    transforms.Resize((28, 28)),
    transforms.ToTensor(),
])

if uploaded_file is not None:

    image = Image.open(uploaded_file)

    st.image(image, caption="Uploaded Image", use_column_width=True)

    img = transform(image)
    img = img.unsqueeze(0)

    with torch.no_grad():
        output = model(img)
        prediction = torch.argmax(output, dim=1).item()

    st.success(f"Predicted Digit: {prediction}")
