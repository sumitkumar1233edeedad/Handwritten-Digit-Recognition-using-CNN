import streamlit as st
import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image

# -------------------------
# CNN Model (MATCH TRAINING)
# -------------------------
from torch import nn
# create a convolutional nerual network
class HandwrittenModel(nn.Module):
    def __init__(self, input_shape: int, hidden_unit: int, output_shape: int):
        super().__init__()
        self.conv_block_1  = nn.Sequential(
            nn.Conv2d(
                in_channels=input_shape,
                out_channels=hidden_unit,
                kernel_size=3,
                stride=1,
                padding=1
            ), # values we can set ourselves is our nn's are called hyperparmeters
            nn.ReLU(),
            nn.Conv2d(
                in_channels=hidden_unit,
                out_channels=hidden_unit,
                kernel_size=3,
                stride=1,
                padding=1
            ),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2)
        )

        self.conv_block_2 = nn.Sequential(
            nn.Conv2d(
                in_channels=hidden_unit,
                out_channels=hidden_unit,
                kernel_size=3,
                stride=1,
                padding=1
            ),
            nn.ReLU(),
            nn.Conv2d(
                in_channels=hidden_unit,
                out_channels=hidden_unit,
                kernel_size=3,
                stride=1,
                padding=1
            ),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2)
        )
        

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_features=hidden_unit*7*7, # there is trick to calculating this...
                      out_features=output_shape)
        )
    def forward(self, x):
        x = self.conv_block_1(x)
      
        x = self.conv_block_2(x)
 
       
        x = self.classifier(x)
        return x

# -------------------------
# Load Model
# -------------------------
model = HandwrittenModel(1, 10, 10)
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
    st.balloons()
    st.balloons()
