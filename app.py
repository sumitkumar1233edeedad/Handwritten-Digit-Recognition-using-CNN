import streamlit as st
import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image

# ---------------- PAGE CONFIG ----------------
st.set_page_config(
    page_title="Digit AI",
    page_icon="✍️",
    layout="centered"
)

# ---------------- CUSTOM CSS ----------------
st.markdown("""
<style>

/* Hide Streamlit Branding */
#MainMenu {visibility: hidden;}
footer {visibility: hidden;}
header {visibility: hidden;}

/* Background */
.stApp {
    background: linear-gradient(to right, #141e30, #243b55);
    color: white;
}

/* Title */
.title {
    text-align: center;
    font-size: 42px;
    font-weight: bold;
    margin-bottom: 5px;
}

/* Subtitle */
.subtitle {
    text-align: center;
    font-size: 18px;
    color: #dcdcdc;
    margin-bottom: 30px;
}

/* Card */
.card {
    background-color: #1f2a40;
    padding: 30px;
    border-radius: 15px;
    box-shadow: 0px 4px 20px rgba(0,0,0,0.4);
}

/* Prediction Box */
.prediction {
    text-align: center;
    font-size: 40px;
    font-weight: bold;
    background: linear-gradient(45deg, #00c6ff, #0072ff);
    padding: 20px;
    border-radius: 12px;
    margin-top: 20px;
}

/* Footer */
.footer {
    text-align: center;
    font-size: 14px;
    margin-top: 40px;
    color: #bbbbbb;
}

</style>
""", unsafe_allow_html=True)

# ---------------- MODEL ----------------
class HandwrittenModel(nn.Module):
    def __init__(self, input_shape, hidden_unit, output_shape):
        super().__init__()

        self.conv_block_1 = nn.Sequential(
            nn.Conv2d(input_shape, hidden_unit, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(hidden_unit, hidden_unit, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )

        self.conv_block_2 = nn.Sequential(
            nn.Conv2d(hidden_unit, hidden_unit, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(hidden_unit, hidden_unit, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(hidden_unit * 7 * 7, output_shape)
        )

    def forward(self, x):
        x = self.conv_block_1(x)
        x = self.conv_block_2(x)
        return self.classifier(x)

# ---------------- LOAD MODEL ----------------
model = HandwrittenModel(1, 32, 10)
model.load_state_dict(torch.load("Handwritten.pth", map_location="cpu"))
model.eval()

# ---------------- HEADER ----------------
st.markdown('<div class="title">✍️ Handwritten Digit Recognition</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">Upload a digit image and AI will predict it</div>', unsafe_allow_html=True)

# ---------------- CARD START ----------------
# st.markdown('<div class="card">', unsafe_allow_html=True)

uploaded_file = st.file_uploader("📤 Upload Digit Image", type=["png", "jpg", "jpeg"])

transform = transforms.Compose([
    transforms.Grayscale(),
    transforms.Resize((28, 28)),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

if uploaded_file:

    image = Image.open(uploaded_file)

    st.image(image, caption="Uploaded Image", width=200)

    img = transform(image).unsqueeze(0)

    with torch.no_grad():
        output = model(img)
        prediction = torch.argmax(output, dim=1).item()

    st.markdown(
        f'<div class="prediction">Prediction : {prediction}</div>',
        unsafe_allow_html=True
    )

    st.balloons()

st.markdown('</div>', unsafe_allow_html=True)

# ---------------- FOOTER ----------------
st.markdown("""
<div class="footer">
Developed ❤️ by Vanshu <br>
<a href="https://github.com/sumitkumar1233edeedad" target="_blank">
GitHub Profile
</a>
</div>
""", unsafe_allow_html=True)
