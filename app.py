import os

import joblib
import pandas as pd
import streamlit as st
import torch
from torch import nn

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

MODEL_PATH = os.path.join(BASE_DIR, "movie_ann_model.pt")
SCALER_PATH = os.path.join(BASE_DIR, "scaler.pkl")

FEATURE_NAMES = [
    "Duration_Min",
    "Violence_Level",
    "Comedy_Scenes",
    "Budget_Million",
    "Star_Power",
]

LABEL_NAMES = ["Action", "Comedy", "Romance"]

# LabelEncoder uses alphabetical order:
# High = 0, Low = 1, Medium = 2
LEVEL_TO_NUMBER = {
    "High": 0,
    "Low": 1,
    "Medium": 2,
}

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

if not os.path.exists(MODEL_PATH):
    st.error("movie_ann_model.pt was not found. Run the training notebook first.")
    st.stop()

if not os.path.exists(SCALER_PATH):
    st.error("scaler.pkl was not found. Run the training notebook first.")
    st.stop()

# This structure must exactly match the model used during training
model = nn.Sequential(
    nn.Linear(5, 16),
    nn.ReLU(),

    nn.Linear(16, 12),
    nn.ReLU(),

    nn.Linear(12, 8),
    nn.ReLU(),

    nn.Linear(8, 4),
    nn.ReLU(),

    nn.Linear(4, 3)
).to(DEVICE)

model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
model.eval()

scaler = joblib.load(SCALER_PATH)

st.title("Movie Genre Prediction using PyTorch ANN")
st.write("Multi-label prediction: Action, Comedy, and Romance")

duration = st.slider("Movie Duration (minutes)", 80, 200, 120)

violence_name = st.selectbox("Violence Level", ["Low", "Medium", "High"])
comedy_name = st.selectbox("Comedy Scene Level", ["Low", "Medium", "High"])

budget = st.slider("Budget (Million $)", 20, 150, 60)
star_power = st.slider("Star Power (1–5)", 1, 5, 3)

if st.button("Predict Genre"):

    violence = LEVEL_TO_NUMBER[violence_name]
    comedy = LEVEL_TO_NUMBER[comedy_name]

    movie = pd.DataFrame(
        [[duration, violence, comedy, budget, star_power]],
        columns=FEATURE_NAMES
    )

    movie_scaled = scaler.transform(movie)

    movie_tensor = torch.tensor(
        movie_scaled,
        dtype=torch.float32,
        device=DEVICE
    )

    with torch.inference_mode():
        logits = model(movie_tensor)
        probabilities = torch.sigmoid(logits).cpu().numpy()[0]

    st.subheader("Prediction Result")

    for label, probability in zip(LABEL_NAMES, probabilities):
        predicted = "Yes" if probability >= 0.5 else "No"
        st.write(f"{label}: {predicted} ({probability:.1%})")