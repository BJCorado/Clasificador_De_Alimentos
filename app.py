import streamlit as st
import torch
import torch.nn as nn
from torchvision import transforms
from torchvision.models import efficientnet_b0
from PIL import Image
import json

# =========================
# CARGAR CLASES (ORDEN CORRECTO)
# =========================
import json

with open("models/classes.json") as f:
    CLASSES = json.load(f)

# =========================
# INFO
# =========================
FOOD_MAP = {
    'pizza': ('Ultraprocesado', '🔴', 'Evitar exceso'),
    'hamburger': ('Ultraprocesado', '🔴', 'Alto en grasas'),
    'ceviche': ('Fresco', '🟢', 'Buena opción'),
    'tacos': ('Mixto', '🟡', 'Depende de ingredientes'),
    'ice_cream': ('Azucarado', '🔴', 'Consumo ocasional'),
}

# =========================
# CARGAR MODELO (BIEN HECHO)
# =========================
@st.cache_resource
def load_model():
    model = efficientnet_b0(weights=None)
    model.classifier[1] = nn.Linear(1280, len(CLASSES))

    model.load_state_dict(
        torch.load("models/modelo_efficientnet.pth", map_location="cpu")
    )

    model.eval()
    return model

model = load_model()

# =========================
# TRANSFORM CORRECTO (CRÍTICO)
# =========================
transform = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
])

# =========================
# INTERFAZ
# =========================
st.title("📸 Clasificador de Alimentos con IA")
st.write("Herramienta educativa. Consulte a un nutricionista.")

modo = st.radio("Selecciona modo:", ["📷 Cámara", "🖼️ Subir imagen"])

# =========================
# FUNCIÓN DE PREDICCIÓN
# =========================
def predict(image):
    img_tensor = transform(image).unsqueeze(0)

    with torch.no_grad():
        outputs = model(img_tensor)
        probs = torch.softmax(outputs, dim=1)

    top_probs, top_classes = torch.topk(probs, 3)

    resultados = []
    for i in range(3):
        clase = CLASSES[top_classes[0][i].item()]
        confianza = float(top_probs[0][i])

        categoria, color, consejo = FOOD_MAP.get(
            clase, ("Desconocido", "⚪", "")
        )

        resultados.append((clase, confianza, categoria, color, consejo))

    return resultados

# =========================
# MODO CÁMARA
# =========================
if modo == "📷 Cámara":

    img_file = st.camera_input("Toma una foto")

    if img_file:
        image = Image.open(img_file).convert("RGB")
        st.image(image, caption="Captura")

        resultados = predict(image)

        st.subheader("🔍 Predicciones:")

        for i, (clase, confianza, categoria, color, consejo) in enumerate(resultados):
            st.write(f"**{i+1}. {clase}** ({confianza:.2f})")
            st.progress(confianza)
            st.write(f"{categoria} {color} - {consejo}")
            st.write("---")

# =========================
# MODO SUBIR IMAGEN
# =========================
else:

    img_file = st.file_uploader("Sube una imagen", type=["jpg", "png"])

    if img_file:
        image = Image.open(img_file).convert("RGB")
        st.image(image, caption="Imagen cargada")

        resultados = predict(image)

        st.subheader("🔍 Predicciones:")

        for i, (clase, confianza, categoria, color, consejo) in enumerate(resultados):
            st.write(f"**{i+1}. {clase}** ({confianza:.2f})")
            st.progress(confianza)
            st.write(f"{categoria} {color} - {consejo}")
            st.write("---")