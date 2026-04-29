import streamlit as st
import torch
import torch.nn as nn
from torchvision import transforms, models
from PIL import Image

# 🔹 CLASES
CLASSES = [
    "pizza",
    "hamburger",
    "ceviche",
    "tacos",
    "steak",
    "ramen",
    "ice_cream",
    "spaghetti_bolognese",
    "fried_rice",
    "chicken_wings"
]

# 🔹 INFO
FOOD_MAP = {
    'pizza': ('Ultraprocesado', '🔴', 'Evitar exceso'),
    'hamburger': ('Ultraprocesado', '🔴', 'Alto en grasas'),
    'ceviche': ('Fresco', '🟢', 'Buena opción'),
    'tacos': ('Mixto', '🟡', 'Depende de ingredientes'),
    'ice_cream': ('Azucarado', '🔴', 'Consumo ocasional'),
}

# 🔹 CARGAR MODELO (EfficientNet)
@st.cache_resource
def load_model():
    model = models.efficientnet_b0(pretrained=False)
    
    # 🔥 ajustar última capa a 10 clases
    model.classifier[1] = nn.Linear(model.classifier[1].in_features, len(CLASSES))
    
    model.load_state_dict(torch.load("models/modelo_efficientnet.pth", map_location="cpu"))
    model.eval()
    return model

model = load_model()

# 🔹 TRANSFORM (IMPORTANTE para EfficientNet)
transform = transforms.Compose([
    transforms.Resize((224, 224)),  # 🔥 EfficientNet usa 224
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],   # 🔥 estándar ImageNet
        std=[0.229, 0.224, 0.225]
    )
])

# 🔹 INTERFAZ
st.title("📸 Clasificador de Alimentos con IA")
st.write("Usa tu cámara o sube una imagen")

modo = st.radio("Selecciona modo:", ["📷 Cámara", "🖼️ Subir imagen"])

# =========================
# 📷 MODO CÁMARA
# =========================
if modo == "📷 Cámara":

    img_file = st.camera_input("Toma una foto")

    if img_file:
        image = Image.open(img_file).convert("RGB")
        st.image(image, caption="Captura")

        img_tensor = transform(image).unsqueeze(0)

        with torch.no_grad():
            outputs = model(img_tensor)
            probs = torch.softmax(outputs, dim=1)

        top_probs, top_classes = torch.topk(probs, 3)

        st.subheader("🔍 Predicciones:")

        for i in range(3):
            clase = CLASSES[top_classes[0][i]]
            confianza = float(top_probs[0][i])

            categoria, color, consejo = FOOD_MAP.get(clase, ("Desconocido", "⚪", ""))

            st.write(f"**{i+1}. {clase}** ({confianza:.2f})")
            st.progress(confianza)
            st.write(f"{categoria} {color} - {consejo}")
            st.write("---")

# =========================
# 🖼️ MODO SUBIR IMAGEN
# =========================
else:

    img_file = st.file_uploader("Sube una imagen", type=["jpg", "png"])

    if img_file:
        image = Image.open(img_file).convert("RGB")
        st.image(image, caption="Imagen cargada")

        img_tensor = transform(image).unsqueeze(0)

        with torch.no_grad():
            outputs = model(img_tensor)
            probs = torch.softmax(outputs, dim=1)

        top_probs, top_classes = torch.topk(probs, 3)

        st.subheader("🔍 Predicciones:")

        for i in range(3):
            clase = CLASSES[top_classes[0][i]]
            confianza = float(top_probs[0][i])

            categoria, color, consejo = FOOD_MAP.get(clase, ("Desconocido", "⚪", ""))

            st.write(f"**{i+1}. {clase}** ({confianza:.2f})")
            st.progress(confianza)
            st.write(f"{categoria} {color} - {consejo}")
            st.write("---")