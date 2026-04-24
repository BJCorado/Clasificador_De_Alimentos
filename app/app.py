import streamlit as st
import torch
from torchvision import transforms
from PIL import Image
from torchvision.models import efficientnet_b0

CLASSES = [
    'apple_pie','banana','salad','fried_egg','grilled_chicken',
    'rice','donut','french_fries','pizza','hamburger',
    'soda','water','juice','cookies','cereal'
]

FOOD_MAP = {
    'apple_pie': ('Procesado', '🟡', 'Consumo ocasional'),
    'banana': ('Fresco', '🟢', 'Consumo diario'),
    'pizza': ('Ultraprocesado', '🔴', 'Evitar exceso'),
    'soda': ('Bebida azucarada', '🔴', 'Alto en azúcar'),
    'water': ('Saludable', '🟢', 'Consumir libremente'),
}

@st.cache_resource
def load_model():
    model = efficientnet_b0(pretrained=False)
    model.classifier[1] = torch.nn.Linear(1280, len(CLASSES))
    model.load_state_dict(torch.load("models/modelo_alimentos.pth", map_location='cpu'))
    return model.eval()

model = load_model()

st.title("Clasificador de Alimentos 🍔")

st.write("Esta clasificación es educativa. Consulte a un nutricionista.")

img_file = st.file_uploader("Sube una imagen", type=["jpg","png"])

if img_file:
    image = Image.open(img_file)
    st.image(image)

    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([.485,.456,.406],[.229,.224,.225])
    ])

    img_tensor = transform(image).unsqueeze(0)

    with torch.no_grad():
        outputs = model(img_tensor)
        probs = torch.softmax(outputs, dim=1)
        top_prob, top_class = torch.max(probs, 1)

    clase = CLASSES[top_class.item()]
    confianza = float(top_prob)

    categoria, color, consejo = FOOD_MAP.get(clase, ("Desconocido","⚪",""))

    st.write(f"Clase: {clase}")
    st.write(f"Categoría: {categoria} {color}")
    st.write(f"Confianza: {confianza:.2f}")
    st.write(f"Consejo: {consejo}")