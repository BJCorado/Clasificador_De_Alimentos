import streamlit as st
import torch
import torch.nn as nn
from torchvision import transforms
from torchvision.models import efficientnet_b0
from PIL import Image, ImageOps
import json

# =========================
# CARGAR CLASES
# =========================
with open("models/classes.json") as f:
    CLASSES = json.load(f)

# =========================
# CARGAR INFO NUTRICIONAL
# =========================
with open("models/food_info.json") as f:
    FOOD_INFO = json.load(f)

# =========================
# MODELO
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
# TRANSFORM
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
# PREDICCIÓN
# =========================
def predict(image):
    img_tensor = transform(image).unsqueeze(0)

    with torch.no_grad():
        outputs = model(img_tensor)
        probs = torch.softmax(outputs, dim=1)

    top_probs, top_classes = torch.topk(probs, 3)

    resultados = []

    top1_conf = float(top_probs[0][0])
    top1_class = CLASSES[top_classes[0][0].item()]

    # DESCONOCIDO
    if top1_conf < 0.70:
        return {
            "main": {
                "clase": "Desconocido",
                "confianza": top1_conf,
                "info": {
                    "categoria": "No identificado",
                    "color": "#888",
                    "calorias": "-",
                    "proteina": "-",
                    "grasas": "-",
                    "azucar": "-",
                    "consejo": "El modelo no está seguro."
                }
            },
            "others": []
        }

    # PRINCIPAL
    main = {
        "clase": top1_class,
        "confianza": top1_conf,
        "info": FOOD_INFO.get(top1_class, {})
    }

    # OTRAS (solo si ≥ 0.30)
    others = []
    for i in range(1, 3):
        conf = float(top_probs[0][i])

        if conf >= 0.30:
            clase = CLASSES[top_classes[0][i].item()]
            others.append({
                "clase": clase,
                "confianza": conf
            })

    return {
        "main": main,
        "others": others
    }
# =========================
# MOSTRAR RESULTADO
# =========================
def mostrar_resultados(resultados):
    top1 = resultados["main"]
    info = top1["info"]

    st.subheader("Resultado principal")

    st.write(f"**{top1['clase']}** ({top1['confianza']:.2f})")
    st.progress(top1["confianza"])

    color = info.get("color", "#999")
    categoria = info.get("categoria", "Desconocido")

    st.markdown(f"""
    <div style="
        background-color:{color};
        padding:10px;
        border-radius:10px;
        color:white;
        font-weight:bold;
        width:fit-content;
    ">
    {categoria}
    </div>
    """, unsafe_allow_html=True)

    st.write(f"🔥 Calorías: {info.get('calorias','-')}")
    st.write(f"🥩 Proteína: {info.get('proteina','-')}")
    st.write(f"🧈 Grasas: {info.get('grasas','-')}")
    st.write(f"🍬 Azúcar: {info.get('azucar','-')}")
    st.write(f"💡 {info.get('consejo','')}")

    # SOLO SI HAY OTRAS
    if resultados["others"]:
        st.subheader("🤔 Otras posibilidades")

        for r in resultados["others"]:
            st.write(f"{r['clase']} ({r['confianza']:.2f})")
# =========================
# MODO CÁMARA
# =========================
if modo == "📷 Cámara":

    img_file = st.camera_input("Toma una foto")

    if img_file:
        image = Image.open(img_file).convert("RGB")
        resultados = predict(image)
        mostrar_resultados(resultados)

# =========================
# MODO SUBIR IMAGEN
# =========================
else:

    img_file = st.file_uploader("Sube una imagen", type=["jpg", "png"])

    if img_file:
        image = Image.open(img_file).convert("RGB")

        st.image(image, caption="Imagen cargada")

        resultados = predict(image)
        mostrar_resultados(resultados)