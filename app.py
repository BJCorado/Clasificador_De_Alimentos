import streamlit as st
import torch
import torch.nn as nn
from torchvision import transforms
from torchvision.models import efficientnet_b0
from PIL import Image
import json

from streamlit_option_menu import option_menu


# =========================
# CONFIG
# =========================
st.set_page_config(
    page_title="Food Vision AI",
    page_icon="🥗",
    layout="centered"
)


# =========================
# CSS
# =========================
st.markdown("""
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>

<link href="https://fonts.googleapis.com/css2?
family=Material+Symbols+Rounded" rel="stylesheet">

<style>

.material-symbols-rounded{
    font-size:28px;
    vertical-align:middle;
}

.metric-box{
    background:#111827;
    padding:18px;
    border-radius:18px;
    margin-bottom:12px;
}

.metric-title{
    font-size:14px;
    color:#9ca3af;
}

.metric-value{
    font-size:28px;
    font-weight:700;
}

</style>
""", unsafe_allow_html=True)


# =========================
# CARGAR CLASES
# =========================
with open("models/classes.json") as f:
    CLASSES = json.load(f)


# =========================
# CARGAR INFO
# =========================
with open(
    "models/food_info.json",
    "r",
    encoding="utf-8"
) as f:

    FOOD_INFO = json.load(f)

# =========================
# MODELO
# =========================

@st.cache_resource
def load_model():

    model = efficientnet_b0(weights=None)

    model.classifier[1] = nn.Linear(
        1280,
        len(CLASSES)
    )

    model.load_state_dict(
        torch.load(
            "models/modelo_efficientnet.pth",
            map_location="cpu"
        )
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
    transforms.Normalize(
        [0.485,0.456,0.406],
        [0.229,0.224,0.225]
    )
])


# =========================
# HEADER
# =========================
st.markdown("""
<p class='title-app'>
Food Vision AI
</p>

<p class='subtitle-app'>
Clasificación nutricional mediante inteligencia artificial
</p>
""", unsafe_allow_html=True)


st.info(
    "Herramienta educativa. Consulte a un nutricionista."
)


# =========================
# MENU
# =========================
modo = option_menu(
    None,
    ["Cámara", "Galería"],
    icons=["camera-fill", "image-fill"],
    orientation="horizontal"
)


# =========================
# PREDICCIÓN
# =========================
def predict(image):

    img_tensor = transform(
        image
    ).unsqueeze(0)

    with torch.no_grad():

        outputs = model(
            img_tensor
        )

        probs = torch.softmax(
            outputs,
            dim=1
        )

    top_probs, top_classes = torch.topk(
        probs,
        3
    )

    top1_conf = float(
        top_probs[0][0]
    )

    top1_class = CLASSES[
        top_classes[0][0].item()
    ]


    if top1_conf < 0.70:

        return {

            "main": {

                "clase":"Desconocido",

                "confianza":top1_conf,

                "info":{

                    "categoria":"No identificado",

                    "color":"#888",

                    "calorias":"-",

                    "proteina":"-",

                    "grasas":"-",

                    "azucar":"-",

                    "consejo":"El modelo no está seguro."
                }
            },

            "others":[]
        }


    main = {

        "clase":top1_class,

        "confianza":top1_conf,

        "info":FOOD_INFO.get(
            top1_class,
            {}
        )
    }


    others = []

    for i in range(1,3):

        conf = float(
            top_probs[0][i]
        )

        if conf >= 0.30:

            clase = CLASSES[
                top_classes[0][i].item()
            ]

            others.append({

                "clase":clase,

                "confianza":conf
            })


    return {

        "main":main,

        "others":others
    }


# =========================
# UI RESULTADOS
# =========================
def mostrar_resultados(resultados):

    top1 = resultados["main"]

    info = top1["info"]

    color = info.get(
        "color",
        "#999"
    )

    categoria = info.get(
        "categoria",
        "Desconocido"
    )


    st.markdown(
        "<div class='main-card'>",
        unsafe_allow_html=True
    )


    st.markdown(f"""
    <h2>
    {top1['clase'].title()}
    </h2>

    <p style='color:#666'>
    Confianza:
    {top1['confianza']*100:.1f}%
    </p>
    """, unsafe_allow_html=True)


    st.progress(
        top1["confianza"]
    )

    st.markdown(f"""
<div style="
display:inline-block;
background:{color};
padding:10px 22px;
border-radius:999px;
color:white;
font-weight:700;
font-size:15px;
margin-top:15px;
margin-bottom:20px;
">
    {categoria}
</div>
""", unsafe_allow_html=True)


    c1, c2 = st.columns(2)


    with c1:

        st.metric(
            "Calorías",
            info.get(
                "calorias",
                "-"
            )
        )

        st.metric(
            "Proteína",
            info.get(
                "proteina",
                "-"
            )
        )


    with c2:

        st.metric(
            "Grasas",
            info.get(
                "grasas",
                "-"
            )
        )

        st.metric(
            "Azúcar",
            info.get(
                "azucar",
                "-"
            )
        )


    st.info(
        info.get(
            "consejo",
            ""
        )
    )


    st.markdown(
        "</div>",
        unsafe_allow_html=True
    )


    if resultados["others"]:

        st.subheader(
            "Otras posibilidades"
        )

        for r in resultados["others"]:

            st.write(
                f"{r['clase']} "
                f"({r['confianza']*100:.1f}%)"
            )

            st.progress(
                r["confianza"]
            )


# =========================
# CÁMARA
# =========================
if modo == "Cámara":

    img_file = st.camera_input(
        "Tomar fotografía"
    )

    if img_file:

        image = Image.open(
            img_file
        ).convert("RGB")

        resultados = predict(
            image
        )

        mostrar_resultados(
            resultados
        )


# =========================
# GALERÍA
# =========================
else:

    img_file = st.file_uploader(
        "Selecciona imagen",
        type=["jpg","png","jpeg"]
    )

    if img_file:

        image = Image.open(
            img_file
        ).convert("RGB")

        st.image(
            image,
            use_container_width=True
        )

        resultados = predict(
            image
        )

        mostrar_resultados(
            resultados
        )