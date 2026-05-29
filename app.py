import streamlit as st
import torch
import torch.nn as nn
import av
import cv2
import json
import time
import numpy as np

from PIL import Image
from torchvision import transforms
from torchvision.models import efficientnet_b0

from streamlit_option_menu import option_menu
from streamlit_webrtc import (
    webrtc_streamer,
    VideoProcessorBase,
    RTCConfiguration
)


# =========================
# CONFIG
# =========================
st.set_page_config(
    page_title="Clasificador De Alimentos y Nutrición Orientativa",
    page_icon="🥗",
    layout="wide"
)

if "camera_id" not in st.session_state:

    st.session_state.camera_id = 0

# =========================
# CSS para mejorar apariencia
# =========================
st.markdown("""
<style>

/* padding general */
.block-container{
    padding-top:1.8rem !important;
    padding-bottom:0rem !important;
}

/* video */
video{
    max-height:340px !important;
    width:100% !important;
    border-radius:12px;
    object-fit:cover;
}

/* espacio entre elementos */
[data-testid="stVerticalBlock"]{
    gap:0.2rem !important;
}

/* metrics más compactas */
div[data-testid="stMetric"]{
    padding:4px !important;
    margin:0px !important;
    border-radius:10px;
}

/* textos de metric */
div[data-testid="stMetricLabel"]{
    font-size:0.8rem !important;
}

div[data-testid="stMetricValue"]{
    font-size:1rem !important;
}

/* botón */
.stButton button{
    padding:0.4rem !important;
}

/* success box */
.stAlert{
    padding:0.4rem !important;
}

/* expander */
.streamlit-expanderHeader{
    font-size:0.9rem !important;
}

</style>
""", unsafe_allow_html=True)

# =========================
# DATA
# =========================
with open("models/classes.json", encoding="utf-8") as f:
    CLASSES = json.load(f)

with open(
    "models/food_info.json",
    encoding="utf-8"
) as f:
    FOOD_INFO = json.load(f)

# =========================
# LABELS EN ESPAÑOL
# =========================
SPANISH_NAMES = {

    "apple": "Manzana",
    "banana": "Banana",
    "ceviche": "Ceviche",
    "chicken_wings": "Alitas de pollo",
    "coca_cola": "Coca Cola",
    "coffee": "Café",
    "french_fries": "Papas fritas",
    "fried_rice": "Arroz frito",
    "hamburger": "Hamburguesa",
    "ice_cream": "Helado",
    "lemon": "Limón",
    "mango": "Mango",
    "monster_energy": "Monster Energy",
    "nachos": "Nachos",
    "pizza": "Pizza",
    "ramen": "Ramen",
    "spaghetti_bolognese": "Espagueti boloñesa",
    "steak": "Bistec",
    "tacos": "Tacos",
    "water_bottle": "Botella de agua",
    "watermelon": "Sandía"
}

# =========================
# MODEL
# =========================
@st.cache_resource
def load_model():

    model = efficientnet_b0(
        weights=None
    )

    model.classifier[1] = nn.Linear(
        1280,
        len(CLASSES)
    )

    model.load_state_dict(
        torch.load(
            "models/modelo_efficientnet.pth",
            map_location="cpu",
            weights_only=True
        )
    )

    model.eval()

    return model


model = load_model()


# =========================
# TRANSFORM
# =========================
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        [0.485, 0.456, 0.406],
        [0.229, 0.224, 0.225]
    )
])


# =========================
# PREDICT
# =========================
def predict(image):

    tensor = transform(
        image
    ).unsqueeze(0)

    with torch.no_grad():

        output = model(
            tensor
        )

        probs = torch.softmax(
            output,
            dim=1
        )

    conf, pred = torch.max(
        probs,
        1
    )

    conf = float(
        conf[0]
    )

    label = CLASSES[
        pred[0].item()
    ]

    info = FOOD_INFO.get(
        label,
        {}
    )

    return label, conf, info


# =========================
# OBJECT ROI
# =========================
def get_food_roi(frame):

    frame_h, frame_w = frame.shape[:2]

    hsv = cv2.cvtColor(
        frame,
        cv2.COLOR_BGR2HSV
    )


    # detectar zonas con color real
    mask = cv2.inRange(
        hsv,
        (0, 40, 40),
        (180, 255, 255)
    )


    kernel = np.ones(
        (9, 9),
        np.uint8
    )


    mask = cv2.morphologyEx(
        mask,
        cv2.MORPH_CLOSE,
        kernel
    )


    contours, _ = cv2.findContours(
        mask,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )


    if contours:

        cx_frame = frame_w // 2
        cy_frame = frame_h // 2


        # prioridad:
        # grande + cerca del centro
        def score(contour):

            x, y, w, h = cv2.boundingRect(
                contour
            )

            cx = x + w // 2
            cy = y + h // 2

            dist = abs(
                cx - cx_frame
            ) + abs(
                cy - cy_frame
            )

            area = cv2.contourArea(
                contour
            )

            return area - dist * 5


        contours = sorted(
            contours,
            key=score,
            reverse=True
        )


        for contour in contours:

            area = cv2.contourArea(
                contour
            )

            frame_area = (
                frame_h * frame_w
            )


            # muy pequeño
            if area < 4000:
                continue


            # muy grande
            if area > frame_area * 0.35:
                continue


            x, y, w, h = cv2.boundingRect(
                contour
            )


            # ====================
            # NUEVO: ignorar bordes
            # ====================
            border = 40

            if (
                x < border
                or y < border
                or x + w > frame_w - border
                or y + h > frame_h - border
            ):
                continue


            # ====================
            # NUEVO: forma lógica
            # ====================
            ratio = w / h

            if (
                ratio < 0.3
                or ratio > 2.0
            ):
                continue


            # ====================
            # NUEVO: cerca centro
            # ====================
            cx = x + w // 2
            cy = y + h // 2

            dist = abs(
                cx - cx_frame
            ) + abs(
                cy - cy_frame
            )

            if dist > 250:
                continue


            # ====================
            # apretar bounding box
            # ====================
            margin = 12

            x = max(
                0,
                x + margin
            )

            y = max(
                0,
                y + margin
            )

            w = max(
                50,
                w - margin * 2
            )

            h = max(
                50,
                h - margin * 2
            )


            roi = frame[
                y:y+h,
                x:x+w
            ]


            # ====================
            # NUEVO: evitar sombras
            # ====================
            mean_color = np.mean(
                roi
            )

            if mean_color < 50:
                continue


            return roi, (
                x,
                y,
                x + w,
                y + h
            )


    # ====================
    # fallback: centro
    # ====================
    size = 250

    x1 = (
        frame_w // 2
    ) - (
        size // 2
    )

    y1 = (
        frame_h // 2
    ) - (
        size // 2
    )

    x2 = x1 + size
    y2 = y1 + size


    roi = frame[
        y1:y2,
        x1:x2
    ]


    return roi, (
        x1,
        y1,
        x2,
        y2
    )
# =========================
# VIDEO PROCESSOR
# =========================
class FoodProcessor(
    VideoProcessorBase
):

    def __init__(self):

        self.frame_counter = 0

        self.freeze = False

        self.detected = None

        self.last_label = None

        self.same_count = 0

        self.frozen_frame = None


    def recv(self, frame):

        img = frame.to_ndarray(
            format="bgr24"
        )


        # =====================
        # FRAME CONGELADO
        # =====================
        if self.freeze:

            frozen = (
                self.frozen_frame
                .copy()
            )

            return av.VideoFrame.from_ndarray(
                frozen,
                format="bgr24"
            )


        self.frame_counter += 1


        # =====================
        # DETECCION
        # =====================
        if self.frame_counter % 8 == 0:

            roi_data = get_food_roi(
                img
            )


            if roi_data:

                roi, box = roi_data

                x1, y1, x2, y2 = box


                cv2.rectangle(
                    img,
                    (x1, y1),
                    (x2, y2),
                    (0, 255, 0),
                    2
                )


                rgb = cv2.cvtColor(
                    roi,
                    cv2.COLOR_BGR2RGB
                )

                pil = Image.fromarray(
                    rgb
                )


                label, conf, info = predict(
                    pil
                )


                if conf >= 0.90:

                    if label == self.last_label:

                        self.same_count += 1

                    else:

                        self.last_label = label

                        self.same_count = 1


                    if self.same_count >= 3:

                        self.freeze = True

                        cv2.putText(
                            img,
                            label,
                            (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.8,
                            (0, 255, 0),
                            2
                        )

                        self.frozen_frame = (
                            img.copy()
                        )

                        self.detected = {

                            "label": label,

                            "conf": conf,

                            "info": info,

                            "box": box
                        }

                else:

                    self.same_count = 0

                    self.last_label = None


        cv2.putText(
            img,
            "ESCANEANDO...",
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 255, 255),
            2
        )


        return av.VideoFrame.from_ndarray(
            img,
            format="bgr24"
        )


# =========================
# UI
# =========================
st.title(
    "🥗 Clasificador De Alimentos y Nutrición Orientativa"
)


modo = option_menu(
    None,
    ["Cámara", "Galería"],
    icons=[
        "camera-fill",
        "image-fill"
    ],
    orientation="horizontal"
)


# =========================
# CAMARA
# =========================
if modo == "Cámara":

    cam_col, info_col = st.columns(
        [1, 1.15]
    )


    # =====================
    # COLUMNA CAMARA
    # =====================
    with cam_col:

        ctx = webrtc_streamer(

            key=f"food-ai-{st.session_state.camera_id}",

            video_processor_factory=(
                FoodProcessor
            ),

            rtc_configuration=RTCConfiguration(
                {
                    "iceServers": [
                        {
                            "urls": [
                                "stun:stun.l.google.com:19302"
                            ]
                        }
                    ]
                }
            ),

            media_stream_constraints={
                "video": True,
                "audio": False
            }
        )


    # =====================
    # COLUMNA INFO
    # =====================
    with info_col:

        result_box = st.empty()


        if st.button(
            "🔄 Nueva detección",
            use_container_width=True
        ):

            st.session_state.camera_id += 1

            st.rerun()


        if ctx.state.playing:

            while True:

                if (
                    ctx.video_processor
                    and
                    ctx.video_processor.detected
                ):

                    data = (
                        ctx.video_processor
                        .detected
                    )

                    info = data["info"]

                    label_es = SPANISH_NAMES.get(
                        data["label"],
                        data["label"]
                    )


                    with result_box.container():

                        st.success(
                            f"{label_es} "
                            f"({data['conf']*100:.1f}%)"
                        )


                        categoria = info.get(
                            "categoria",
                            "-"
                        )

                        color = info.get(
                            "color",
                            "#999"
                        )


                        st.markdown(f"""
                        <div style="
                        background:{color};
                        color:white;
                        padding:10px;
                        border-radius:12px;
                        font-weight:bold;
                        text-align:center;
                        margin-bottom:10px;
                        ">
                        {categoria}
                        </div>
                        """,
                        unsafe_allow_html=True)


                        c1, c2 = st.columns(2)


                        with c1:

                            st.metric(
                                "🔥 Calorías",
                                info.get(
                                    "calorias",
                                    "-"
                                )
                            )

                            st.metric(
                                "💪 Proteína",
                                info.get(
                                    "proteina",
                                    "-"
                                )
                            )


                        with c2:

                            st.metric(
                                "🥑 Grasas",
                                info.get(
                                    "grasas",
                                    "-"
                                )
                            )

                            st.metric(
                                "🍬 Azúcar",
                                info.get(
                                    "azucar",
                                    "-"
                                )
                            )


                        with st.expander(
                            "💡 Consejo nutricional"
                        ):

                            st.write(
                                info.get(
                                    "consejo",
                                    "-"
                                )
                            )
                            st.caption(
    "La información mostrada es aproximada y con fines educativos, NO reemplaza la evaluación de un nutricionista, médico o especialista en salud."
)

                    break


                time.sleep(
                    0.2
                )


# =========================
# GALERIA
# =========================
else:

    img_file = st.file_uploader(
        "Selecciona imagen",
        type=[
            "jpg",
            "png",
            "jpeg"
        ]
    )


    if img_file:

        image = Image.open(
            img_file
        ).convert("RGB")


        label, conf, info = predict(
            image
        )


        label_es = SPANISH_NAMES.get(
            label,
            label
        )


        img_col, info_col = st.columns(
            [1, 1.15]
        )


        # =====================
        # IMAGEN
        # =====================
        with img_col:

            st.image(
                image,
                use_container_width=True
            )


        # =====================
        # INFO
        # =====================
        with info_col:

            st.success(
                f"{label_es} "
                f"({conf*100:.1f}%)"
            )


            categoria = info.get(
                "categoria",
                "-"
            )

            color = info.get(
                "color",
                "#999"
            )


            st.markdown(f"""
            <div style="
            background:{color};
            color:white;
            padding:10px;
            border-radius:12px;
            font-weight:bold;
            text-align:center;
            margin-bottom:10px;
            ">
            {categoria}
            </div>
            """,
            unsafe_allow_html=True)


            c1, c2 = st.columns(2)


            with c1:

                st.metric(
                    "🔥 Calorías",
                    info.get(
                        "calorias",
                        "-"
                    )
                )

                st.metric(
                    "💪 Proteína",
                    info.get(
                        "proteina",
                        "-"
                    )
                )


            with c2:

                st.metric(
                    "🥑 Grasas",
                    info.get(
                        "grasas",
                        "-"
                    )
                )

                st.metric(
                    "🍬 Azúcar",
                    info.get(
                        "azucar",
                        "-"
                    )
                )


            with st.expander(
                "💡 Consejo nutricional"
            ):

                st.write(
                    info.get(
                        "consejo",
                        "-"
                    )
                )
                st.caption(
    "La información mostrada es aproximada y con fines educativos, NO reemplaza la evaluación de un nutricionista, médico o especialista en salud."
)