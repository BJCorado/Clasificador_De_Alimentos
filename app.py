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
    page_title="Food Vision AI",
    page_icon="🥗",
    layout="wide"
)


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

    hsv = cv2.cvtColor(
        frame,
        cv2.COLOR_BGR2HSV
    )


    # zonas con color real
    mask = cv2.inRange(
        hsv,
        (0, 40, 40),
        (180, 255, 255)
    )


    kernel = np.ones(
        (5, 5),
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

        contours = sorted(
            contours,
            key=cv2.contourArea,
            reverse=True
        )


        for contour in contours:

            area = cv2.contourArea(
                contour
            )


            if area < 2000:
                continue


            x, y, w, h = cv2.boundingRect(
                contour
            )


            roi = frame[
                y:y+h,
                x:x+w
            ]


            return roi, (
                x,
                y,
                x+w,
                y+h
            )


    # fallback: centro de cámara
    h, w = frame.shape[:2]

    size = 250

    x1 = (w // 2) - (size // 2)
    y1 = (h // 2) - (size // 2)

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
    "🥗 Food Vision AI"
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

    ctx = webrtc_streamer(

        key="food-ai",

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


    result_box = st.empty()


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


                with result_box.container():

                    st.success(
                        f"{data['label']} "
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
                    margin-bottom:20px;
                    ">
                    {categoria}
                    </div>
                    """,
                    unsafe_allow_html=True)


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
                            "-"
                        )
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


        st.image(
            image,
            use_container_width=True
        )


        label, conf, info = predict(
            image
        )


        st.success(
            f"{label} "
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
        margin-bottom:20px;
        ">
        {categoria}
        </div>
        """,
        unsafe_allow_html=True)


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
                "-"
            )
        )