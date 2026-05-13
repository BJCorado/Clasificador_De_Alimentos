import cv2
import json
import time
import torch
import torch.nn as nn

from PIL import Image
from torchvision import transforms
from torchvision.models import efficientnet_b0


# =================================
# DATA
# =================================
with open("models/classes.json", encoding="utf-8") as f:
    CLASSES = json.load(f)

with open(
    "models/food_info.json",
    encoding="utf-8"
) as f:
    FOOD_INFO = json.load(f)


# =================================
# MODEL
# =================================
print("Cargando modelo...")

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


# =================================
# TRANSFORM
# =================================
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        [0.485, 0.456, 0.406],
        [0.229, 0.224, 0.225]
    )
])


# =================================
# COLORS
# =================================
BG = (15, 18, 30)

WHITE = (255, 255, 255)

GREEN = (0, 220, 120)

YELLOW = (0, 220, 255)

RED = (80, 80, 255)

BLUE = (255, 160, 0)


# =================================
# TRACKER
# =================================
bg_subtractor = cv2.createBackgroundSubtractorMOG2(
    history=300,
    varThreshold=40
)


# =================================
# HELPERS
# =================================
def draw_panel(frame):

    h, w = frame.shape[:2]

    panel_x = w - 320

    overlay = frame.copy()

    cv2.rectangle(
        overlay,
        (panel_x, 0),
        (w, h),
        BG,
        -1
    )

    cv2.addWeighted(
        overlay,
        0.92,
        frame,
        0.08,
        0,
        frame
    )

    return panel_x


def draw_text(
    img,
    text,
    x,
    y,
    scale=0.65,
    color=WHITE,
    thickness=2
):

    cv2.putText(
        img,
        str(text),
        (x, y),
        cv2.FONT_HERSHEY_SIMPLEX,
        scale,
        color,
        thickness
    )


def category_color(cat):

    cat = str(cat).lower()


    if (
        "fresco" in cat
        or
        "natural" in cat
    ):
        return GREEN


    if (
        "procesado" in cat
        or
        "mixto" in cat
    ):
        return YELLOW


    if (
        "ultra" in cat
        or
        "azucar" in cat
    ):
        return RED


    return WHITE


def predict(img):

    pil = Image.fromarray(
        img
    )

    tensor = transform(
        pil
    ).unsqueeze(0)


    with torch.no_grad():

        outputs = model(
            tensor
        )

        probs = torch.softmax(
            outputs,
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


    label = label.lower().strip()


    info = FOOD_INFO.get(
        label,
        {
            "categoria": "Sin datos",
            "calorias": "-",
            "proteina": "-",
            "grasas": "-",
            "azucar": "-",
            "consejo": "-"
        }
    )


    return label, conf, info


# =================================
# DETECT OBJECT
# =================================
def detect_object(frame):

    mask = bg_subtractor.apply(
        frame
    )

    _, mask = cv2.threshold(
        mask,
        200,
        255,
        cv2.THRESH_BINARY
    )


    contours, _ = cv2.findContours(
        mask,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )


    h, w = frame.shape[:2]

    cx_screen = w // 2

    cy_screen = h // 2


    best_score = 999999

    best_box = None


    for cnt in contours:

        area = cv2.contourArea(
            cnt
        )

        if area < 15000:
            continue


        x, y, bw, bh = cv2.boundingRect(
            cnt
        )


        cx = x + bw // 2

        cy = y + bh // 2


        dist = (
            (cx - cx_screen) ** 2
            +
            (cy - cy_screen) ** 2
        )


        if dist < best_score:

            best_score = dist

            best_box = (
                x,
                y,
                bw,
                bh
            )


    return best_box


# =================================
# CAMERA
# =================================
cap = cv2.VideoCapture(0)

if not cap.isOpened():

    print("No se pudo abrir cámara")
    exit()


cap.set(
    cv2.CAP_PROP_FRAME_WIDTH,
    1280
)

cap.set(
    cv2.CAP_PROP_FRAME_HEIGHT,
    720
)


# =================================
# STATE
# =================================
last_prediction = None

freeze = False

tracked_box = None

frame_counter = 0

last_time = time.time()


print("SPACE = continuar")
print("Q = salir")


# =================================
# LOOP
# =================================
while True:


    if not freeze:

        ret, frame = cap.read()

        if not ret:
            break

        frame = cv2.flip(
            frame,
            1
        )


    display = frame.copy()


    h, w = display.shape[:2]


    panel_x = draw_panel(
        display
    )


    # FPS
    now = time.time()

    delta = max(
        now - last_time,
        0.0001
    )

    fps = int(
        1 / delta
    )

    last_time = now


    # TRACKING
    if not freeze:

        box = detect_object(
            display
        )

        if box is not None:
            tracked_box = box


    # BOX
    if tracked_box is not None:

        x, y, bw, bh = tracked_box


        cv2.rectangle(
            display,
            (x, y),
            (x + bw, y + bh),
            BLUE,
            3
        )


        if not freeze:

            frame_counter += 1


            if frame_counter % 8 == 0:

                roi = display[
                    y:y+bh,
                    x:x+bw
                ]


                if roi.size > 0:

                    label, conf, info = predict(

                        cv2.cvtColor(
                            roi,
                            cv2.COLOR_BGR2RGB
                        )
                    )


                    if conf > 0.75:

                        freeze = True


                        last_prediction = {

                            "label": label,

                            "conf": conf,

                            "info": info
                        }


    # =================================
    # UI
    # =================================
    draw_text(
        display,
        "FOOD VISION AI",
        panel_x + 20,
        50,
        0.9
    )


    draw_text(
        display,
        f"FPS: {fps}",
        panel_x + 20,
        90
    )


    if freeze:

        draw_text(
            display,
            "DETECTADO",
            panel_x + 20,
            130,
            color=GREEN
        )

    else:

        draw_text(
            display,
            "ESCANEANDO...",
            panel_x + 20,
            130,
            color=YELLOW
        )


    # =================================
    # RESULTADOS
    # =================================
    if last_prediction:

        info = last_prediction[
            "info"
        ]


        cat = info.get(
            "categoria",
            "?"
        )


        c = category_color(
            cat
        )


        draw_text(
            display,

            last_prediction[
                "label"
            ].replace(
                "_",
                " "
            ).title(),

            panel_x + 20,
            230
        )


        draw_text(
            display,

            f"{last_prediction['conf']*100:.1f}%",

            panel_x + 20,
            270
        )


        draw_text(
            display,

            f"Categoria: {cat}",

            panel_x + 20,
            320,

            color=c
        )


        draw_text(
            display,

            f"Kcal: {info['calorias']}",

            panel_x + 20,
            380
        )


        draw_text(
            display,

            f"Prot: {info['proteina']}",

            panel_x + 20,
            420
        )


        draw_text(
            display,

            f"Grasas: {info['grasas']}",

            panel_x + 20,
            460
        )


        draw_text(
            display,

            f"Azucar: {info['azucar']}",

            panel_x + 20,
            500
        )


        consejo = info[
            "consejo"
        ][:38]


        draw_text(
            display,

            consejo,

            panel_x + 20,
            560,

            scale=0.45
        )


    cv2.imshow(
        "Food Vision AI",
        display
    )


    key = cv2.waitKey(1) & 0xFF


    if key == ord("q"):
        break


    if key == 32:

        freeze = False

        last_prediction = None


cap.release()

cv2.destroyAllWindows()