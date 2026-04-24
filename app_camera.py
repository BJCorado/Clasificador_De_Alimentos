import cv2
import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image

# 🔹 TUS CLASES (10)
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

# 🔹 MODELO (igual que el entrenamiento)
class SimpleCNN(nn.Module):
    def __init__(self, num_classes):
        super(SimpleCNN, self).__init__()
        
        self.conv = nn.Sequential(
            nn.Conv2d(3, 16, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(16, 32, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(32 * 32 * 32, 128),
            nn.ReLU(),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        x = self.conv(x)
        x = self.fc(x)
        return x


# 🔹 CARGAR MODELO
model = SimpleCNN(len(CLASSES))
model.load_state_dict(torch.load("modelo_food.pth"))
model.eval()

# 🔹 TRANSFORMACIONES (igual que entrenamiento)
transform = transforms.Compose([
    transforms.Resize((128,128)),
    transforms.ToTensor(),
    transforms.Normalize([0.5,0.5,0.5],[0.5,0.5,0.5])
])

# 🔹 INICIAR CÁMARA
cap = cv2.VideoCapture(0)

print("Presiona Q para salir")

while True:
    ret, frame = cap.read()
    if not ret:
        print("Error al acceder a la cámara")
        break

    # Convertir a RGB
    img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    pil_img = Image.fromarray(img)

    # Preprocesar
    input_tensor = transform(pil_img).unsqueeze(0)

    # Predicción
    with torch.no_grad():
        output = model(input_tensor)
        probs = torch.softmax(output, dim=1)
        confidence, pred = torch.max(probs, 1)

    label = CLASSES[pred.item()]
    conf = confidence.item() * 100

    # Mostrar texto (con porcentaje 🔥)
    texto = f"{label} ({conf:.1f}%)"

    cv2.putText(frame, texto, (10, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                1, (0,255,0), 2)

    cv2.imshow("Food Classifier", frame)

    # Salir con Q
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()