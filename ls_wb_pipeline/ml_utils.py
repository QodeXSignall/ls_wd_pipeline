import torch
import torchvision.transforms as T
from PIL import Image
import cv2
import os
from tqdm import tqdm

# Модель
model = torch.load("best.pt", map_location="cpu")  # или "cuda"
model.eval()

# Классы
class_names = [
    "лодка опрокинута",
    "евроконтейнер опрокинут",
    "лодка захвачена",
    "евроконтейнер захвачен",
    "свободно",
]

# Преобразование
transform = T.Compose([
    T.Resize((224, 224)),
    T.ToTensor(),
    T.Normalize([0.485, 0.456, 0.406],
                [0.229, 0.224, 0.225])
])


def classify_and_draw(input_dir, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    frame_files = sorted(f for f in os.listdir(input_dir) if f.endswith(".jpg"))

    for file in tqdm(frame_files):
        img_path = os.path.join(input_dir, file)
        img = Image.open(img_path).convert("RGB")
        tensor = transform(img).unsqueeze(0)

        with torch.no_grad():
            preds = model(tensor)
            class_id = preds.argmax(1).item()
            label = class_names[class_id]

        # Рисуем на оригинале
        frame = cv2.imread(img_path)
        cv2.putText(frame, label, (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 2)
        cv2.imwrite(os.path.join(output_dir, file), frame)

    print("✅ Все кадры обработаны")

