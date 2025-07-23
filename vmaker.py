import torch
import torchvision.transforms as T
from PIL import Image
import cv2
import os
from tqdm import tqdm
import cv2
import os
import sys
from torch.serialization import safe_globals
from PIL import ImageDraw, ImageFont


sys.path.append("/Users/artur/PycharmProjects/yolov5")

from models.yolo import ClassificationModel  # если у тебя есть yolov5



def video_to_frames(video_path, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    cap = cv2.VideoCapture(video_path)
    i = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        cv2.imwrite(f"{output_dir}/frame_{i:05d}.jpg", frame)
        i += 1
    cap.release()
    print(f"✅ {i} кадров сохранено")

# Модель
ckpt = torch.load("best.pt", map_location="cpu", weights_only=False)
model = ckpt["model"].float().eval()
model.names = [
    "лодка опрокинута",
    "евроконтейнер опрокинут",
    "лодка захвачена",
    "евроконтейнер захвачен",
    "свободно",
]

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

    font = ImageFont.truetype("/System/Library/Fonts/Supplemental/Arial.ttf", 32)  # под Mac

    for file in tqdm(frame_files):
        img_path = os.path.join(input_dir, file)
        img = Image.open(img_path).convert("RGB")
        tensor = transform(img).unsqueeze(0)

        with torch.no_grad():
            preds = model(tensor)
            class_id = preds.argmax(1).item()
            label = class_names[class_id]

        # Рисуем текст на изображении через PIL
        draw = ImageDraw.Draw(img)
        draw.text((400, 30), label, font=font, fill=(255, 0, 0))

        img.save(os.path.join(output_dir, file))

    print("✅ Все кадры обработаны")

# Использование

def frames_to_video(input_dir, output_video_path, fps=25):
    frame_files = sorted(f for f in os.listdir(input_dir) if f.endswith(".jpg"))
    sample = cv2.imread(os.path.join(input_dir, frame_files[0]))
    height, width, _ = sample.shape

    out = cv2.VideoWriter(output_video_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))

    for file in frame_files:
        frame = cv2.imread(os.path.join(input_dir, file))
        out.write(frame)
    out.release()
    print(f"🎞 Видео сохранено: {output_video_path}")

# Использование

video_to_frames("test.mp4", "frames")
classify_and_draw("frames", "frames_labeled")
frames_to_video("frames_labeled", "result_video.mp4", fps=5)