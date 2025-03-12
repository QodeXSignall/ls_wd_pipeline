import os
import cv2
import json
import time
import requests
from multiprocessing import Pool
from webdav3.client import Client
from pathlib import Path

# Конфигурация WebDAV
WEBDAV_OPTIONS = {
    'webdav_hostname': os.environ.get("webdav_host"),
    'webdav_login': os.environ.get("webdav_login"),
    'webdav_password': os.environ.get("webdav_password"),
    'disable_check': True  # Отключает кеширование
}
client = Client(WEBDAV_OPTIONS)

# Параметры
REMOTE_VIDEO_DIR = "/Tracker/Видео выгрузок/104039/Тесты для wb_ls_pipeline/source_videos"
LOCAL_VIDEO_DIR = str(Path(
    __file__).parent / "misc/videos_temp")  # Локальная папка для временных видео
FRAME_DIR_TEMP = str(Path(
    __file__).parent / "misc/frames_temp")
REMOTE_FRAME_DIR = r"/Tracker/Видео выгрузок/104039/Тесты для wb_ls_pipeline/frames"
ANNOTATIONS_FILE = "annotations.json"
LABELSTUDIO_API_URL = "http://localhost:8081/api/projects/1/import"
LABELSTUDIO_TOKEN = os.environ.get("labelstudio_token")
DATASET_SPLIT = {"train": 0.7, "test": 0.2, "val": 0.1}
CYCLE_INTERVAL = 3600  # Время между циклами в секундах (1 час)
MOUNTED_PATH = "/mnt/webdav_frames"  # Локальный путь для монтирования WebDAV
FRAMES_PER_SECOND = 1

def download_videos():
    """Загружает видеофайлы из WebDAV."""
    files = client.list(REMOTE_VIDEO_DIR)
    os.makedirs(LOCAL_VIDEO_DIR, exist_ok=True)

    for file in files:
        if not file.lower().endswith(".mp4"):  # Проверяем расширение файла
            print(f"Skipping {file}, not an MP4 file.")
            continue

        remote_file_path = f"{REMOTE_VIDEO_DIR}/{file}"  # Формируем полный путь
        local_path = os.path.join(LOCAL_VIDEO_DIR, os.path.basename(file))
        print(f"Downloading {remote_file_path}")
        if not os.path.exists(local_path):
            client.download_sync(remote_path=remote_file_path,
                                 local_path=local_path)
            print(f"Downloaded {remote_file_path} to {local_path}")
    print("Downloading done")


def extract_frames(video_path, frames_per_second=FRAMES_PER_SECOND):
    """Разбивает видео на кадры и загружает в WebDAV, извлекая заданное количество кадров в секунду."""

    # Локальный WebDAV-клиент в каждом процессе
    client = Client(WEBDAV_OPTIONS)

    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_interval = max(int(fps / frames_per_second), 1)  # Интервал кадров
    frame_count = 0
    saved_frame_count = 0

    print(
        f"Extracting frames from {video_path} (FPS: {fps}, Interval: {frame_interval})")

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        if frame_count % frame_interval == 0:  # Сохраняем только каждые frame_interval кадров
            frame_filename = f"{Path(video_path).stem}_{saved_frame_count:06d}.jpg"
            local_frame_path = os.path.join(FRAME_DIR_TEMP, frame_filename)
            remote_frame_path = f"/Tracker/Видео выгрузок/104039/Тесты для wb_ls_pipeline/frames/{frame_filename}"

            cv2.imwrite(local_frame_path, frame)
            if os.path.exists(
                    local_frame_path):  # Проверяем, что файл существует перед загрузкой
                client.upload_sync(remote_path=remote_frame_path,
                                   local_path=local_frame_path)
                os.remove(local_frame_path)
            else:
                print(f"Warning: Frame {local_frame_path} was not created.")

            saved_frame_count += 1

        frame_count += 1

    cap.release()
    print(
        f"Extracted and uploaded {saved_frame_count} frames from {video_path}")


def mount_webdav():
    """Монтирует WebDAV папку с кадрами как локальную директорию."""
    if not os.path.exists(MOUNTED_PATH):
        try:
            os.makedirs(MOUNTED_PATH, exist_ok=True)
        except PermissionError:
            print(
                f"Ошибка: Недостаточно прав для создания {MOUNTED_PATH}. Запусти с sudo или измени права.")
            return

    if not os.path.ismount(MOUNTED_PATH):
        os.system(
            f"mount -t davfs {WEBDAV_OPTIONS['webdav_hostname']}/{REMOTE_FRAME_DIR} {MOUNTED_PATH}"
        )
        print(f"Mounted WebDAV storage at {MOUNTED_PATH}")


def import_to_labelstudio():
    """Импортирует изображения из смонтированной WebDAV папки в LabelStudio."""
    images = [f for f in os.listdir(MOUNTED_PATH) if f.endswith(".jpg")]
    tasks = [{"data": {"image": f"/mnt/webdav_frames/{img}"}} for img in
             images]
    print(tasks)
    headers = {
        "Authorization": f"Token {LABELSTUDIO_TOKEN}",
        "Content-Type": "application/json; charset=utf-8"
    }
    print(json.dumps(tasks, indent=2))
    response = requests.post(LABELSTUDIO_API_URL, headers=headers,
                             data=json.dumps(tasks))
    print("Imported images to LabelStudio:", response.status_code)


def cleanup_videos():
    """Удаляет локальные видео после обработки."""
    videos = [os.path.join(LOCAL_VIDEO_DIR, f) for f in
              os.listdir(LOCAL_VIDEO_DIR) if
              f.endswith(".mp4")]
    for video in videos:
        os.remove(video)
        print(f"Deleted {video}")


def main():
    print(f"Mainloop has been launched")
    while True:
        download_videos()
        videos = [os.path.join(LOCAL_VIDEO_DIR, f) for f in
                  os.listdir(LOCAL_VIDEO_DIR) if
                  f.endswith(".mp4")]

        with Pool(processes=4) as pool:
            pool.map(extract_frames, videos)

        mount_webdav()
        import_to_labelstudio()
        cleanup_videos()

        print(f"Cycle complete. Sleeping for {CYCLE_INTERVAL} seconds...")
        time.sleep(CYCLE_INTERVAL)


if __name__ == "__main__":
    main()
