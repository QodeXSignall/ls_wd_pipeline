import os
import cv2
import json
import time
import requests
import subprocess
import logging
from logging.handlers import TimedRotatingFileHandler
from multiprocessing import Pool
from webdav3.client import Client
from pathlib import Path

# Настройка логирования
LOG_DIR = str(Path(__file__).parent / "logs")
os.makedirs(LOG_DIR, exist_ok=True)
LOG_FILE = os.path.join(LOG_DIR, "pipeline.log")

logger = logging.getLogger("PipelineLogger")
logger.setLevel(logging.DEBUG)
formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')

# Обработчик для записи логов в файл
file_handler = TimedRotatingFileHandler(LOG_FILE, when="midnight", interval=1,
                                        backupCount=30, encoding='utf-8')
file_handler.setFormatter(formatter)
file_handler.setLevel(logging.INFO)
logger.addHandler(file_handler)

# Обработчик для вывода логов в stdout
console_handler = logging.StreamHandler()
console_handler.setFormatter(formatter)
console_handler.setLevel(logging.DEBUG)
logger.addHandler(console_handler)

# Конфигурация WebDAV
WEBDAV_OPTIONS = {
    'webdav_hostname': os.environ.get("webdav_host"),
    'webdav_login': os.environ.get("webdav_login"),
    'webdav_password': os.environ.get("webdav_password"),
    'disable_check': True  # Отключает кеширование
}
client = Client(WEBDAV_OPTIONS)

# Параметры
BASE_URL = r"https://cloud.mail.ru/public/tnYz/VA3qxQgFa"
BASE_REMOTE_DIR = "/Tracker/Видео выгрузок"
LOCAL_VIDEO_DIR = str(Path(
    __file__).parent / "misc/videos_temp")  # Локальная папка для временных видео
FRAME_DIR_TEMP = str(Path(__file__).parent / "misc/frames_temp")
REMOTE_FRAME_DIR = "/Tracker/annotation_frames"
ANNOTATIONS_FILE = "annotations.json"
LABELSTUDIO_API_URL = "http://localhost:8081/api/projects/1/import"
LABELSTUDIO_TOKEN = os.environ.get("labelstudio_token")
DATASET_SPLIT = {"train": 0.7, "test": 0.2, "val": 0.1}
CYCLE_INTERVAL = 3600  # Время между циклами в секундах (1 час)
MOUNTED_PATH = "/mnt/webdav_frames"  # Локальный путь для монтирования WebDAV
FRAMES_PER_SECOND = 1
WEBDAV_REMOTE = "webdav:/Tracker/Видео выгрузок/104039/Тесты для wb_ls_pipeline/frames"
DOWNLOAD_HISTORY_FILE = "downloaded_videos.json"

# Загруженные файлы
if os.path.exists(DOWNLOAD_HISTORY_FILE):
    with open(DOWNLOAD_HISTORY_FILE, "r") as f:
        downloaded_videos = set(json.load(f))
else:
    downloaded_videos = set()

def save_download_history():
    with open(DOWNLOAD_HISTORY_FILE, "w") as f:
        json.dump(list(downloaded_videos), f)

def is_mounted():
    """Проверяет, смонтирована ли папка WebDAV."""
    output = subprocess.run(["mount"], capture_output=True, text=True)
    return MOUNTED_PATH in output.stdout


def mount_webdav():
    """Монтирует WebDAV папку с кадрами как локальную директорию."""
    if is_mounted():
        logger.info("WebDAV уже смонтирован.")
        return

    try:
        logger.info("Монтируем WebDAV...")
        os.makedirs(MOUNTED_PATH, exist_ok=True)
        subprocess.run(
            ["rclone", "mount", WEBDAV_REMOTE, MOUNTED_PATH, "--daemon",
             "--no-modtime"], check=True)
        time.sleep(3)  # Даем время на монтирование
        if is_mounted():
            logger.info(f"WebDAV успешно смонтирован в {MOUNTED_PATH}")
        else:
            logger.error("Ошибка: WebDAV не смонтирован.")
    except Exception as e:
        logger.error(f"Ошибка при монтировании WebDAV: {e}")


def download_videos():
    """Загружает новые видеофайлы из WebDAV."""
    all_videos = get_all_video_files()
    os.makedirs(LOCAL_VIDEO_DIR, exist_ok=True)

    for video in all_videos:
        if video in downloaded_videos:
            logger.debug(f"Пропущено {video}, уже скачано.")
            continue

        local_path = os.path.join(LOCAL_VIDEO_DIR, os.path.basename(video))
        logger.info(f"Скачивание {video}")
        client.download_sync(remote_path=video, local_path=local_path)
        downloaded_videos.add(video)
        logger.info(f"Скачано {video} в {local_path}")
    save_download_history()
    logger.info("Загрузка завершена")


def get_all_video_files():
    """Рекурсивно обходит директорию BASE_REMOTE_DIR и возвращает список всех mp4 файлов."""
    all_videos = []
    registrators = client.list(BASE_REMOTE_DIR)

    for reg in registrators:
        reg_path = f"{BASE_REMOTE_DIR}/{reg}"
        if not client.is_dir(reg_path):
            continue

        date_dirs = client.list(reg_path)
        for date in date_dirs:
            date_path = f"{reg_path}/{date}"
            if not client.is_dir(date_path):
                continue

            video_dirs = client.list(date_path)
            for vid_dir in video_dirs:
                video_path = f"{date_path}/{vid_dir}"
                if not client.is_dir(video_path):
                    continue

                video_files = client.list(video_path)
                for video in video_files:
                    if video.endswith(".mp4"):
                        all_videos.append(f"{video_path}/{video}")
    return all_videos

def extract_frames(video_path, frames_per_second=FRAMES_PER_SECOND):
    """Разбивает видео на кадры и загружает в WebDAV, извлекая заданное количество кадров в секунду."""
    client = Client(WEBDAV_OPTIONS)
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_interval = max(int(fps / frames_per_second), 1)
    frame_count = 0
    saved_frame_count = 0

    logger.info(
        f"Извлекаем кадры из {video_path} (FPS: {fps}, Интервал: {frame_interval})")

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        if frame_count % frame_interval == 0:
            frame_filename = f"{Path(video_path).stem}_{saved_frame_count:06d}.jpg"
            local_frame_path = os.path.join(FRAME_DIR_TEMP, frame_filename)
            remote_frame_path = f"{REMOTE_FRAME_DIR}/{frame_filename}"

            cv2.imwrite(local_frame_path, frame)
            if os.path.exists(local_frame_path):
                client.upload_sync(remote_path=remote_frame_path,
                                   local_path=local_frame_path)
                os.remove(local_frame_path)
            else:
                logger.warning(
                    f"Предупреждение: Кадр {local_frame_path} не был создан.")
            saved_frame_count += 1
        frame_count += 1

    cap.release()
    logger.info(
        f"Извлечено и загружено {saved_frame_count} кадров из {video_path}")


def import_to_labelstudio():
    """Импортирует изображения из смонтированной WebDAV папки в LabelStudio."""
    if not is_mounted():
        logger.error("Ошибка: WebDAV не смонтирован! Прерываем импорт.")
        return

    images = client.list(REMOTE_FRAME_DIR)
    images = [img for img in images if img.endswith(".jpg")]
    tasks = [{"data": {"image": f"file://{MOUNTED_PATH}/{img}"}} for img in images]

    headers = {
        "Authorization": f"Token {LABELSTUDIO_TOKEN}",
        "Content-Type": "application/json; charset=utf-8"
    }

    response = requests.post(LABELSTUDIO_API_URL, headers=headers, json=tasks)
    logger.info(f"Импортировано изображений в LabelStudio: {response.status_code}")


def main():
    logger.info("Запущен основной цикл")
    while True:
        download_videos()
        videos = [os.path.join(LOCAL_VIDEO_DIR, f) for f in
                  os.listdir(LOCAL_VIDEO_DIR) if f.endswith(".mp4")]
        with Pool(processes=4) as pool:
            pool.map(extract_frames, videos)
        mount_webdav()
        import_to_labelstudio()
        logger.info("Цикл завершен. Ожидание...")
        time.sleep(CYCLE_INTERVAL)


if __name__ == "__main__":
    main()
