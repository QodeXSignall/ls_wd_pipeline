import os
import cv2
import json
import time
import requests
import subprocess
import logging
from logging.handlers import TimedRotatingFileHandler
from urllib.parse import quote, unquote
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
LABELSTUDIO_HOST = "http://localhost"
LABELSTUDIO_PORT = 8081
LABELSTUDIO_STORAGE_ID = 1
BASE_REMOTE_DIR = "/Tracker/Видео выгрузок"
LOCAL_VIDEO_DIR = str(Path(
    __file__).parent / "misc/videos_temp")  # Локальная папка для временных видео
FRAME_DIR_TEMP = str(Path(__file__).parent / "misc/frames_temp")
REMOTE_FRAME_DIR = "/Tracker/annotation_frames"
ANNOTATIONS_FILE = "annotations.json"
LABELSTUDIO_API_URL = f"{LABELSTUDIO_HOST}:{LABELSTUDIO_PORT}/api/projects/1/import"
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


def remount_webdav():
    """Пытается перемонтировать WebDAV, если он отключился."""
    if is_mounted():
        return

    logger.warning("WebDAV отключен. Перемонтируем...")
    subprocess.run(["fusermount", "-u", MOUNTED_PATH], check=False)
    time.sleep(2)

    try:
        os.makedirs(MOUNTED_PATH, exist_ok=True)
        subprocess.run(
            ["rclone", "mount", WEBDAV_REMOTE, MOUNTED_PATH, "--daemon",
             "--no-modtime"], check=True)
        time.sleep(3)  # Даем время на монтирование
        if is_mounted():
            logger.info("WebDAV успешно перемонтирован.")
        else:
            logger.error(
                "Ошибка: WebDAV не смонтирован после попытки перемонтирования.")
    except Exception as e:
        logger.error(f"Ошибка при монтировании WebDAV: {e}")



import urllib.parse


def normalize_video_structure():
    registrators = client.list(BASE_REMOTE_DIR)

    for reg in registrators:
        reg_path = os.path.join(BASE_REMOTE_DIR, reg)
        if not client.is_dir(reg_path):
            continue

        date_dirs = client.list(reg_path)
        for date in date_dirs:
            date_path = os.path.join(reg_path, date)
            if not client.is_dir(date_path):
                continue

            video_files = client.list(date_path)
            for video in video_files:
                video_path = os.path.join(date_path, video)
                if client.is_dir(video_path):
                    continue  # Пропускаем папки, обрабатываем только файлы

                if not video.endswith(".mp4"):
                    continue  # Оставляем только видеофайлы

                # Создаём папку с названием видео
                video_folder = os.path.join(date_path, os.path.splitext(video)[0])
                if not client.check(video_folder):
                    client.mkdir(video_folder)

                # Перемещаем видео в новую папку
                new_video_path = os.path.join(video_folder, video)
                client.move(video_path, new_video_path)

                # Создаём before_pics/ и after_pics/
                before_pics = os.path.join(video_folder, "before_pics")
                after_pics = os.path.join(video_folder, "after_pics")

                if not client.check(before_pics):
                    client.mkdir(before_pics)
                if not client.check(after_pics):
                    client.mkdir(after_pics)

                print(f"✅ {video} перемещён в {video_folder} и созданы before_pics/, after_pics/")

def download_videos():
    """Загружает новые видеофайлы из WebDAV."""
    remount_webdav()
    all_videos = get_all_video_files()
    os.makedirs(LOCAL_VIDEO_DIR, exist_ok=True)

    for video in all_videos:
        if video in downloaded_videos:
            logger.debug(f"Пропущено {video}, уже скачано.")
            continue

        local_path = os.path.join(LOCAL_VIDEO_DIR, os.path.basename(video))
        logger.info(f"Скачивание {video}")
        try:
            temp_path = local_path + ".part"
            client.download_sync(remote_path=video, local_path=temp_path)
            os.rename(temp_path, local_path)
            downloaded_videos.add(video)
            logger.info(f"Скачано {video} в {local_path}")
        except Exception as e:
            logger.error(f"Ошибка при скачивании {video}: {e}")
    save_download_history()
    logger.info("Загрузка завершена")


def sanitize_path(path):
    return path.replace("//", "/")


def get_all_video_files():
    """Рекурсивно обходит директорию BASE_REMOTE_DIR и возвращает список всех mp4 файлов."""
    all_videos = []
    registrators = client.list(BASE_REMOTE_DIR)

    for reg in registrators:
        reg_path = sanitize_path(f"{BASE_REMOTE_DIR}/{reg}")
        if not client.is_dir(reg_path):
            continue

        date_dirs = client.list(reg_path)
        for date in date_dirs:
            date_path = sanitize_path(f"{reg_path}/{date}")
            if not client.is_dir(date_path):
                continue

            video_dirs = client.list(date_path)
            for vid_dir in video_dirs:
                video_path = sanitize_path(f"{date_path}/{vid_dir}")
                if not client.is_dir(video_path):
                    continue

                video_files = client.list(video_path)
                for video in video_files:
                    if video.endswith(".mp4"):
                        all_videos.append(
                            sanitize_path(f"{video_path}/{video}"))
    return all_videos


def extract_frames(video_path):
    """Разбивает видео на кадры и загружает в WebDAV с повторной попыткой при ошибках."""
    local_client = Client(WEBDAV_OPTIONS)
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        logger.error(f"Ошибка: Не удалось открыть видео {video_path}")
        return video_path, False  # Возвращаем видео с ошибкой

    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0:
        logger.error(f"Ошибка: FPS не определен для {video_path}")
        cap.release()
        return video_path, False

    frame_interval = max(int(fps / FRAMES_PER_SECOND), 1)
    frame_count = 0
    saved_frame_count = 0
    max_retries = 3  # Количество повторных попыток при ошибке

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
                success = False
                for attempt in range(1, max_retries + 1):
                    try:
                        local_client.upload_sync(remote_path=remote_frame_path,
                                                 local_path=local_frame_path)
                        os.remove(local_frame_path)
                        success = True
                        break  # Успешная загрузка, выходим из цикла
                    except Exception as e:
                        logger.error(
                            f"Ошибка при загрузке кадра {frame_filename} (Попытка {attempt}/{max_retries}): {e}")
                        time.sleep(5)  # Ждем 5 секунд перед повторной попыткой

                if not success:
                    logger.error(
                        f"Не удалось загрузить кадр {frame_filename} после {max_retries} попыток.")
                    cap.release()
                    return video_path, False
            else:
                logger.warning(
                    f"Предупреждение: Кадр {local_frame_path} не был создан.")
            saved_frame_count += 1
        frame_count += 1

    cap.release()
    logger.info(
        f"Извлечено и загружено {saved_frame_count} кадров из {video_path}")
    return video_path, True


def cleanup_videos():
    """Удаляет локальные видео после обработки."""
    logger.info("Удаление локальных видео")
    videos = [os.path.join(LOCAL_VIDEO_DIR, f) for f in
              os.listdir(LOCAL_VIDEO_DIR) if
              f.endswith(".mp4")]
    for video in videos:
        os.remove(video)
        print(f"Deleted {video}")


def sync_label_studio_storage():
    """
    Функция для синхронизации локального хранилища в Label Studio через API.

    :return: Результат синхронизации (True - успех, False - ошибка)
    """
    remount_webdav()
    sync_url = f"{LABELSTUDIO_HOST}:{LABELSTUDIO_PORT}/api/storages/localfiles/{LABELSTUDIO_STORAGE_ID}/sync"
    headers = {"Authorization": f"Token {LABELSTUDIO_TOKEN}", }

    response = requests.post(sync_url, headers=headers)

    if response.status_code == 200:
        logger.info("Хранилище успешно синхронизовано")
        return True
    else:
        logger.info(f"Результат синхронизации: {response.text}")
        return False


def main():
    logger.info("Запущен основной цикл")
    while True:
        normalize_directory_structure()
        download_videos()
        videos = [os.path.join(LOCAL_VIDEO_DIR, f) for f in
                  os.listdir(LOCAL_VIDEO_DIR) if f.endswith(".mp4")]
        with Pool(processes=4) as pool:
            results = pool.map(extract_frames, videos)

        failed_videos = [video for video, success in results if not success]
        if failed_videos:
            logger.warning(
                f"Не удалось обработать следующие видео: {failed_videos}")

        mount_webdav()
        # import_to_labelstudio()

        sync_label_studio_storage()
        cleanup_videos()
        logger.info("Цикл завершен. Ожидание...")
        time.sleep(CYCLE_INTERVAL)


if __name__ == "__main__":
    main()
