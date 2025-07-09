from logging.handlers import TimedRotatingFileHandler
from multiprocessing import Pool
from webdav3.client import Client
from itertools import islice
from pathlib import Path
import subprocess
import logging
import requests
import json
import time
import os
import cv2


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
BLACKLISTED_REGISTRATORS = {"018270348452", "104039", "2024050601",
                            "118270348452"}
LABELSTUDIO_HOST = "http://localhost"
LABELSTUDIO_PORT = 8081
LABELSTUDIO_STORAGE_ID = 2
PROJECT_ID = 2
BASE_REMOTE_DIR = "/Tracker/Видео выгрузок"
LOCAL_VIDEO_DIR = str(Path(
    __file__).parent / "misc/videos_temp")  # Локальная папка для временных видео
FRAME_DIR_TEMP = str(Path(__file__).parent / "misc/frames_temp")
REMOTE_FRAME_DIR = "/Tracker/annotation_frames"
ANNOTATIONS_FILE = "annotations.json"
LABELSTUDIO_API_URL = f"{LABELSTUDIO_HOST}:{LABELSTUDIO_PORT}/api"
LABELSTUDIO_TOKEN = os.environ.get("labelstudio_token")
HEADERS = {"Authorization": f"Token {LABELSTUDIO_TOKEN}", }
DATASET_SPLIT = {"train": 0.7, "test": 0.2, "val": 0.1}
CYCLE_INTERVAL = 3600  # Время между циклами в секундах (1 час)
MOUNTED_PATH = "/mnt/webdav_frames"  # Локальный путь для монтирования WebDAV
MOUNTED_FRAME_DIR = os.path.join(MOUNTED_PATH, "frames")
FRAMES_PER_SECOND = 1
WEBDAV_REMOTE = "webdav:/Tracker/annotation_frames"
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
    """Проверяет, смонтирована ли папка WebDAV и работает ли соединение."""
    # 1. Проверяем, что путь действительно смонтирован
    if not os.path.ismount(MOUNTED_PATH):
        return False

    # 2. Пробуем получить список файлов — если endpoint мёртв, тут вылетит OSError
    try:
        test = os.listdir(MOUNTED_PATH)
        return True
    except OSError as e:
        logger.warning(f"Путь {MOUNTED_PATH} смонтирован, но недоступен: {e}")
        return False



def mount_webdav(from_systemd=False):
    """Монтирует WebDAV как локальную директорию."""
    if is_mounted():
        logger.info("WebDAV уже смонтирован.")
        return

    try:
        logger.info("Монтируем WebDAV...")
        os.makedirs(MOUNTED_PATH, exist_ok=True)
        args = [
            "rclone", "mount", WEBDAV_REMOTE, MOUNTED_PATH,
            "--no-modtime"
        ]
        if not from_systemd:
            args.append("--daemon")

        subprocess.run(args, check=True)
        time.sleep(2)
        if is_mounted():
            logger.info(f"WebDAV успешно смонтирован в {MOUNTED_PATH}")
        else:
            logger.error("WebDAV не смонтирован.")
    except Exception as e:
        logger.error(f"Ошибка при монтировании WebDAV: {e}")


def remount_webdav():
    """Пытается перемонтировать WebDAV, если он отключился."""
    if is_mounted():
        return

    logger.warning("WebDAV отключен. Перемонтируем...")

    subprocess.run(["fusermount", "-uz", MOUNTED_PATH], check=False)
    time.sleep(2)

    try:
        os.makedirs(MOUNTED_PATH, exist_ok=True)
        subprocess.run(
            ["rclone", "mount", WEBDAV_REMOTE, MOUNTED_PATH, "--daemon", "--no-modtime"],
            check=True
        )
        time.sleep(3)
        if is_mounted():
            logger.info("WebDAV успешно перемонтирован.")
        else:
            logger.error("Ошибка: WebDAV не смонтирован после попытки перемонтирования.")
    except Exception as e:
        logger.error(f"Ошибка при монтировании WebDAV: {e}")



def iter_video_files(path):
    """Генератор, лениво обходит WebDAV и yield'ит валидные mp4-файлы."""
    items = client.list(path)
    for item in items:
        item_path = sanitize_path(f"{path}/{item}")
        if client.is_dir(item_path):
            yield from iter_video_files(item_path)
        elif item.endswith(".mp4"):
            if any(reg in item for reg in BLACKLISTED_REGISTRATORS):
                logger.debug(f"Пропущен файл: {item_path} (в чёрном списке)")
                continue
            if item_path in downloaded_videos:
                continue
            yield item_path


def get_all_video_files(max_files=3):
    """Возвращает не более `max_files` валидных видео из WebDAV."""
    return list(islice(iter_video_files(BASE_REMOTE_DIR), max_files))


def download_videos(max_frames=1000, max_files=1):
    """Загружает видео из WebDAV по одному, пока не достигнет max_frames кадров."""
    remount_webdav()

    os.makedirs(LOCAL_VIDEO_DIR, exist_ok=True)

    try:
        items = client.list(REMOTE_FRAME_DIR)
        frame_count = sum(1 for item in items if item.endswith(".jpg"))
    except Exception as e:
        logger.error(f"Ошибка при проверке лимита кадров: {e}")
        return

    if frame_count >= max_frames:
        logger.warning(f"Пропущена загрузка видео: уже {frame_count} кадров.")
        return

    videos = get_all_video_files(max_files=max_files)
    logger.debug(f"Получены {len(videos)} видеофайлов")

    for video in videos:
        if frame_count >= max_frames:
            logger.info(f"Достигнут лимит кадров ({frame_count}/{max_frames}). Остановка загрузки.")
            break
        else:
            logger.info(f"В хранилище {frame_count}/{max_frames} кадров. Разрешается обработать еще")
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

            # Обновляем количество кадров после каждого видео
            try:
                items = client.list(REMOTE_FRAME_DIR)
                frame_count = sum(1 for item in items if item.endswith(".jpg"))
            except Exception as e:
                logger.warning(f"Ошибка при обновлении счётчика кадров: {e}")
                break

        except Exception as e:
            logger.error(f"Ошибка при скачивании {video}: {e}")

    save_download_history()
    logger.info("Загрузка завершена")


def sanitize_path(path):
    return path.replace("//", "/")

def count_remote_frames(webdav_client):
    """Подсчитывает количество кадров (jpg) в удалённой папке."""
    try:
        items = webdav_client.list(REMOTE_FRAME_DIR)
        jpg_count = sum(1 for item in items if item.endswith(".jpg"))
        return jpg_count
    except Exception as e:
        logger.error(f"Ошибка при подсчёте кадров в WebDAV: {e}")
        return 0

def clean_cloud_files(json_path, dry_run=False):
    import json, os

    # Загрузка размеченных файлов
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    marked_files = set()

    for task in data:
        anns = task.get("annotations")
        if not anns or not isinstance(anns, list):
            continue
        first_ann = anns[0]
        results = first_ann.get("result", [])
        if not results:
            continue
        try:
            class_name = results[0]["value"]["choices"][0]
            image_url = task["data"]["image"]
            image_name = os.path.basename(image_url)
            marked_files.add(image_name)
        except Exception as e:
            print(e)
            continue

    # Удаление мусора
    logger.debug(f"Всего размеченных файлов: {len(marked_files)}. Удаление неразмеченных...")
    deleted, skipped = 0, 0
    for file in os.listdir(MOUNTED_PATH):
        if not file.lower().endswith(".jpg"):
            continue
        if file not in marked_files:
            file_path = os.path.join(MOUNTED_PATH, file)
            if dry_run:
                logger.debug(f"[DRY RUN] Будет удалено: {file}")
            else:
                os.remove(file_path)
                #logger.debug(f"[DEL] {file}")
                deleted += 1
        else:
            skipped += 1

    logger.info(f"{'[DRY RUN] ' if dry_run else ''}Удаление завершено. Удалено: {deleted}, оставлено: {skipped}")


def delete_ls_tasks(dry_run=False):
    page = 1
    page_size = 100
    all_tasks = []
    seen_ids = set()

    logger.info("[LS] Загружаем все задачи с пагинацией (по страницам)...")

    while True:
        url = f"{LABELSTUDIO_API_URL}/tasks?project={PROJECT_ID}&page={page}&page_size={page_size}"
        logger.debug(f"[DEBUG] URL: {url}")
        r = requests.get(url, headers=HEADERS)

        if r.status_code != 200:
            logger.error(f"[LS] Ошибка {r.status_code}: {r.text}")
            return

        data = r.json()
        page_tasks = data.get("tasks", [])
        total = data.get("total")

        logger.debug(f"[DEBUG] page={page}, получено задач: {len(page_tasks)}, total={total}")

        if not page_tasks:
            logger.info("[LS] Получена пустая страница, завершаем.")
            break

        task_ids = [t['id'] for t in page_tasks]
        repeats = [tid for tid in task_ids if tid in seen_ids]
        if repeats:
            logger.warning(f"[LS] Повтор задач: {repeats[:5]} ... ({len(repeats)} всего), остановка.")
            break

        for task in page_tasks:
            seen_ids.add(task["id"])
            all_tasks.append(task)

        if len(all_tasks) >= total:
            logger.info("[LS] Все задачи получены.")
            break

        page += 1

    logger.info(f"[LS] Уникальных задач: {len(all_tasks)}")

    to_delete = []
    for task in all_tasks:
        anns = task.get("annotations")
        if not anns or not anns[0].get("result"):
            to_delete.append(task["id"])

    logger.info(f"[LS] К удалению отобрано: {len(to_delete)} задач")

    for task_id in to_delete:
        if dry_run:
            logger.info(f"[DRY RUN] Будет удалена задача {task_id}")
        else:
            r = requests.delete(f"{LABELSTUDIO_API_URL}/tasks/{task_id}", headers=HEADERS)
            if r.status_code == 204:
                logger.info(f"[LS DEL] Удалена задача {task_id}")
            else:
                logger.error(f"[ERR] Не удалось удалить задачу {task_id} — {r.status_code}: {r.text}")

    logger.info(f"{'[DRY RUN] ' if dry_run else ''}Удаление завершено. Всего удалено: {len(to_delete)}")





def extract_frames(video_path):
    """Разбивает видео на кадры и загружает в WebDAV с повторной попыткой при ошибках."""
    local_client = Client(WEBDAV_OPTIONS)
    cap = cv2.VideoCapture(video_path)
    existing_frames = count_remote_frames(webdav_client=local_client)
    if existing_frames >= 5000:
        logger.warning(
            f"Превышен лимит кадров в хранилище ({existing_frames} >= 5000). Пропускаем видео {video_path}.")
        cap.release()
        return video_path, False

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

    response = requests.post(sync_url, headers=HEADERS)

    if response.status_code == 200:
        logger.info("Хранилище успешно синхронизовано")
        return True
    else:
        logger.info(f"Результат синхронизации: {response.text}")
        return False


'''
def delete_blacklisted_files():
    """Удаляет все файлы, которые начинаются с '018270348452'."""
    PREFIX_TO_DELETE = "018270348452"  # Префикс для удаления

    def traverse_and_delete(path):
        """Рекурсивно обходит директорию и удаляет файлы с указанным префиксом."""
        items = client.list(path)
        for item in items:
            item_path = sanitize_path(f"{path}/{item}")

            if client.is_dir(item_path):
                traverse_and_delete(item_path)  # Рекурсивно идём внутрь
            elif item.startswith(PREFIX_TO_DELETE):
                print(f"🗑 Удаляю файл: {item_path}")
                client.clean(item_path)  # Удаляем файл

    traverse_and_delete(REMOTE_FRAME_DIR)
'''


def main_process_new_frames(max_frames=3000):
    logger.info("\n\U0001f504 Запущен основной цикл создания фреймов")
    process_video_loop(max_frames=max_frames)
    mount_webdav()
    sync_label_studio_storage()
    cleanup_videos()
    logger.info("\n✅ Цикл завершен.")

def process_video_loop(max_frames=3000):
    remount_webdav()
    os.makedirs(LOCAL_VIDEO_DIR, exist_ok=True)

    video_generator = iter_video_files(BASE_REMOTE_DIR)

    while True:
        # Проверяем количество кадров перед началом обработки видео
        try:
            items = client.list(REMOTE_FRAME_DIR)
            frame_count = sum(1 for item in items if item.endswith(".jpg"))
        except Exception as e:
            logger.error(f"Ошибка при проверке лимита кадров: {e}")
            break

        if frame_count >= max_frames:
            logger.info(f"\nДостигнут лимит кадров ({frame_count}/{max_frames}). Остановка загрузки.")
            break
        else:
            logger.info(f"\nВ данный момент в хранилище ({frame_count}/{max_frames}). Продолжаем обработку.")

        try:
            video = next(video_generator)
        except StopIteration:
            logger.info("Все видео обработаны")
            break

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
            continue

        # Нарезаем кадры сразу после скачивания
        logger.info(f"Нарезка кадров из {local_path}")
        video_path, success = extract_frames(local_path)
        if not success:
            logger.warning(f"Не удалось обработать видео: {video_path}")

        save_download_history()

