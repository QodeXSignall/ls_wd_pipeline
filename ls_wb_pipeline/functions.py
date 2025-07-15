from importlib.resources import read_text
from urllib.parse import urlparse, parse_qs, unquote
from ls_wb_pipeline.logger import logger
from ls_wb_pipeline.settings import *
from webdav3.client import Client
from itertools import islice
from pathlib import Path
import subprocess
import requests
import tempfile
import random
import json
import time
import os
import cv2
import re




# Конфигурация WebDAV
WEBDAV_OPTIONS = {
    'webdav_hostname': os.environ.get("webdav_host"),
    'webdav_login': os.environ.get("webdav_login"),
    'webdav_password': os.environ.get("webdav_password"),
    'disable_check': True  # Отключает кеширование
}
client = Client(WEBDAV_OPTIONS)



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


def remount_webdav(from_systemd=False):
    """Пытается перемонтировать WebDAV, если он отключился."""
    if is_mounted():
        return

    logger.warning("WebDAV отключен. Перемонтируем...")

    subprocess.run(["fusermount", "-uz", MOUNTED_PATH], check=False)
    time.sleep(3)

    try:
        os.makedirs(MOUNTED_PATH, exist_ok=True)
        args = ["rclone", "mount", WEBDAV_REMOTE, MOUNTED_PATH, "--no-modtime"]
        if not from_systemd:
            args.append("--daemon", )
        else:
            args += [
                "--vfs-cache-mode", "writes",
                "--dir-cache-time", "5s",
                "--poll-interval", "5s"
            ]
        subprocess.run(
            args,
            check=True
        )
        time.sleep(3)
        if is_mounted():
            logger.info("WebDAV успешно перемонтирован.")
        else:
            logger.error("Ошибка: WebDAV не смонтирован после попытки перемонтирования.")
    except Exception as e:
        logger.error(f"Ошибка при монтировании WebDAV: {e}")


def get_all_video_files(max_files=3):
    """Возвращает не более `max_files` валидных видео из WebDAV."""
    return list(islice(iter_video_files(BASE_REMOTE_DIR), max_files))


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

def clean_cloud_files_from_path(json_path, dry_run=False):
    # Загрузка размеченных файлов
    with open(json_path, "r", encoding="utf-8") as f:
        tasks = json.load(f)
    return clean_cloud_files_from_tasks(tasks, dry_run=dry_run)

def clean_cloud_files_from_tasks(tasks, dry_run=False, save_annotated=True):
    marked_files = []
    unmarked_files = []

    for task in tasks:
        try:
            image_url = task["data"]["image"]
            parsed = urlparse(image_url)
            query = parse_qs(parsed.query)
            image_path = query.get("d", [""])[0]
            image_name = os.path.basename(image_path)
            if check_if_ann(task):
                marked_files.append(image_path)
            else:
                unmarked_files.append(image_path)
        except Exception as e:
            logger.warning(f"[EXC] Ошибка при парсинге имени файла: {e}")
            continue
    files_to_delete = unmarked_files if save_annotated else marked_files + unmarked_files
    delete_files(files_to_delete, dry_run=dry_run)
    logger.info(f"{'[DRY RUN] ' if dry_run else ''}Удаление завершено. Удалено: {len(files_to_delete)}, "
                f"оставлено: {len(marked_files)}")
    return {"deleted_amount": len(files_to_delete), "saved": len(marked_files), "deleted": files_to_delete}

def check_if_ann(task):
    if task.get("total_annotations", 0):
        return True


def delete_all_cloud_files(dry_run=False):
    try:
        actual_files = [f for f in os.listdir(MOUNTED_PATH) if f.lower().endswith(".jpg")]
    except Exception as e:
        logger.error(f"Не удалось прочитать директорию {MOUNTED_PATH}: {e}")
        return {"error": e}
    report =  delete_files(files=actual_files, dry_run=dry_run)
    report["saved_amount"] = 0
    report["saved"] = []
    return report


def delete_files(files, dry_run=False):
    deleted_amount = 0
    deleted = []
    for file in files:
        if dry_run:
            logger.info(f"[DRY RUN] Будет удалено: {file}")
        else:
            try:
                os.remove(os.path.join("/mnt", file))
                deleted_amount += 1
                deleted.append(file)
            except Exception as e:
                logger.error(f"Ошибка при удалении {file}: {e}")
    return {"deleted": deleted, "deleted_amount": deleted_amount}


def get_all_tasks():
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
    return all_tasks


def delete_ls_tasks(tasks, dry_run=False, save_annotated=True):
    saved = 0

    logger.info("[LS] Загружаем все задачи с пагинацией (по страницам)...")

    to_delete = []
    for task in tasks:
        task_id = task.get("id")
        anns = task.get("total_annotations", 0)
        if not save_annotated or not anns:
            logger.debug(f"[LS DEBUG] Задача {task_id} отмечена под удаление - {'нет аннотаций' if not anns else 'отключено сохранение аннотаций'}")
            to_delete.append(task_id)
            continue

    logger.info(f"[LS] К удалению отобрано: {len(to_delete)} задач")

    for task_id in to_delete:
        if dry_run:
            logger.debug(f"[DRY RUN] Будет удалена задача {task_id}")
        else:
            r = requests.delete(f"{LABELSTUDIO_API_URL}/tasks/{task_id}", headers=HEADERS)
            if r.status_code == 204:
                logger.debug(f"[LS DEL] Удалена задача {task_id}")
            else:
                logger.error(f"[ERR] Не удалось удалить задачу {task_id} — {r.status_code}: {r.text}")
    try:
        saved = len(tasks) - len(to_delete)
    except:
        pass
    logger.info(f"{'[DRY RUN] ' if dry_run else ''}Удаление завершено. Всего удалено: {len(to_delete)}. Сохранено: {saved}")
    return to_delete, saved



def extract_frames(video_path, frames_per_second: float = None):
    """Разбивает видео на кадры и загружает в WebDAV с повторной попыткой при ошибках."""
    local_client = Client(WEBDAV_OPTIONS)
    cap = cv2.VideoCapture(video_path)
    existing_frames = count_remote_frames(webdav_client=local_client)
    logger.info(f"Извлекаем кадры из {video_path}. FPS - {frames_per_second}")
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

    frame_interval = max(int(fps / frames_per_second), 1)
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
    return True, video_path, saved_frame_count


def cleanup_videos():
    """Удаляет локальные видео после обработки."""
    logger.info("Удаление локальных видео")
    videos = [os.path.join(LOCAL_VIDEO_DIR, f) for f in
              os.listdir(LOCAL_VIDEO_DIR) if
              f.endswith(".mp4")]
    for video in videos:
        os.remove(video)
        logger.debug(f"Deleted {video}")


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


def main_process_new_frames(max_frames=3000, only_cargo_type: str = None, fps: float = None, video_name: str = None):
    logger.info("\n\U0001f504 Запущен основной цикл создания фреймов")
    result = process_video_loop(max_frames=max_frames, only_cargo_type=only_cargo_type, fps=fps, concrete_video_name=video_name)
    remount_webdav()
    time.sleep(3)
    sync_label_studio_storage()
    cleanup_videos()
    result["status"] = "frames processed"
    for item in client.list(REMOTE_FRAME_DIR):
        client.check(item)
    return result



def with_retries(func, max_attempts=3, delay=1.0, jitter=0.5, exceptions=(Exception,), log_prefix=""):
    for attempt in range(1, max_attempts + 1):
        try:
            return func()
        except exceptions as e:
            if attempt == max_attempts:
                raise
            logger.warning(f"{log_prefix}Ошибка (попытка {attempt}/{max_attempts}): {e}. Повтор через {delay} сек.")
            time.sleep(delay + random.uniform(0, jitter))


def parse_video_name(video_name: str):
    """Парсит имя файла и возвращает (reg_id, day, base_name)."""
    pattern = re.compile(
        r"(?P<reg_id>[A-Z0-9]+)_(?P<year>\d{4})\.(?P<month>\d{1,2})\.(?P<day>\d{1,2}) "
        r"(?P<start_time>\d{1,2}\.\d{1,2}\.\d{1,2})-(?P<end_time>\d{1,2}\.\d{1,2}\.\d{1,2})"
        r"\.(?P<video_format>\w+)"
    )
    match = pattern.match(video_name)
    if not match:
        raise ValueError(f"Неверный формат имени: {video_name}")

    reg_id = match.group("reg_id")
    day = f"{match.group('year')}.{match.group('month')}.{match.group('day')}"
    base_name = video_name.rsplit('.', 1)[0]
    return reg_id, day, base_name

def resolve_video_path(concrete_video_name: str, base_remote_dir: str, client) -> str:
    """
    Возвращает путь к .mp4-файлу по имени видео.
    Пример: concrete_video_name = "K630AX702_2025.5.21 8.54.11-8.55.34.mp4"
    """
    reg_id, day, base_name = parse_video_name(concrete_video_name)
    remote_dir = f"{base_remote_dir}/{reg_id}/{day}/{base_name}"
    try:
        items = client.list(remote_dir)
    except Exception as e:
        raise FileNotFoundError(f"Не удалось открыть папку: {remote_dir}. Ошибка: {e}")

    mp4_files = [f for f in items if f.endswith(".mp4")]
    if not mp4_files:
        raise FileNotFoundError(f"В папке {remote_dir} нет .mp4 файлов")

    return f"{remote_dir}/{mp4_files[0]}"

def process_video_loop(max_frames=3000, only_cargo_type: str = None, fps: float = None, concrete_video_name: str = None):
    remount_webdav()
    os.makedirs(LOCAL_VIDEO_DIR, exist_ok=True)
    downloaded_video_counter = 0

    # Ускоряем поиск видео, распарсив название и выполняя поиск в конкретной папке
    if concrete_video_name:
        try:
            resolved_path = resolve_video_path(concrete_video_name, BASE_REMOTE_DIR, client)
            video_generator = iter([resolved_path])  # Обрабатываем конкретное видео
        except Exception as e:
            return {"error": f"Ошибка при разрешении пути к видео {concrete_video_name}: {e}"}
    else:
        remote_dir = BASE_REMOTE_DIR
        video_generator = iter_video_files(remote_dir)

    result_dict = {"total_frames_downloaded": 0, "vid_process_results": [], "total_frames_in_storage": 0}
    while True:

        # Проверяем количество кадров перед началом обработки видео
        try:
            items = with_retries(lambda: client.list(REMOTE_FRAME_DIR), log_prefix="[WebDAV:list REMOTE_FRAME_DIR] ")
            frame_count = sum(1 for item in items if item.endswith(".jpg"))
        except Exception as e:
            logger.error(f"Ошибка при проверке лимита кадров: {e}")
            break

        if frame_count >= max_frames:
            logger.info(f"\nДостигнут лимит кадров ({frame_count}/{max_frames}). Остановка загрузки.")
            if not downloaded_video_counter:
                return {"error": f"Достигнут лимит кадров ({frame_count}/{max_frames})"}
            else:
                return result_dict
        #else:
        #   logger.info(f"\nВ данный момент в хранилище ({frame_count}/{max_frames}). Продолжаем обработку.")

        try:
            video = next(video_generator)
        except StopIteration:
            logger.info("Все видео обработаны")
            return {"error": "Все видео обработаны"}

        current_video_name = os.path.basename(video)
        if concrete_video_name and concrete_video_name != current_video_name:
            logger.debug(f"Пропущен файл: {current_video_name} (ищем видео {concrete_video_name})")
            continue

        if video in downloaded_videos and not concrete_video_name:
            logger.debug(f"Пропущено {video}, уже скачано.")
            continue

        # ➕ Получаем тип груза из report.json
        report_path = os.path.join(os.path.dirname(video), "report.json")
        try:
            with tempfile.NamedTemporaryFile(mode="w+b", delete=False) as tmpf:
                client.download_sync(remote_path=report_path, local_path=tmpf.name)
                tmpf.seek(0)
                report_data = json.load(tmpf)
                switch_events = report_data.get("switch_events", [])
                if switch_events and isinstance(switch_events, list):
                    switch_code = switch_events[0].get("switch")
                    if switch_code == 22:
                        cargo_type = "bunker"
                    elif switch_code == 23:
                        cargo_type = "euro"
                    else:
                        cargo_type = "unknown"
                    logger.info(f"[TYPE] {video} → тип груза: {cargo_type} (switch={switch_code})")
                else:
                    logger.warning(f"[WARN] Нет switch_events в {report_path}")
        except Exception as e:
            logger.warning(f"[WARN] Не удалось загрузить или распарсить report.json для {video}: {e}")

        if only_cargo_type and cargo_type != only_cargo_type:
            logger.debug(f"Тип груза - {cargo_type}. Но качаем только - {only_cargo_type}, пропуск...")
            continue

        local_path = os.path.join(LOCAL_VIDEO_DIR, current_video_name)
        logger.info(f"Скачивание {video}")
        try:
            temp_path = local_path + ".part"
            with_retries(lambda: client.download_sync(remote_path=video, local_path=temp_path),
                         log_prefix=f"[WebDAV:download {video}] ")
            os.rename(temp_path, local_path)
            downloaded_videos.add(video)
            logger.info(f"Скачано {video} в {local_path}")
            downloaded_video_counter += 1
        except Exception as e:
            logger.error(f"Ошибка при скачивании {video}: {e}")
            continue

        # Нарезаем кадры сразу после скачивания
        logger.info(f"Нарезка кадров из {local_path}")
        if not fps:
            fps = 1 if cargo_type == "euro" else 0.25
        success, video_path, frames = (extract_frames(local_path, frames_per_second=fps))
        if not success:
            logger.warning(f"Не удалось обработать видео: {video_path}")
        result_dict["vid_process_results"].append({"video_path": video_path, "frames": frames, "success": success})
        result_dict["total_frames_downloaded"] += int(frames)
        result_dict["total_frames_in_storage"] = frame_count + int(frames)
        save_download_history()
        if concrete_video_name:
            break
    return result_dict


