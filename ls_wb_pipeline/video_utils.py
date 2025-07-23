
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


# Загруженные файлы
if os.path.exists(DOWNLOAD_HISTORY_FILE):
    with open(DOWNLOAD_HISTORY_FILE, "r") as f:
        downloaded_videos = set(json.load(f))
else:
    downloaded_videos = set()


def save_download_history():
    with open(DOWNLOAD_HISTORY_FILE, "w") as f:
        json.dump(list(downloaded_videos), f)


def list_remote_videos(base_dir, client, concrete_video_name=None):
    if concrete_video_name:
        try:
            resolved_path = resolve_video_path(concrete_video_name, base_dir, client)
            return iter([resolved_path])
        except Exception as e:
            logger.error(f"Ошибка при разрешении видео {concrete_video_name}: {e}")
            return iter([])
    return iter_video_files(base_dir)


def parse_cargo_type(video_path, client) -> str:
    report_path = os.path.join(os.path.dirname(video_path), "report.json")
    try:
        with tempfile.NamedTemporaryFile(mode="w+b", delete=False) as tmpf:
            client.download_sync(remote_path=report_path, local_path=tmpf.name)
            tmpf.seek(0)
            report_data = json.load(tmpf)
            switch_events = report_data.get("switch_events", [])
            if switch_events and isinstance(switch_events, list):
                switch_code = switch_events[0].get("switch")
                if switch_code == 22:
                    return "bunker"
                elif switch_code == 23:
                    return "euro"
            logger.warning(f"[WARN] Неизвестный switch_events в {report_path}")
    except Exception as e:
        logger.warning(f"[WARN] Не удалось загрузить или распарсить report.json: {e}")
    return "unknown"


def should_skip_video(video_path, downloaded, only_cargo_type, current_type, concrete_video_name):
    video_name = os.path.basename(video_path)
    if concrete_video_name and video_name != concrete_video_name:
        return True
    if video_path in downloaded and not concrete_video_name:
        return True
    if only_cargo_type and current_type != only_cargo_type:
        logger.debug(f"Тип груза {current_type}, фильтр: {only_cargo_type}")
        return True
    return False


def download_video(client, remote_path, local_path):
    temp_path = local_path + ".part"
    with_retries(lambda: client.download_sync(remote_path=remote_path, local_path=temp_path),
                 log_prefix=f"[WebDAV:download {remote_path}] ")
    os.rename(temp_path, local_path)


def cut_video_to_frames(local_path, fps):
    return extract_frames(local_path, frames_per_second=fps)


def process_video_loop(max_frames=3000, only_cargo_type: str = None, fps: float = None, concrete_video_name: str = None):
    remount_webdav()
    os.makedirs(LOCAL_VIDEO_DIR, exist_ok=True)
    downloaded_video_counter = 0
    result_dict = {"total_frames_downloaded": 0, "vid_process_results": [], "total_frames_in_storage": 0}

    try:
        items = with_retries(lambda: client.list(REMOTE_FRAME_DIR), log_prefix="[WebDAV:list REMOTE_FRAME_DIR] ")
        frame_count = sum(1 for item in items if item.endswith(".jpg"))
    except Exception as e:
        logger.error(f"Ошибка при проверке лимита кадров: {e}")
        return result_dict

    if frame_count >= max_frames:
        logger.info(f"Достигнут лимит кадров ({frame_count}/{max_frames}). Остановка загрузки.")
        return {"error": f"Лимит кадров достигнут ({frame_count}/{max_frames})"}

    video_generator = list_remote_videos(BASE_REMOTE_DIR, client, concrete_video_name)

    for video in video_generator:
        video_name = os.path.basename(video)

        cargo_type = parse_cargo_type(video, client)
        if should_skip_video(video, downloaded_videos, only_cargo_type, cargo_type, concrete_video_name):
            logger.debug(f"Пропущен файл: {video_name}")
            continue

        local_path = os.path.join(LOCAL_VIDEO_DIR, video_name)
        logger.info(f"Скачивание {video}")
        try:
            download_video(client, video, local_path)
            downloaded_videos.add(video)
            downloaded_video_counter += 1
        except Exception as e:
            logger.error(f"Ошибка при скачивании {video}: {e}")
            continue

        effective_fps = fps or (FRAMES_PER_SECOND_EURO if cargo_type == "euro" else FRAMES_PER_SECOND_BUNKER)
        logger.info(f"Нарезка кадров из {local_path}. FPS: {effective_fps}")
        success, video_path, frames = cut_video_to_frames(local_path, fps=effective_fps)
        if not success:
            logger.warning(f"Не удалось обработать видео: {video_path}")
            continue

        result_dict["vid_process_results"].append(
            {"video_path": video_path, "frames": frames, "success": success, "cargo_type": cargo_type})
        result_dict["total_frames_downloaded"] += int(frames)
        result_dict["total_frames_in_storage"] += int(frames)
        save_download_history()

        if concrete_video_name:
            break

        try:
            items = with_retries(lambda: client.list(REMOTE_FRAME_DIR), log_prefix="[WebDAV:list REMOTE_FRAME_DIR] ")
            frame_count = sum(1 for item in items if item.endswith(".jpg"))
        except Exception as e:
            logger.error(f"Ошибка при повторной проверке лимита кадров: {e}")
            break

        if frame_count >= max_frames:
            logger.info(f"Достигнут лимит кадров ({frame_count}/{max_frames}).")
            break

    return result_dict