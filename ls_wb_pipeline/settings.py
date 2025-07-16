from pathlib import Path
import os

# Параметры
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CONFIG_PATH = os.path.join(BASE_DIR, "config.cfg")
BLACKLISTED_REGISTRATORS = {"018270348452", "104039", "2024050601",
                            "118270348452"}
DATASET_PATH = os.path.join(BASE_DIR, "dataset")
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
FRAMES_PER_SECOND_EURO = 1
FRAMES_PER_SECOND_BUNKER = 0.2
WEBDAV_REMOTE = "webdav:/Tracker/annotation_frames"
DOWNLOAD_HISTORY_FILE = "downloaded_videos.json"