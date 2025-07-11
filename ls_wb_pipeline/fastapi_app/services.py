from ls_wb_pipeline import functions, build_dataset
import configparser
import tempfile
import shutil
import json
import io
import os


CONFIG_PATH = "config.cfg"


def analyze_dataset():
    result = build_dataset.analyze_dataset()
    return {"status": "analyzed", "result": result}


def build_dataset_and_cleanup(json_bytes: bytes, dry_run: bool = True):
    before = analyze_dataset()

    # Читаем JSON из байтов
    json_data = json.load(io.BytesIO(json_bytes))
    build_dataset.main_from_data(json_data)  # нужна будет версия main, принимающая уже загруженные данные

    functions.clean_cloud_files_from_data(json_data, dry_run=dry_run)  # аналогично
    functions.delete_ls_tasks(dry_run=dry_run)

    after = analyze_dataset()
    return {
        "status": "dataset built",
        "dry_run": dry_run,
        "before": before,
        "after": after
    }


def load_new_frames(max_frames: int = 300, cargo_type: str = None):
    functions.main_process_new_frames(max_frames=max_frames, cargo_type=cargo_type)
    return {"status": "frames loaded", "max_frames": max_frames}


def get_zip_dataset():
    dataset_dir = "./dataset_yolo"

    if not os.path.exists(dataset_dir):
        return {"error": "Датасет ещё не создан."}

    # Создаём временный ZIP-файл
    tmp_dir = tempfile.mkdtemp()
    archive_path = os.path.join(tmp_dir, "dataset_yolo.zip")
    shutil.make_archive(archive_path.replace(".zip", ""), "zip", dataset_dir)

    return archive_path

def get_config_service():
    config = configparser.ConfigParser()
    config.read(CONFIG_PATH)
    return {section: dict(config[section]) for section in config.sections()}

def set_config_from_dict(new_config: dict):
    config = configparser.ConfigParser()
    for section, values in new_config.items():
        config[section] = values
    with open(CONFIG_PATH, "w") as f:
        config.write(f)
    return {"status": "config updated"}


def upload_config_file(file):
    with open(CONFIG_PATH, "wb") as f:
        f.write(file.file.read())
    return {"status": "uploaded"}

def get_config_path():
    return CONFIG_PATH

