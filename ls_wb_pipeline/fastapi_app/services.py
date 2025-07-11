from ls_wb_pipeline import functions, build_dataset
from ls_wb_pipeline import settings
import configparser
import tempfile
import shutil
import json
import io
import os


def analyze_dataset_service():
    result = build_dataset.analyze_dataset()
    return {"status": "analyzed", "result": result}


def build_dataset_and_cleanup(json_bytes: bytes, dry_run: bool = True, train_ratio=0.8, test_ratio=0.1, val_ratio=0.1):
    before = analyze_dataset_service()

    # Читаем JSON из байтов
    json_data = json.load(io.BytesIO(json_bytes))
    build_dataset.main_from_data(json_data, train_ratio=train_ratio, test_ratio=test_ratio, val_ratio=val_ratio)  # нужна будет версия main, принимающая уже загруженные данные

    functions.clean_cloud_files_from_data(json_data, dry_run=dry_run)  # аналогично
    functions.delete_ls_tasks(dry_run=dry_run)

    after = analyze_dataset_service()
    return {
        "status": "dataset built",
        "dry_run": dry_run,
        "before": before,
        "after": after
    }


def load_new_frames(max_frames: int = 300, only_cargo_type: str = None):
    functions.main_process_new_frames(max_frames=max_frames, only_cargo_type=only_cargo_type)
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


def delete_dataset_service():
    if os.path.exists(settings.DATASET_PATH):
        shutil.rmtree(settings.DATASET_PATH)
        return {"status": "deleted", "path": settings.DATASET_PATH}
    else:
        return {"status": "not found", "path": settings.DATASET_PATH}