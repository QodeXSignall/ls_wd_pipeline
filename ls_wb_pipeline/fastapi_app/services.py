from ls_wb_pipeline import functions, build_dataset
from ls_wb_pipeline import settings
import tempfile
import shutil
import json
import io
import os


def analyze_dataset_service():
    result = build_dataset.analyze_dataset()
    return {"status": "analyzed", "result": result}


def cleanup_frames_tasks(json_data: bytes = None, dry_run:bool = False):
    all_tasks, deleted, saved = functions.delete_ls_tasks(dry_run=dry_run)
    if not json_data:
        json_data = all_tasks
    deleted_files_report = functions.clean_cloud_files_from_data(json_data=json_data, dry_run=dry_run)
    return {"status": "cleaned", "result":
        {"files": deleted_files_report,
         "tasks": {"deleted": len(deleted)},
                    "saved": len(saved)}}

def enrich_dataset_and_cleanup(json_bytes: bytes, dry_run: bool = True, train_ratio=0.8, test_ratio=0.1, val_ratio=0.1):
    before = analyze_dataset_service()

    # Читаем JSON из байтов
    json_data = json.load(io.BytesIO(json_bytes))
    build_dataset.main_from_data(json_data, train_ratio=train_ratio, test_ratio=test_ratio, val_ratio=val_ratio)  # нужна будет версия main, принимающая уже загруженные данные

    cleanup_frames_tasks(json_data=json_data, dry_run=dry_run)

    after = analyze_dataset_service()
    return {
        "status": "dataset built",
        "dry_run": dry_run,
        "before": before,
        "after": after
    }


def load_new_frames(max_frames: int = 300, only_cargo_type: str = None, fps: float = None):
    return functions.main_process_new_frames(max_frames=max_frames, only_cargo_type=only_cargo_type, fps=fps)


def get_zip_dataset():
    dataset_dir = settings.DATASET_PATH
    if not os.path.exists(dataset_dir):
        raise FileNotFoundError("Датасет ещё не создан.")

    tmp_dir = tempfile.mkdtemp()
    archive_path = os.path.join(tmp_dir, "dataset.zip")
    shutil.make_archive(archive_path[:-4], "zip", dataset_dir)

    return archive_path

def delete_dataset_service():
    if os.path.exists(settings.DATASET_PATH):
        shutil.rmtree(settings.DATASET_PATH)
        return {"status": "Датасет успешно удален", "path": settings.DATASET_PATH}
    else:
        return {"status": "Датасет не найден", "path": settings.DATASET_PATH}