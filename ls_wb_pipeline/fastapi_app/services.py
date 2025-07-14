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


def cleanup_frames_tasks(json_data: bytes = None, dry_run:bool = False, save_annotated: bool = True):
    all_tasks, deleted_tasks, saved_amount = functions.delete_ls_tasks(dry_run=dry_run, save_annotated=save_annotated)
    if json_data:
        all_tasks = json.loads(json_data)
    deleted_files_report = functions.clean_cloud_files_from_tasks(
        tasks=all_tasks, dry_run=dry_run, save_annotated=save_annotated)
    return {"status": "cleaned", "result":
        {"files": {"deleted": deleted_files_report["deleted"],
                   "saved": deleted_files_report["saved"]},
         "tasks": {"deleted": len(deleted_tasks)},
                    "saved": saved_amount},
            "dry_run": dry_run}

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