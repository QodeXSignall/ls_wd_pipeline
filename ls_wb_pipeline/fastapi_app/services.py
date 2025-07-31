from ls_wb_pipeline import functions, build_dataset_cls
from ls_wb_pipeline.logger import logger
from ls_wb_pipeline import settings
import tempfile
import shutil
import json
import io
import os


def analyze_dataset_service():
    result = build_dataset_cls.analyze_classification_dataset(settings.DATASET_PATH)
    return {"status": "analyzed", "result": result}


def cleanup_frames_tasks(tasks, dry_run:bool = False, save_annotated: bool = True):
    logger.info("Удаление задач labelstudio")
    deleted_tasks, saved_amount = functions.delete_ls_tasks(tasks=tasks, dry_run=dry_run, save_annotated=save_annotated)
    logger.info("Удаление файлов с облака")
    deleted_files_report = functions.clean_cloud_files_from_tasks(
        tasks=tasks, dry_run=dry_run, save_annotated=save_annotated)
    logger.info("Удаление завершено")
    return {"status": "cleaned", "result":
        {"files": {"deleted_amount": deleted_files_report["deleted_amount"],
                   "saved_amount": deleted_files_report["saved"],
                   "deleted": deleted_files_report["deleted"]},
         "tasks": {"deleted": len(deleted_tasks)},
                    "saved": saved_amount},
            "dry_run": dry_run}

def enrich_dataset_and_cleanup(dry_run: bool = True, train_ratio=0.8, test_ratio=0.1, val_ratio=0.1,
                               del_unannotated: bool = True):
    report =  {
        "status": "dataset built",
        "dry_run": dry_run,
        "before": None,
        "after": None
    }
    report["before"] = analyze_dataset_service()

    all_tasks = functions.get_all_tasks()
    build_dataset_cls.build_classification_dataset(all_tasks, train_ratio=train_ratio, test_ratio=test_ratio, val_ratio=val_ratio)  # нужна будет версия main, принимающая уже загруженные данные

    if del_unannotated:
        delete_report = cleanup_frames_tasks(all_tasks, dry_run=dry_run, save_annotated=True)
        report["delete_report"] = delete_report
    after = analyze_dataset_service()
    report["after"] = after
    return report


def load_new_frames(max_frames: int = 300, only_cargo_type: str = None, fps: float = None, video_name: str = None):
    return functions.main_process_new_frames(max_frames=max_frames, only_cargo_type=only_cargo_type, fps=fps, video_name=video_name)


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

def clean_downloaded_list():
    with open(settings.DOWNLOAD_HISTORY_FILE, "w") as f:
        json.dump([], f)
    return {"status": "cleaned", "path": settings.DOWNLOAD_HISTORY_FILE}
