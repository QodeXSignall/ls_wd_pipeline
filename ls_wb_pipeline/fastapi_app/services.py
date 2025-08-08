import json
import os
import shutil
import tempfile
import threading
import time
import uuid
import zipfile
from pathlib import Path
from typing import Optional, Dict, Any, Iterator

from ls_wb_pipeline import functions, build_dataset_cls
from ls_wb_pipeline.logger import logger
from ls_wb_pipeline import settings


def analyze_dataset_service():
    result = build_dataset_cls.analyze_classification_dataset(settings.DATASET_PATH)
    return {"status": "analyzed", "result": result}


def cleanup_frames_tasks(tasks, dry_run: bool = False, save_annotated: bool = True):
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
    report = {
        "status": "dataset built",
        "dry_run": dry_run,
        "before": None,
        "after": None
    }
    report["before"] = analyze_dataset_service()

    all_tasks = functions.get_all_tasks()
    build_dataset_cls.build_classification_dataset(all_tasks, train_ratio=train_ratio, test_ratio=test_ratio, val_ratio=val_ratio)

    if del_unannotated:
        delete_report = cleanup_frames_tasks(all_tasks, dry_run=dry_run, save_annotated=True)
        report["delete_report"] = delete_report
    after = analyze_dataset_service()
    report["after"] = after
    return report


def load_new_frames(max_frames: int = 300, only_cargo_type: str = None, fps: float = None, video_name: str = None):
    return functions.main_process_new_frames(max_frames=max_frames, only_cargo_type=only_cargo_type, fps=fps, video_name=video_name)


# ==== ZIP background preparation ====

_ARCHIVE_DIR = Path(getattr(settings, "DATASET_ARCHIVE_DIR",
                            Path(settings.DATASET_PATH).parent / "dataset_archives"))
_ARCHIVE_DIR.mkdir(parents=True, exist_ok=True)
_ARCHIVE_PATH = _ARCHIVE_DIR / "dataset.zip"
_META_PATH = _ARCHIVE_DIR / "dataset.zip.meta.json"
_LOCK_PATH = _ARCHIVE_DIR / ".dataset_zip.lock"

_TASKS: Dict[str, Dict[str, Any]] = {}
_TASKS_LOCK = threading.Lock()


def _count_files(root: Path) -> int:
    total = 0
    for dp, _, fns in os.walk(root):
        total += len(fns)
    return total


def _latest_mtime(root: Path) -> float:
    latest = 0.0
    for dp, _, fns in os.walk(root):
        for f in fns:
            p = os.path.join(dp, f)
            try:
                m = os.path.getmtime(p)
                if m > latest:
                    latest = m
            except FileNotFoundError:
                continue
    return latest


def _need_rebuild(dataset_dir: Path, archive_path: Path, meta_path: Path) -> bool:
    if not archive_path.exists() or not meta_path.exists():
        return True
    try:
        meta = json.loads(meta_path.read_text() or "{}")
    except Exception:
        return True
    latest = _latest_mtime(dataset_dir)
    return float(meta.get("latest_mtime", 0)) < latest


def _write_meta(dataset_dir: Path):
    meta = {"latest_mtime": _latest_mtime(dataset_dir), "built_at": time.time()}
    _META_PATH.write_text(json.dumps(meta, ensure_ascii=False))


def _acquire_lock() -> bool:
    try:
        fd = os.open(_LOCK_PATH, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        os.close(fd)
        return True
    except FileExistsError:
        return False


def _release_lock():
    try:
        _LOCK_PATH.unlink()
    except FileNotFoundError:
        pass


def _zip_build_worker(task_id: str, dataset_dir: Path):
    with _TASKS_LOCK:
        task = _TASKS.get(task_id)
        if not task:
            return
        task.update({"status": "running", "progress": 0, "detail": "Initializing"})

    total_files = _count_files(dataset_dir)
    if total_files == 0:
        with _TASKS_LOCK:
            _TASKS[task_id].update({"status": "error", "error": "Dataset is empty"})
        return

    waited = 0
    while not _acquire_lock():
        time.sleep(0.5)
        waited += 0.5
        if waited > 300:
            with _TASKS_LOCK:
                _TASKS[task_id].update({"status": "error", "error": "Timeout waiting for another build"})
            return

    try:
        if not _need_rebuild(dataset_dir, _ARCHIVE_PATH, _META_PATH):
            with _TASKS_LOCK:
                _TASKS[task_id].update({
                    "status": "done",
                    "progress": 100,
                    "result": {"archive_path": str(_ARCHIVE_PATH)}
                })
            return

        tmp_path = _ARCHIVE_PATH.with_suffix(".zip.tmp")
        if tmp_path.exists():
            try:
                tmp_path.unlink()
            except FileNotFoundError:
                pass

        written = 0
        with zipfile.ZipFile(tmp_path, "w", compression=zipfile.ZIP_STORED) as zf:
            root = Path(dataset_dir)
            for dp, _, fns in os.walk(root):
                for f in fns:
                    full = Path(dp) / f
                    arcname = str(full.relative_to(root))
                    try:
                        zf.write(full, arcname)
                    except FileNotFoundError:
                        continue
                    written += 1
                    if written % 50 == 0 or written == total_files:
                        with _TASKS_LOCK:
                            _TASKS[task_id].update({
                                "status": "running",
                                "progress": int(written * 100 / max(1, total_files)),
                                "detail": f"Packed {written}/{total_files} files"
                            })

        tmp_path.replace(_ARCHIVE_PATH)
        _write_meta(dataset_dir)

        with _TASKS_LOCK:
            _TASKS[task_id].update({
                "status": "done",
                "progress": 100,
                "result": {"archive_path": str(_ARCHIVE_PATH)}
            })
    except Exception as e:
        logger.exception("ZIP build failed")
        with _TASKS_LOCK:
            _TASKS[task_id].update({"status": "error", "error": str(e)})
    finally:
        _release_lock()


def prepare_dataset_start() -> Dict[str, Any]:
    dataset_dir = Path(settings.DATASET_PATH)
    if not dataset_dir.exists():
        raise FileNotFoundError("Датасет ещё не создан.")

    task_id = uuid.uuid4().hex
    with _TASKS_LOCK:
        _TASKS[task_id] = {"status": "queued", "progress": 0, "created_at": time.time()}

    t = threading.Thread(target=_zip_build_worker, args=(task_id, dataset_dir), daemon=True)
    t.start()

    return {"task_id": task_id, "status": "queued"}


def prepare_dataset_status(task_id: str) -> Optional[Dict[str, Any]]:
    with _TASKS_LOCK:
        return _TASKS.get(task_id)


def get_ready_zip_path() -> str:
    if not _ARCHIVE_PATH.exists():
        raise FileNotFoundError("Архив ещё не готов. Сначала вызовите /prepare-dataset и дождитесь статуса done.")
    return str(_ARCHIVE_PATH)


# ===== Streaming download helpers =====

CHUNK_SIZE = int(getattr(settings, "DOWNLOAD_CHUNK_SIZE", 1024 * 1024))  # 1 MiB default

def get_download_headers_and_path():
    path = get_ready_zip_path()
    stat = os.stat(path)
    headers = {
        "Content-Length": str(stat.st_size),
        "Content-Disposition": 'attachment; filename="dataset.zip"',
        "Accept-Ranges": "bytes",
        "Cache-Control": "no-store",
    }
    return path, stat.st_size, headers


def iter_file(path: str, start: int = 0, end: Optional[int] = None) -> Iterator[bytes]:
    with open(path, "rb") as f:
        f.seek(start)
        bytes_left = None if end is None else (end - start + 1)
        while True:
            to_read = CHUNK_SIZE if bytes_left is None else min(CHUNK_SIZE, bytes_left)
            data = f.read(to_read)
            if not data:
                break
            yield data
            if bytes_left is not None:
                bytes_left -= len(data)
                if bytes_left <= 0:
                    break


def parse_range_header(range_header: str, total_size: int):
    try:
        unit, rng = range_header.split("=", 1)
        if unit.strip().lower() != "bytes":
            return None, None
        start_str, end_str = rng.split("-", 1)
        if start_str == "":
            length = int(end_str)
            start = max(0, total_size - length)
            end = total_size - 1
        else:
            start = int(start_str)
            end = total_size - 1 if end_str == "" else int(end_str)
        if start < 0 or end < start or start >= total_size:
            return None, None
        return start, min(end, total_size - 1)
    except Exception:
        return None, None


def build_range_headers(start: int, end: int, total: int):
    return {
        "Content-Range": f"bytes {start}-{end}/{total}",
        "Content-Length": str(end - start + 1),
        "Accept-Ranges": "bytes",
        "Content-Disposition": 'attachment; filename="dataset.zip"',
        "Cache-Control": "no-store",
    }


# ===== Legacy endpoints kept for compatibility =====

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
        try:
            if _ARCHIVE_PATH.exists():
                _ARCHIVE_PATH.unlink()
            if _META_PATH.exists():
                _META_PATH.unlink()
        except Exception:
            pass
        return {"status": "Датасет успешно удален", "path": settings.DATASET_PATH}
    else:
        return {"status": "Датасет не найден", "path": settings.DATASET_PATH}


def clean_downloaded_list():
    with open(settings.DOWNLOAD_HISTORY_FILE, "w") as f:
        json.dump([], f)
    return {"status": "cleaned", "path": settings.DOWNLOAD_HISTORY_FILE}
