from fastapi import APIRouter, Query, HTTPException, Request
from fastapi.responses import StreamingResponse
from ls_wb_pipeline import settings
from ls_wb_pipeline.fastapi_app import services

router = APIRouter()


@router.post("/build_dataset", tags=["dataset"])
def build_dataset(
    train_ratio: float = Query(0.8, description="Тренировочная часть"),
    val_ratio: float = Query(0.1, description="Валидационная часть"),
    test_ratio: float = Query(0.1, description="Тестовая часть"),
    del_unannotated: bool = Query(True, description="Удалить неразмеченные кадры"),
    dry_run: bool = Query(default=False, description="Имитация удаления")):
    return services.enrich_dataset_and_cleanup(dry_run=dry_run,
        del_unannotated=del_unannotated, train_ratio=train_ratio, test_ratio=test_ratio, val_ratio=val_ratio)


@router.get("/analyze-dataset", tags=["dataset"])
def analyze_dataset():
    return services.analyze_dataset_service()


@router.post("/prepare-dataset", tags=["dataset"])
def prepare_dataset():
    try:
        task = services.prepare_dataset_start()
        return task
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to start prepare task: {e}")


@router.get("/prepare-dataset/{task_id}", tags=["dataset"])
def prepare_dataset_status(task_id: str):
    st = services.prepare_dataset_status(task_id)
    if st is None:
        raise HTTPException(status_code=404, detail="Task not found")
    return st


@router.get("/download-dataset", tags=["dataset"])
def download_dataset(request: Request):
    try:
        path, total_size, headers = services.get_download_headers_and_path()
        range_header = request.headers.get('range')
        if range_header:
            start, end = services.parse_range_header(range_header, total_size)
            if start is None:
                raise HTTPException(status_code=416, detail="Invalid Range")
            headers.update(services.build_range_headers(start, end, total_size))
            return StreamingResponse(
                services.iter_file(path, start=start, end=end),
                status_code=206,
                media_type="application/zip",
                headers=headers,
            )
        return StreamingResponse(
            services.iter_file(path),
            media_type="application/zip",
            headers=headers,
        )
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to stream archive: {e}")


@router.delete("/del-dataset", tags=["dataset"])
def delete_dataset():
    return services.delete_dataset_service()


@router.post("/load-frames", tags=["frames"])
def load_frames(max_frames: int = Query(300, description="Максимум кадров"),
                only_cargo_type: str = Query(default=None, description="Вид контейнера (bunker/euro). По умолчанию, качает все"),
                fps: float = Query(default=None,
                                 description=f"Количество кадров в секунду. "
                                             f"По умолчанию: {settings.FRAMES_PER_SECOND_EURO}fps euro, "
                                             f"{settings.FRAMES_PER_SECOND_BUNKER}fps bunker"),
                video_name: str = Query(default=None, description="Скачать конкретное видео (можно скачать уже скачанное ранее)")):
    return services.load_new_frames(max_frames=max_frames, only_cargo_type=only_cargo_type, fps=fps, video_name=video_name)


@router.delete("/del-frames", tags=["frames"])
def delete_frames(
        dry_run: bool = Query(False, description="Имитация удаления"),
        save_annotated: bool = Query(default=True,
                                     description="Сохранить уже анотированые кадры?"),):
    tasks = services.functions.get_all_tasks()
    return services.cleanup_frames_tasks(tasks=tasks, dry_run=dry_run, save_annotated=save_annotated)


@router.delete("/clean-download-history", tags=["service"])
def clean_download_history():
    return services.clean_downloaded_list()
