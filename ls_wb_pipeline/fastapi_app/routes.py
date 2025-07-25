from fastapi import APIRouter, Query
from fastapi.responses import FileResponse
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

@router.get("/download-dataset", tags=["dataset"])
def download_dataset():
    try:
        archive_path = services.get_zip_dataset()
        return FileResponse(
            archive_path,
            media_type="application/zip",
            filename="dataset.zip"
        )
    except FileNotFoundError as e:
        return {"error": str(e)}

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
