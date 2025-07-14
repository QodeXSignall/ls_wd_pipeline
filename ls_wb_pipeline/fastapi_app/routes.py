from fastapi import APIRouter, UploadFile, File, Query
from fastapi.responses import FileResponse
from ls_wb_pipeline import settings
from ls_wb_pipeline.fastapi_app import services

router = APIRouter()


@router.post("/enrich-dataset", tags=["dataset"])
def enrich_dataset(
    file: UploadFile = File(...),
    dry_run: bool = Query(False, description="Построить датасет без удаления неразмеченных кадров"),
    train_ratio: float = Query(0.8, description="Тренировочная часть"),
    val_ratio: float = Query(0.1, description="Валидационная часть"),
    test_ratio: float = Query(0.1, description="Тестовая часть")):
    contents = file.file.read()
    return services.enrich_dataset_and_cleanup(
        contents, dry_run=dry_run, train_ratio=train_ratio, test_ratio=test_ratio, val_ratio=val_ratio)

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
                                             f"По умолчанию, {settings.FRAMES_PER_SECOND_EURO}fps euro, "
                                             f"{settings.FRAMES_PER_SECOND_BUNKER}fps bunker"),
                video_name: str = Query(default=None, description="Скачать конкретное видео (можно скачать уже скачанное ранее)")):
    return services.load_new_frames(max_frames=max_frames, only_cargo_type=only_cargo_type, fps=fps, video_name=video_name)

@router.delete("/del-frames", tags=["frames"])
def delete_frames(
        dry_run: bool = Query(False, description="Имитация удаления"),
        save_annotated: bool = Query(default=True,
                                     description="Сохранить уже анотированые кадры?"),):
    return services.cleanup_frames_tasks(dry_run=dry_run, save_annotated=save_annotated)

@router.delete("/clean-download-history", tags=["frames"])
def delete_frames():
    return services.clean_downloaded_list()
