from fastapi import APIRouter, UploadFile, File, Query
from fastapi.responses import FileResponse
from ls_wb_pipeline.fastapi_app.services import (
    analyze_dataset,
    build_dataset_and_cleanup,
    get_zip_dataset,
    load_new_frames,
)

router = APIRouter()


@router.get("/analyze")
def analyze():
    return analyze_dataset()

@router.post("/build")
def build_dataset(
    file: UploadFile = File(...),
    dry_run: bool = Query(False, description="Построить датасет без удаления неразмеченных кадров"),
    train_ratio: float = Query(0.8, description="Тренировочная часть"),
    val_ratio: int = Query(0.1, description="Валидационная часть"),
    test_ratio: int = Query(0.1, description="Тестовая часть")):
    contents = file.file.read()
    return build_dataset_and_cleanup(
        contents, dry_run=dry_run, train_ratio=train_ratio, test_ratio=test_ratio, val_ratio=val_ratio)

@router.post("/load-frames")
def load_frames(max_frames: int = Query(300, description="Максимум кадров"),
                cargo_type: str = Query(default=None, description="Вид контейнера (bunker/euro). Если не указан, качает все"),):
    return load_new_frames(max_frames=max_frames, cargo_type=cargo_type)

@router.get("/download-dataset")
def download_dataset():
    archive_path = get_zip_dataset()
    return FileResponse(
        archive_path,
        media_type="application/zip",
        filename="dataset_yolo.zip"
    )