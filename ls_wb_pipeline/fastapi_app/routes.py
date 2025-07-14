from fastapi import APIRouter, UploadFile, File, Query
from fastapi.responses import FileResponse, JSONResponse
from ls_wb_pipeline.fastapi_app.services import (
    analyze_dataset_service,
    build_dataset_and_cleanup,
    get_zip_dataset,
    load_new_frames,
    delete_dataset_service
)

router = APIRouter()




@router.post("/build-dataset", tags=["dataset"])
def enrich_dataset(
    file: UploadFile = File(...),
    dry_run: bool = Query(False, description="Построить датасет без удаления неразмеченных кадров"),
    train_ratio: float = Query(0.8, description="Тренировочная часть"),
    val_ratio: float = Query(0.1, description="Валидационная часть"),
    test_ratio: float = Query(0.1, description="Тестовая часть")):
    contents = file.file.read()
    return build_dataset_and_cleanup(
        contents, dry_run=dry_run, train_ratio=train_ratio, test_ratio=test_ratio, val_ratio=val_ratio)

@router.get("/analyze-dataset", tags=["dataset"])
def analyze_dataset():
    return analyze_dataset_service()

@router.get("/download-dataset", tags=["dataset"])
def download_dataset():
    try:
        archive_path = get_zip_dataset()
        return FileResponse(
            archive_path,
            media_type="application/zip",
            filename="dataset.zip"
        )
    except FileNotFoundError as e:
        return {"error": str(e)}

@router.delete("/del-dataset", tags=["dataset"])
def delete_dataset():
    return delete_dataset_service()

@router.post("/load-frames", tags=["frames"])
def load_frames(max_frames: int = Query(300, description="Максимум кадров"),
                only_cargo_type: str = Query(default=None, description="Вид контейнера (bunker/euro). Если не указан, качает все"),):
    return load_new_frames(max_frames=max_frames, only_cargo_type=only_cargo_type)