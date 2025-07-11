from fastapi import APIRouter, UploadFile, File, Query
from fastapi.responses import FileResponse
from ls_wb_pipeline.fastapi_app.services import (
    analyze_dataset,
    build_dataset_and_cleanup,
    get_zip_dataset,
    load_new_frames,
    get_config,
    set_config_from_dict,
    upload_config_file,
    get_config_path
)

router = APIRouter()

@router.get("/config")
def get_config():
    return get_config()


@router.post("/config")
def set_config(config: dict):
    return set_config_from_dict(config)


@router.post("/config/upload")
def upload_config(file: UploadFile = File(...)):
    return upload_config_file(file)


@router.get("/config/download")
def download_config():
    path = get_config_path()
    return FileResponse(path, filename="config.cfg")

@router.get("/analyze")
def analyze():
    return analyze_dataset()

@router.post("/build")
def build_dataset(
    file: UploadFile = File(...),
    dry_run: bool = Query(False, description="Построить датасет без удаления неразмеченных кадров"),
    max_frames: int = Query(200, description="Максимум кадров")
):
    contents = file.file.read()
    return build_dataset_and_cleanup(contents, dry_run=dry_run, max_frames=max_frames)

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