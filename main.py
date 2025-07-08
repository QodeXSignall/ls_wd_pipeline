from ls_wb_pipeline import build_yolo_dataset as build_dataset
from ls_wb_pipeline import functions
import argparse

def build_dataset_and_cleanup(json_path, dry_run=True):
    build_dataset.main(json_path)
    functions.clean_cloud_files(json_path, dry_run=dry_run)
    functions.delete_ls_tasks(dry_run=dry_run)
    print(f"Завершено: датасет собран, мусор удалён (dry_run={dry_run})")
    build_dataset.analyze_full_dataset()
    #functions.main_process_new_frames()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Сборка YOLO датасета из Label Studio JSON")
    parser.add_argument("--json", required=True, help="Путь до экспортированного JSON-файла из Label Studio")
    parser.add_argument("--dry-run", action="store_true", help="Запуск без удаления — только просмотр действий")
    args = parser.parse_args()
    build_dataset_and_cleanup(args.json, args.dry_run)
