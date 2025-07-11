from ls_wb_pipeline import build_dataset as build_dataset
from ls_wb_pipeline import functions
import argparse

def build_dataset_and_cleanup(json_path, dry_run=True, max_frames=300):
    print("Исходный датасет:")
    build_dataset.analyze_dataset()
    build_dataset.main_from_path(json_path)
    functions.clean_cloud_files_from_path(json_path, dry_run=dry_run)
    functions.delete_ls_tasks(dry_run=dry_run)
    print(f"Завершено: датасет собран, мусор удалён (dry_run={dry_run})")
    print("Конечный датасет:")
    build_dataset.analyze_dataset()
    functions.main_process_new_frames(max_frames=max_frames)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Сборка YOLO датасета из Label Studio JSON")
    parser.add_argument("--json", required=True, help="Путь до экспортированного JSON-файла из Label Studio")
    parser.add_argument("--max_frames", type=int, default=300, help="Максимальное количество фреймов, которое нужно подготовить для разметки")
    parser.add_argument("--dry-run", action="store_true", help="Запуск без удаления — только просмотр действий")
    args = parser.parse_args()
    build_dataset_and_cleanup(args.json, args.dry_run, args.max_frames)
