import os
import json
import argparse
from collections import Counter
from urllib.parse import unquote
from sklearn.model_selection import train_test_split
from ls_wb_pipeline import functions
import shutil

# ==== НАСТРОЙКИ (можно менять внутри скрипта) ====
SOURCE_IMAGE_DIR = functions.MOUNTED_PATH
OUTPUT_DIR = "./dataset_yolo"
SPLIT_RATIO = (0.8, 0.1, 0.1)  # train, val, test

def main(json_path):
    # 1. Загрузка уже существующих изображений (чтобы избежать дубликатов)
    existing_images = set()
    for split in ("train", "val", "test"):
        img_dir = os.path.join(OUTPUT_DIR, "images", split)
        if os.path.exists(img_dir):
            for fname in os.listdir(img_dir):
                if fname.lower().endswith(".jpg"):
                    existing_images.add(fname)

    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    entries = []
    class_names = set()

    for task in data:
        anns = task.get("annotations")
        if not anns or not isinstance(anns, list):
            continue
        first_ann = anns[0]
        results = first_ann.get("result", [])
        if not results:
            continue
        try:
            class_name = results[0]["value"]["choices"][0]
            image_url = task["data"]["image"]
            image_name = os.path.basename(unquote(image_url))
            if image_name in existing_images:
                continue  # ❗️ Пропускаем уже обработанные
            class_names.add(class_name)
            entries.append({
                "image": image_name,
                "class": class_name
            })
        except Exception:
            continue

    if not entries:
        print("Не найдено валидных размеченных задач.")
        return

    # Загрузка уже существующих классов (если есть)
    existing_classes = []
    classes_path = os.path.join(OUTPUT_DIR, "classes.txt")

    if os.path.exists(classes_path):
        with open(classes_path, "r", encoding="utf-8") as f:
            existing_classes = [line.strip() for line in f if line.strip()]

    # Объединяем старые и новые классы, убираем дубли
    all_classes = list(dict.fromkeys(existing_classes + sorted(class_names)))  # сохраняем порядок

    # Гарантируем, что OUTPUT_DIR существует
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Сохраняем объединённый список
    with open(classes_path, "w", encoding="utf-8") as f:
        for name in all_classes:
            f.write(f"{name}\n")

    # ✅ Создаём class_to_index на основе all_classes
    class_to_index = {name: i for i, name in enumerate(all_classes)}

    # Создание папок
    splits = ["train", "val", "test"]
    for split in splits:
        os.makedirs(os.path.join(OUTPUT_DIR, "images", split), exist_ok=True)
        os.makedirs(os.path.join(OUTPUT_DIR, "labels", split), exist_ok=True)

    # Разделение на train/val/test
    if len(entries) < 3:
        split_data = {"train": entries, "val": [], "test": []}
    else:
        train_val, test = train_test_split(entries, test_size=SPLIT_RATIO[2], random_state=42)
        train, val = train_test_split(train_val, test_size=SPLIT_RATIO[1]/(SPLIT_RATIO[0]+SPLIT_RATIO[1]), random_state=42)
        split_data = {"train": train, "val": val, "test": test}


    # Копирование и генерация .txt аннотаций
    for split, items in split_data.items():
        for item in items:
            image_name = item["image"]
            class_id = class_to_index[item["class"]]
            label_file = os.path.join(OUTPUT_DIR, "labels", split, image_name.replace(".jpg", ".txt"))
            image_src = os.path.join(SOURCE_IMAGE_DIR, image_name)
            image_dst = os.path.join(OUTPUT_DIR, "images", split, image_name)

            # пишем класс в YOLO-формате
            with open(label_file, "w") as f:
                f.write(f"{class_id}\n")

            # копируем изображение
            if os.path.exists(image_src):
                shutil.copy(image_src, image_dst)

    # Новый summary по индексу классов
    summary = Counter(class_to_index[e["class"]] for e in entries)

    print(f"\nДатасет собран. {OUTPUT_DIR}")
    total = sum(summary.values())
    print("\nРаспределение классов в заданном JSON:")
    for class_id, class_name in enumerate(all_classes):
        count = summary[class_id]
        percent = (count / total) * 100 if total else 0
        print(f"{class_name:25} — {count:3} изображений ({percent:.1f}%)")


def analyze_full_dataset(dataset_path=OUTPUT_DIR):
    labels_root = os.path.join(dataset_path, "labels")
    classes_file = os.path.join(dataset_path, "classes.txt")

    if not os.path.exists(labels_root) or not os.path.exists(classes_file):
        print("❌ Не найден labels/ или classes.txt — датасет ещё не создан?")
        return

    # Загрузка названий классов
    with open(classes_file, "r", encoding="utf-8") as f:
        classes = [line.strip() for line in f if line.strip()]

    # Счётчики по каждому сплиту
    split_counters = {
        "train": Counter(),
        "val": Counter(),
        "test": Counter()
    }

    for split in split_counters:
        label_dir = os.path.join(labels_root, split)
        if not os.path.exists(label_dir):
            continue
        for fname in os.listdir(label_dir):
            if fname.endswith(".txt"):
                fpath = os.path.join(label_dir, fname)
                with open(fpath, "r", encoding="utf-8") as f:
                    line = f.readline().strip()
                    if line.isdigit():
                        class_id = int(line)
                        split_counters[split][class_id] += 1

    total = sum(sum(c.values()) for c in split_counters.values())
    print("\n📦 Общая картина по всем размеченным классам (всего: {}):".format(total))
    avg = total / len(classes) if classes else 0

    print(f"{'ID':<3} {'Класс':<25} {'Train':>6} {'Val':>6} {'Test':>6} {'Total':>6} {'%':>6}")
    print("-" * 60)
    for class_id, class_name in enumerate(classes):
        tr = split_counters["train"][class_id]
        va = split_counters["val"][class_id]
        te = split_counters["test"][class_id]
        total_cls = tr + va + te
        percent = (total_cls / total) * 100 if total else 0
        print(f"{class_id:<3} {class_name:<25} {tr:6} {va:6} {te:6} {total_cls:6} {percent:5.1f}%")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Сборка YOLO датасета из Label Studio JSON")
    parser.add_argument("--json", required=True, help="Путь до экспортированного JSON-файла из Label Studio")
    args = parser.parse_args()
    main(args.json)
