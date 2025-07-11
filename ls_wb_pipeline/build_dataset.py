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

def main_from_data(data):
    # Загрузка уже размеченных изображений по .txt
    existing_labels = set()
    for split in ("train", "val", "test"):
        label_dir = os.path.join(OUTPUT_DIR, "labels", split)
        if os.path.exists(label_dir):
            for fname in os.listdir(label_dir):
                if fname.lower().endswith(".txt"):
                    image_name = fname.replace(".txt", ".jpg")
                    existing_labels.add(image_name)

    existing_images = set()
    for split in ("train", "val", "test"):
        img_dir = os.path.join(OUTPUT_DIR, "images", split)
        if os.path.exists(img_dir):
            for fname in os.listdir(img_dir):
                if fname.lower().endswith(".jpg"):
                    existing_images.add(fname)

    entries = []
    class_names = set()
    full_summary = Counter()

    # Первый проход — для полной статистики
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
            full_summary[class_name] += 1
        except Exception:
            continue

    # Второй проход — для новых изображений
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
            if image_name in existing_images or image_name in existing_labels:
                continue  # ❗️ Пропускаем уже размеченные

            class_names.add(class_name)
            entries.append({
                "image": image_name,
                "class": class_name
            })
        except Exception:
            continue

    # Распределение классов в JSON
    print(f"\nРаспределение классов в заданном JSON:")
    total_full = sum(full_summary.values())
    for cls in sorted(full_summary.keys()):
        count = full_summary[cls]
        percent = (count / total_full) * 100 if total_full else 0
        print(f"{cls:25} — {count:3} изображений ({percent:.1f}%)")

    if not entries:
        print("Не найдено новых изображений для добавления.")
        return


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

    print(f"\nДатасет собран. {OUTPUT_DIR}")


def main_from_path(json_path):
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    main_from_data(data)


def analyze_dataset(dataset_path=OUTPUT_DIR):
    labels_root = os.path.join(dataset_path, "labels")
    classes_file = os.path.join(dataset_path, "classes.txt")

    if not os.path.exists(labels_root) or not os.path.exists(classes_file):
        return {"error": "labels/ или classes.txt не найдены — датасет ещё не создан"}

    with open(classes_file, "r", encoding="utf-8") as f:
        classes = [line.strip() for line in f if line.strip()]

    split_counters = {"train": Counter(), "val": Counter(), "test": Counter()}

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
    result = {
        "total": total,
        "classes": []
    }

    for class_id, class_name in enumerate(classes):
        tr = split_counters["train"][class_id]
        va = split_counters["val"][class_id]
        te = split_counters["test"][class_id]
        total_cls = tr + va + te
        percent = (total_cls / total) * 100 if total else 0
        result["classes"].append({
            "id": class_id,
            "name": class_name,
            "train": tr,
            "val": va,
            "test": te,
            "total": total_cls,
            "percent": round(percent, 1)
        })

    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Сборка YOLO датасета из Label Studio JSON")
    parser.add_argument("--json", required=True, help="Путь до экспортированного JSON-файла из Label Studio")
    args = parser.parse_args()
    main_from_path(args.json)
