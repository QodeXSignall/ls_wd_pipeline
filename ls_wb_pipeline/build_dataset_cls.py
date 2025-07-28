import os
import shutil
import json
from urllib.parse import unquote
from collections import Counter
from ls_wb_pipeline.dataset_checker import check_dataset_duplicates
from sklearn.model_selection import train_test_split
from ls_wb_pipeline import settings

def get_latest_valid_annotation(annotations):
    valid = [a for a in annotations if not a.get("was_cancelled", False)]
    if not valid:
        return None
    return max(valid, key=lambda x: x.get("created_at", ""))

def build_classification_dataset(all_tasks, train_ratio=0.8, test_ratio=0.1, val_ratio=0.1):
    entries = []
    stats = Counter()
    used_image_names = set()

    existing_files = set()
    for split in ("train", "val", "test"):
        split_path = os.path.join(settings.DATASET_PATH, split)
        if not os.path.exists(split_path):
            continue
        for class_dir in os.listdir(split_path):
            class_path = os.path.join(split_path, class_dir)
            if not os.path.isdir(class_path):
                continue
            for fname in os.listdir(class_path):
                if fname.lower().endswith((".jpg", ".jpeg", ".png")):
                    existing_files.add(fname)

    for task in all_tasks:

        anns = task.get("annotations", [])
        if not anns or not isinstance(anns, list):
            continue

        latest = get_latest_valid_annotation(anns)
        if not latest:
            continue

        results = latest.get("result", [])
        if not results:
            continue

        try:
            class_name = results[0]["value"]["choices"][0]
            image_url = task["data"]["image"]
            image_name = os.path.basename(unquote(image_url))
            if image_name in used_image_names:
                continue  # ⚠️ Уже обработан
            if image_name in existing_files:
                continue  # ⚠️ Файл уже есть в датасете
            used_image_names.add(image_name)
            entries.append({
                "image": image_name,
                "class": class_name
            })
            stats[class_name] += 1
        except Exception:
            continue

    if not entries:
        print("❗ Нет валидных размеченных задач.")
        return

    # Загрузка существующих классов (если есть)
    classes_path = os.path.join(settings.DATASET_PATH, "labels.txt")
    existing_classes = []
    if os.path.exists(classes_path):
        with open(classes_path, "r", encoding="utf-8") as f:
            existing_classes = [line.strip() for line in f if line.strip()]

    # Объединение классов: старые + новые
    new_classes = sorted(stats.keys())
    all_classes = list(dict.fromkeys(existing_classes + new_classes))  # сохраняем порядок, избегаем дубликатов
    class_to_id = {cls: idx for idx, cls in enumerate(all_classes)}

    # Перезапись labels.txt
    os.makedirs(settings.DATASET_PATH, exist_ok=True)
    with open(classes_path, "w", encoding="utf-8") as f:
        for cls in all_classes:
            f.write(f"{cls}\n")

    print("\n📊 Распределение классов:")
    for cls, count in stats.items():
        print(f"{cls:25} — {count} изображений")

    # Разделение
    if len(entries) < 3:
        split_data = {"train": entries, "val": [], "test": []}
    else:
        train_val, test = train_test_split(entries, test_size=test_ratio, random_state=42, stratify=[e["class"] for e in entries])
        train, val = train_test_split(train_val, test_size=val_ratio / (train_ratio + val_ratio), random_state=42, stratify=[e["class"] for e in train_val])
        split_data = {"train": train, "val": val, "test": test}

    # Копирование
    for split, items in split_data.items():
        for item in items:
            class_id = class_to_id[item["class"]]
            class_dir = os.path.join(settings.DATASET_PATH, split, f"class_{class_id}")
            os.makedirs(class_dir, exist_ok=True)

            src = os.path.join(settings.MOUNTED_PATH, item["image"])
            dst = os.path.join(class_dir, item["image"])
            if os.path.exists(src):
                shutil.copy(src, dst)
    print(f"\n✅ Классификационный датасет собран: {settings.DATASET_PATH}")
    return {"stats": True, "path": settings.DATASET_PATH}




def analyze_classification_dataset(dataset_path):
    """
    Анализирует датасет классификации (по структуре class_0, class_1...).
    Возвращает словарь с количеством изображений по классам и сплитам.
    """
    try:
        classes_file = os.path.join(dataset_path, "labels.txt")
        if not os.path.exists(classes_file):
            return {"error": "Файл labels.txt не найден — датасет ещё не создан"}

        with open(classes_file, "r", encoding="utf-8") as f:
            classes = [line.strip() for line in f if line.strip()]

        split_counters = {"train": Counter(), "val": Counter(), "test": Counter()}

        for split in split_counters:
            split_dir = os.path.join(dataset_path, split)
            if not os.path.exists(split_dir):
                continue
            for class_id in range(len(classes)):
                class_dir = os.path.join(split_dir, f"class_{class_id}")
                if not os.path.isdir(class_dir):
                    continue
                image_files = [
                    f for f in os.listdir(class_dir)
                    if f.lower().endswith((".jpg", ".jpeg", ".png"))
                ]
                split_counters[split][class_id] = len(image_files)

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
        result["duplicates"] = check_dataset_duplicates(settings.DATASET_PATH)
        return result
    except Exception as e:
        return {"error": f"Ошибка при анализе датасета: {str(e)}"}



def main_from_json(json_path):
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    build_classification_dataset(data)
