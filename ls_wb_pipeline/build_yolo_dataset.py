import os
import json
import argparse
from collections import Counter
from sklearn.model_selection import train_test_split
from ls_wb_pipeline import functions
import shutil

# ==== НАСТРОЙКИ (можно менять внутри скрипта) ====
SOURCE_IMAGE_DIR = functions.MOUNTED_PATH
OUTPUT_DIR = "./dataset_yolo"
SPLIT_RATIO = (0.8, 0.1, 0.1)  # train, val, test

def main(json_path):
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
            image_name = os.path.basename(image_url)
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

    # Подготовка классов
    class_list = sorted(class_names)
    class_to_index = {name: i for i, name in enumerate(class_list)}

    # Создание папок
    splits = ["train", "val", "test"]
    for split in splits:
        os.makedirs(os.path.join(OUTPUT_DIR, "images", split), exist_ok=True)
        os.makedirs(os.path.join(OUTPUT_DIR, "labels", split), exist_ok=True)

    # Разделение на train/val/test
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

    # Сохраняем classes.txt
    with open(os.path.join(OUTPUT_DIR, "classes.txt"), "w") as f:
        for name in class_list:
            f.write(f"{name}\n")

    # Распределение классов
    summary = Counter(e["class"] for e in entries)
    print(f"\nДатасет собран. {OUTPUT_DIR}")
    total = sum(summary.values())
    print("\nРаспределение классов:")
    for cls in class_list:
        count = summary[cls]
        percent = (count / total) * 100
        print(f"{cls:25} — {count:3} изображений ({percent:.1f}%)")

    print("\nРекомендации:")
    avg = total / len(class_list)
    for cls in class_list:
        diff = summary[cls] - avg
        if diff < -10:
            print(f"Классу '{cls}' не хватает примерно {int(-diff)} примеров для баланса.")
        elif diff > 10:
            print(f"Класса '{cls}' заметно больше остальных (на +{int(diff)}).")

def analyze_full_dataset(dataset_path=OUTPUT_DIR):
    labels_root = os.path.join(dataset_path, "labels")
    classes_file = os.path.join(dataset_path, "classes.txt")

    if not os.path.exists(labels_root) or not os.path.exists(classes_file):
        print("❌ Не найден labels/ или classes.txt — датасет ещё не создан?")
        return

    # Загрузка названий классов
    with open(classes_file, "r", encoding="utf-8") as f:
        classes = [line.strip() for line in f if line.strip()]

    counter = Counter()
    for split in ("train", "val", "test"):
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
                        counter[class_id] += 1

    total = sum(counter.values())
    print("\n📦 Общая картина по всем размеченным классам:")
    avg = total / len(classes)

    for class_id, name in enumerate(classes):
        count = counter[class_id]
        percent = (count / total) * 100 if total else 0
        print(f"{class_id}: {name:25} — {count:3} шт. ({percent:.1f}%)", end="")
        if count < avg - 10:
            print("   ⚠️ Мало примеров")
        elif count > avg + 10:
            print("   ℹ️ Превышает среднее")
        else:
            print("")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Сборка YOLO датасета из Label Studio JSON")
    parser.add_argument("--json", required=True, help="Путь до экспортированного JSON-файла из Label Studio")
    args = parser.parse_args()
    main(args.json)
