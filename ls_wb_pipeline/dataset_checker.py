import os
from collections import defaultdict, Counter

def check_dataset_duplicates(dataset_path):
    splits = ['train', 'val', 'test']
    file_to_classes = defaultdict(set)  # (split, filename) -> set of classes
    file_to_splits = defaultdict(set)   # filename -> set of splits
    filename_counts = Counter()
    errors = []

    for split in splits:
        split_path = os.path.join(dataset_path, split)
        if not os.path.isdir(split_path):
            continue

        for class_dir in os.listdir(split_path):
            class_path = os.path.join(split_path, class_dir)
            if not os.path.isdir(class_path):
                continue

            for file in os.listdir(class_path):
                if not file.lower().endswith((".jpg", ".jpeg", ".png")):
                    continue
                key = (split, file)
                file_to_classes[key].add(class_dir)
                file_to_splits[file].add(split)
                filename_counts[file] += 1

    conflict_in_classes = []
    conflict_in_splits = []
    repeated_names = []

    # Проверка 1: один и тот же файл в нескольких классах внутри одного сплита
    for (split, filename), class_set in file_to_classes.items():
        if len(class_set) > 1:
            conflict_in_classes.append({
                "split": split,
                "filename": filename,
                "classes": list(class_set)
            })

    # Проверка 2: один и тот же файл в нескольких сплитах
    for filename, split_set in file_to_splits.items():
        if len(split_set) > 1:
            conflict_in_splits.append({
                "filename": filename,
                "splits": list(split_set)
            })

    # Проверка 3: дублирующиеся имена
    for filename, count in filename_counts.items():
        if count > 1:
            repeated_names.append({
                "filename": filename,
                "count": count
            })

    return {
        "ok": not (conflict_in_classes or conflict_in_splits or repeated_names),
        "conflict_in_classes": conflict_in_classes,
        "conflict_in_splits": conflict_in_splits,
        "repeated_names": repeated_names
    }


if __name__ == "__main__":
    dataset_dir = '/Users/artur/Downloads/dataset (4)'  # <-- Укажи путь к датасету
    check_dataset_duplicates(dataset_dir)
