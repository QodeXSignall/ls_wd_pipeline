# config_manager.py
import json
import os

CONFIG_PATH = "./config.json"

default_config = {
    "max_frames": 3000,
    "dry_run": True,
    "dataset_dir": "./dataset_yolo"
}

def load_config():
    if os.path.exists(CONFIG_PATH):
        with open(CONFIG_PATH, "r") as f:
            return json.load(f)
    return default_config.copy()

def save_config(cfg):
    with open(CONFIG_PATH, "w") as f:
        json.dump(cfg, f, indent=2)

def update_config(new_data):
    cfg = load_config()
    cfg.update(new_data)
    save_config(cfg)
    return cfg
