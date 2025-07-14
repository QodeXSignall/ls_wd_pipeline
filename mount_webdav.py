from ls_wb_pipeline.functions import remount_webdav, sync_label_studio_storage
import time


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--from-systemd", action="store_true", help="Не использовать --daemon")
    args = parser.parse_args()
    remount_webdav(from_systemd=args.from_systemd)
    sync_label_studio_storage()
    time.sleep(36000)