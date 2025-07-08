from ls_wb_pipeline.functions import mount_webdav

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--from-systemd", action="store_true", help="Не использовать --daemon")
    args = parser.parse_args()
    mount_webdav(from_systemd=args.from_systemd)