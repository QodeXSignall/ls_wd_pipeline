import cv2

video_path = r"C:\Users\faizi\PycharmProjects\Tracker\ls_wb_pipeline\ls_wb_pipeline\fly00126_converted.mp4"

cap = cv2.VideoCapture(video_path)
if not cap.isOpened():
    print(f"Ошибка: Не удалось открыть видео {video_path}")
fourcc = int(cap.get(cv2.CAP_PROP_FOURCC))
codec = "".join([chr((fourcc >> 8 * i) & 0xFF) for i in range(4)])
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

print(f"Видео: {video_path}, Кодек: {codec}, Размер: {width}x{height}, Кадров: {frame_count}")

