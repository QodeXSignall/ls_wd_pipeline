WebDAV LabelStudio Pipeline
Этот модуль автоматизирует процесс обработки видео, аннотации кадров и хранения данных в Label Studio с использованием WebDAV-хранилища.

Функциональность:
✔ Загрузка видео из WebDAV
✔ Разбиение видео на кадры
✔ Сохранение кадров в WebDAV (а не локально)
✔ Монтирование WebDAV как локальной папки
✔ Импорт кадров в Label Studio
✔ Автоматическое удаление обработанных видео
✔ Циклическое выполнение через systemd

Установка зависимостей:
pip install opencv-python requests webdavclient3

Запуск:
python webdav_labelstudio_pipeline.py

Требования:
Python 3.8+
Label Studio
WebDAV-хранилище
