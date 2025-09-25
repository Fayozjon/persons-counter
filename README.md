Конечно! Вот красивый и профессиональный `README.md` для вашего проекта с полными инструкциями по установке, запуску и использованию:

---

# 🎥 People Counter — Real-Time Video Analytics

> **Считайте людей в реальном времени** через веб-камеру, видеофайл или RTSP-поток.  
> Простой, быстрый и готовый к использованию сервис на базе YOLOv8 и Flask.

![Demo](https://img.shields.io/badge/Python-3.8%2B-blue?logo=python)
![License](https://img.shields.io/badge/License-MIT-green)
![Platform](https://img.shields.io/badge/Platform-Windows%20%7C%20Linux%20%7C%20macOS-lightgrey)

---

## 🚀 Быстрый старт

```bash
# 1. Клонируйте репозиторий
git clone https://github.com/your-username/people-counter.git
cd people-counter

# 2. Создайте и активируйте виртуальное окружение
python -m venv venv

# Windows (PowerShell):
venv\Scripts\Activate.ps1
# ИЛИ (CMD):
venv\Scripts\activate.bat

# Linux/macOS:
source venv/bin/activate

# 3. Установите зависимости
pip install -r requirements.txt

# 4. Запустите сервер
python peoplecounter.py

# 5. Откройте в браузере:
http://localhost:8069/video
```

> ✨ **Магия!** Чтобы использовать другой источник видео, просто добавьте параметр `?source`:
> ```
> http://localhost:8069/video?source=2.mp4
> http://localhost:8069/video?source=rtsp://192.168.1.100:554/stream
> http://localhost:8069/video?source=0          # веб-камера (по умолчанию)
> ```

---

## 📦 Требования

- **Python 3.8 или новее**
- **OpenCV** (устанавливается через `requirements.txt`)
- **YOLOv8** от Ultralytics
- **Flask** для веб-сервера
- **Веб-камера** или видеофайл (MP4, AVI и др.)

> 💡 **GPU рекомендуется** для максимальной производительности (NVIDIA с CUDA).

---

## 📁 Структура проекта

```
people-counter/
├── peoplecounter.py        # Основной скрипт
├── requirements.txt        # Зависимости
├── README.md
└── venv/                   # Виртуальное окружение (создаётся при установке)
```

---

## ⚙️ Параметры запуска

Сервер всегда запускается на `http://localhost:8069`.

| Параметр URL       | Описание                                      | Пример                          |
|--------------------|-----------------------------------------------|----------------------------------|
| (без `source`)     | Использовать веб-камеру по умолчанию (индекс 0)| `http://localhost:8069/video`   |
| `?source=0`        | Веб-камера (устройство 0)                     | `.../video?source=0`            |
| `?source=2.mp4`    | Локальный видеофайл                           | `.../video?source=2.mp4`        |
| `?source=rtsp://...`| RTSP-поток (IP-камера)                        | `.../video?source=rtsp://...`   |

> 📌 **Важно**: видеофайлы должны находиться в **корне проекта** или указываться с полным путём.

---

## 🛠️ Устранение неполадок

### ❌ `ModuleNotFoundError`
Убедитесь, что вы **активировали виртуальное окружение** перед запуском:
```bash
# Windows
venv\Scripts\activate

# Linux/macOS
source venv/bin/activate
```

### ❌ Камера не открывается
- Проверьте, что камера не используется другим приложением.
- Попробуйте другой индекс: `?source=1`, `?source=2`.

### ❌ Видеофайл не загружается
- Убедитесь, что файл лежит в папке проекта.
- Используйте только **поддерживаемые форматы**: MP4, AVI, MOV.

### ❌ Низкая производительность
- Установите **PyTorch с поддержкой CUDA** (для NVIDIA GPU):
  ```bash
  pip uninstall torch torchvision -y
  pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
  ```

---

## 📜 Лицензия

Этот проект распространяется под лицензией **MIT**.  
См. файл [LICENSE](LICENSE) для подробностей.

---

## 🙌 Благодарности

- [Ultralytics YOLOv8](https://github.com/ultralytics/ultralytics) — для детекции людей
- [Flask](https://flask.palletsprojects.com/) — для веб-сервера
- [OpenCV](https://opencv.org/) — для работы с видео

---

> 💡 **Совет**: добавьте этот проект в закладки — он идеально подходит для хакатонов, демо и быстрого прототипирования!  
> Made with ❤️ and `:magic:`
 