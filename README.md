# OCR cyrillic

**Цель проекта:** создать модель для распознавания текста на отсканированных изображениях школьных учебников.

### Структура проекта
| Файл | Описание |
| --- | --- |
| **[castom_utils.py](castom_utils.py)** | кастомные функции. |
| **[config.yaml](config.yaml)** | конфигурационный файл. Можно отредактировать вручную или из ноутбука [ocr_cyrillic_project.ipynb](ocr_cyrillic_project.ipynb). |
| **[pdf_to_images.ipynb](pdf_to_images.ipynb)** | скрипт для конвертации PDF файлов с текстовым слоем в отдельные изображения и текст. Можно применять для создания тестового датасета. |
| **[synthetic_datasets.ipynb](synthetic_datasets.ipynb)** | в данном ноутбуке реализован парсинг и предобработка текста а так же создание на его основе синтетических датасетов из изображений для обучения моделей распознавания и детекции. |
| **[train_recognition.ipynb](train_recognition.ipynb)** | обучение модели распознавания текста. |
| **[train_detection.ipynb](train_detection.ipynb)** | обучение модели детекции текста. |
| **[ocr_cyrillic_project.ipynb](ocr_cyrillic_project.ipynb)** | объединение ранее обученных моделей в одну OCR модель и тестирование на реальных данных. |


### Stack
`doctr`, `torch`, `torchvision`.