# OCR cyrillic

**Цель проекта:** создать модель для распознавания текста на фотографиях или сканах из книг.

За основу возьмем модель из библиотеки [docTR](https://github.com/mindee/doctr).

### Структура проекта
| Файл | Описание |
| --- | --- |
| **[castom_utils.py](castom_utils.py)** | Кастомные функции, используемые в нескольких ноутбуках. |
| **[config.yaml](config.yaml)** | Конфигурационный файл. Можно отредактировать вручную или из ноутбука [ocr_cyrillic_project.ipynb](ocr_cyrillic_project.ipynb). |
| **[pdf_to_images.ipynb](pdf_to_images.ipynb)** | Скрипт для конвертации PDF файлов с текстовым слоем в отдельные изображения и текст. Можно применять для создания тестового датасета. |
| **[synthetic_datasets.ipynb](synthetic_datasets.ipynb)** | В данном ноутбуке реализован парсинг и предобработка текста а так же создание на его основе синтетических датасетов из изображений для обучения моделей распознавания и детекции. |
| **[train_recognition.ipynb](train_recognition.ipynb)** | Обучение модели распознавания текста. |
| **[train_detection.ipynb](train_detection.ipynb)** | Обучение модели детекции текста. |
| **[ocr_cyrillic.ipynb](ocr_cyrillic.ipynb)** | Объединение ранее обученных моделей в одну OCR модель и тестирование на реальных данных. |


### Stack
`doctr`, `torch`, `torchvision`.