# Модель классификации игр

Бинарный классификатор на основе ResNet18, различающий игровые и неигровые изображения.

## Структура проекта

```
.
├── src/
│   ├── data_preparation.py  # Скачивание и объединение датасетов
│   ├── dataset_split.py     # Разделение на train/test
│   ├── dataset.py           # PyTorch Dataset
│   ├── model.py             # Классификатор ResNet18
│   ├── train.py             # Скрипт обучения
│   └── export_onnx.py       # Экспорт модели в TorchScript
├── tests/
│   ├── evaluate.py          # Оценка модели
│   └── benchmark.py         # Бенчмарк производительности
├── data/
│   ├── combined/            # Объединённый датасет
│   └── split/               # Разбиение train/test
└── models/                  # Сохранённые модели
```

## Установка

Установка зависимостей через Poetry:

```bash
poetry install
```

## Использование

### 1. Скачивание и подготовка данных

Скачать изображения из HuggingFace-датасетов и объединить их:

```bash
poetry run python src/data_preparation.py
```

### 2. Разделение датасета

Разделить объединённый датасет на train (80%) и test (20%):

```bash
poetry run python src/dataset_split.py
```

### 3. Обучение модели

Обучить классификатор на ResNet18:

```bash
poetry run python src/train.py
```

Результаты обучения:
- Train-датасет: 1600 изображений
- Test-датасет: 400 изображений
- Лучшая accuracy на test: 100%
- Модель сохраняется в: `models/model_best.pth`

### 4. Оценка модели

Оценить обученную модель на тестовом наборе:

```bash
poetry run python tests/evaluate.py
```

### 5. Экспорт модели

Экспортировать модель в формат TorchScript:

```bash
poetry run python src/export_onnx.py
```

Замечание: у ONNX-экспорта есть проблемы совместимости с Python 3.14, поэтому используется TorchScript.

### 6. Бенчмарк моделей

Сравнение производительности PyTorch и TorchScript:

```bash
poetry run python tests/benchmark.py
```

Результаты бенчмарка (на Apple M-series):
- PyTorch: ~4.43 мс на изображение (~226 FPS)
- TorchScript: ~4.52 мс на изображение (~221 FPS)

## Архитектура модели

- База: ResNet18 (предобучена на ImageNet)
- Вход: RGB-изображения 224×224
- Выход: 2 класса (game, not_game)
- Обучение: 10 эпох, оптимизатор Adam, learning rate 0.001

## Датасет

### Игровые изображения (метка: 0)
- taesiri/GameplayCaptions-GPT-4V
- taesiri/GameplayCaptions-GPT-4V-V2
- Bingsu/Gameplay_Images

### Неигровые изображения (метка: 1)
- mlfoundations-cua-dev/easyr1-showui-desktop-only-4k9-omniparser-qwen-tool-call-4MP
- showlab/ShowUI-desktop

Всего: ~2000 изображений (1500 игровых, 500 неигровых).

## Результаты

- Общая accuracy: 100%
- Game accuracy: 100% (300/300)
- Not-Game accuracy: 100% (100/100)

## Требования

- Python 3.14
- PyTorch 2.9+
- torchvision 0.24+
- datasets 4.5+
- Pillow
- Остальные зависимости — в `pyproject.toml`
