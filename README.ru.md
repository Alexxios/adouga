# Adouga — автоматическое определение игровой активности пользователя

Adouga — это desktop-ориентированная система, которая отслеживает активность пользователя на ПК, в реальном времени классифицирует её (Idle / Not Gaming / Gaming) и предоставляет предсказания внешним сервисам наград через REST API. В комплекте идёт инструмент сбора данных для обучения и ML-пайплайн для обучения моделей.

## Модули

| Модуль | Назначение | Документация |
|---|---|---|
| [`desktop/`](desktop/) | Кроссплатформенное desktop-приложение на Python. Пользовательский режим — ONNX-инференс по скриншотам и телеметрии; dev-режим — сбор датасетов и загрузка на Яндекс.Диск. | [Русский](desktop/README.ru.md) · [English](desktop/README.md) |
| [`backend/`](backend/) | FastAPI-сервис: аутентификация пользователей, CRUD предсказаний, доступ для внешних сервисов по API-ключам, админ-эндпойнты, метрики Prometheus. | [Русский](backend/README.ru.md) · [English](backend/README.md) |
| [`ml/`](ml/) | Пайплайны обучения: ResNet18 (одно-модальный) и YOLOv8 + табличные признаки (мультимодальный, 3 класса). Оценка универсальности на разных тестировщиках. | [Русский](ml/README.ru.md) · [English](ml/README.md) |
| [`common/`](common/) | Poetry-метапакет с общими зависимостями Python для backend/desktop. | [Русский](common/README.ru.md) · [English](common/README.md) |
| [`monitoring/`](monitoring/) | Конфигурация Prometheus и provisioned-дашборд Grafana для backend. | [Русский](monitoring/README.ru.md) · [English](monitoring/README.md) |
| [`docs/`](docs/) | Академические документы (ТЗ / ПМИ / ТП / ВКР), шаблон презентации, диаграммы. | [Русский](docs/README.ru.md) · [English](docs/README.md) |

## Быстрый старт

Самый быстрый способ поднять backend-стек (PostgreSQL + FastAPI + Prometheus + Grafana):

```bash
docker compose up --build
```

| Сервис | URL |
|---|---|
| FastAPI / Swagger | http://localhost:8008/docs |
| Prometheus | http://localhost:9090 |
| Grafana (admin/admin) | http://localhost:3000 |

Запуск пользовательского desktop-приложения локально:

```bash
cd desktop
poetry install
poetry run python -m src.main
```

## Структура репозитория

```
adouga/
├── backend/        FastAPI + PostgreSQL + Alembic
├── common/         Общий Poetry-метапакет
├── desktop/        Пользовательское приложение + dev-инструмент сбора данных
├── docs/           Академические материалы (RU/EN) и презентации
├── ml/             Обучение на PyTorch, мультимодальный классификатор YOLO+табличные
├── monitoring/     Конфиги Prometheus + Grafana
└── docker-compose.yml
```

## Требования

- Python 3.14
- Poetry
- Docker + docker compose (для backend-стека)

## Лицензия

Проект учебно-исследовательский.
