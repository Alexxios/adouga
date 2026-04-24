# Adouga — Automatic Detection of User Gaming Activity

Adouga is a desktop-first system that monitors user activity on a PC, classifies it in real time (Idle / Not Gaming / Gaming), and exposes those predictions to external reward providers via a REST API. It also ships a data-collection tool for gathering training data and an ML training pipeline for the underlying models.

## Modules

| Module | Purpose | Docs |
|---|---|---|
| [`desktop/`](desktop/) | Cross-platform Python desktop app. User mode runs ONNX inference on screenshots + telemetry; dev mode records datasets and uploads them to YaDisk. | [English](desktop/README.md) · [Russian](desktop/README.ru.md) |
| [`backend/`](backend/) | FastAPI service: user auth, predictions CRUD, external API-key access, admin endpoints, Prometheus metrics. | [English](backend/README.md) · [Russian](backend/README.ru.md) |
| [`ml/`](ml/) | Training pipelines: ResNet18 (single-modal) and YOLOv8 + tabular (multimodal, 3-class). Versatility evaluation across game/non-game testers. | [English](ml/README.md) · [Russian](ml/README.ru.md) |
| [`common/`](common/) | Poetry metapackage for shared Python dependencies across backend/desktop. | [English](common/README.md) · [Russian](common/README.ru.md) |
| [`monitoring/`](monitoring/) | Prometheus scrape config and provisioned Grafana dashboard for the backend. | [English](monitoring/README.md) · [Russian](monitoring/README.ru.md) |
| [`docs/`](docs/) | Academic documents (ТЗ / ПМИ / ТП / ВКР), presentation template, diagrams. | [English](docs/README.md) · [Russian](docs/README.ru.md) |

## Quick start

The fastest way to spin up the backend stack (PostgreSQL + FastAPI + Prometheus + Grafana):

```bash
docker compose up --build
```

| Service | URL |
|---|---|
| FastAPI / Swagger | http://localhost:8008/docs |
| Prometheus | http://localhost:9090 |
| Grafana (admin/admin) | http://localhost:3000 |

To run the desktop user app locally:

```bash
cd desktop
poetry install
poetry run python -m src.main
```

## Repository layout

```
adouga/
├── backend/        FastAPI + PostgreSQL + Alembic
├── common/         Shared Poetry metapackage
├── desktop/        Tkinter/PyQt user app + dev data-collection tool
├── docs/           Academic deliverables (RU/EN) and presentations
├── ml/             PyTorch training, YOLO+tabular multimodal classifier
├── monitoring/     Prometheus + Grafana configs
└── docker-compose.yml
```

## Requirements

- Python 3.14
- Poetry
- Docker + docker compose (for the backend stack)

## License

Educational / research project.
