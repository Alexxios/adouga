# common

Poetry-метапакет, в котором централизованы Python-зависимости, общие для `backend/` и `desktop/`. Собственного кода в нём нет — только `pyproject.toml`, пиннинг общих инструментов (pytest, setuptools, wheel), чтобы все нижестоящие модули резолвили одинаковые версии.

## Зачем он нужен

Держать базовые зависимости в одном месте — значит избежать расхождения вида «каждый подпроект пинит pytest по-своему». Нижестоящие модули подключают его как локальную path-зависимость в своём `pyproject.toml`:

```toml
[tool.poetry.dependencies]
common-deps = { path = "../common", develop = true }
```

## Структура

```
common/
├── pyproject.toml    # name = "common-deps"; общие пины зависимостей
├── poetry.lock
└── src/              # пустой пакет-заглушка
    └── __init__.py
```

## Установка

Пакет подтягивается транзитивно при `poetry install` в нижестоящем модуле (`backend/`, `desktop/`). Отдельно его устанавливать обычно не нужно. Чтобы обновить lockfile после правки `pyproject.toml`:

```bash
cd common
poetry lock --no-update
```

## Добавление общей зависимости

1. Добавьте пакет в секцию `[tool.poetry.dependencies]` файла `common/pyproject.toml`.
2. Выполните `poetry lock` внутри `common/`.
3. Выполните `poetry install` в каждом нижестоящем модуле, которому нужно обновление.
