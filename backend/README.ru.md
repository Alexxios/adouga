# Adouga Backend API

REST API, связывающий desktop-приложение Adouga (классификатор игровой активности) с внешними поставщиками услуг наград.

## Возможности

- **Регистрация пользователей и JWT-аутентификация**
- **CRUD предсказаний игровой активности** — desktop-приложение загружает результаты инференса, пользователи управляют своими данными
- **API для внешних сервисов** — провайдеры получают предсказания по API-ключу
- **Админ-панель** — управление жизненным циклом API-ключей
- **Мониторинг Prometheus + Grafana** — HTTP-метрики, активные пользователи, частота предсказаний, распределение по внешним сервисам

## Технологический стек

- Python 3.14, FastAPI, Uvicorn
- PostgreSQL, SQLAlchemy 2.x (async), Alembic
- JWT (python-jose), bcrypt
- Prometheus (prometheus-fastapi-instrumentator)

## API-эндпойнты

| Метод | Путь | Авторизация | Описание |
|--------|------|------|-------------|
| POST | `/auth/register` | — | Регистрация нового пользователя |
| POST | `/auth/login` | — | Получение JWT-токена |
| GET | `/users/me` | JWT | Профиль текущего пользователя |
| POST | `/predictions` | JWT | Загрузка предсказания |
| GET | `/predictions` | JWT | Список своих предсказаний (пагинация, фильтрация) |
| GET | `/predictions/{id}` | JWT | Получить одно предсказание |
| DELETE | `/predictions/{id}` | JWT | Удалить своё предсказание |
| GET | `/external/predictions` | API-ключ | Запрос предсказаний по user_id |
| GET | `/external/predictions/{id}` | API-ключ | Получить предсказание по ID |
| POST | `/admin/api-keys` | JWT (admin) | Создать API-ключ |
| GET | `/admin/api-keys` | JWT (admin) | Список API-ключей |
| DELETE | `/admin/api-keys/{id}` | JWT (admin) | Отозвать API-ключ |

## Быстрый старт

### В Docker (рекомендуется)

Из корня репозитория:

```bash
docker compose up --build
```

Запустятся четыре сервиса:

| Сервис | URL |
|---------|-----|
| FastAPI | http://localhost:8008 |
| Swagger UI | http://localhost:8008/docs |
| Prometheus | http://localhost:9090 |
| Grafana | http://localhost:3000 (admin / admin) |

Entrypoint автоматически выполняет `alembic upgrade head` перед запуском сервера.

### Локальная разработка

```bash
cd backend
poetry install
```

Задайте переменные окружения в `.env`:

```
DATABASE_URL=postgresql+asyncpg://adouga:adouga@localhost:5432/adouga
SECRET_KEY=your-secret-key
ACCESS_TOKEN_EXPIRE_MINUTES=60
```

Примените миграции и запустите сервер:

```bash
poetry run alembic upgrade head
poetry run uvicorn src.main:app --host 0.0.0.0 --port 8008 --reload
```

## Запуск тестов

Тесты используют in-memory SQLite и не требуют PostgreSQL.

```bash
cd backend
poetry run pytest tests/ -v
```

## Структура проекта

```
src/
    main.py              # Точка входа, метрики Prometheus, подключение роутеров
    core/
        config.py        # Настройки (pydantic-settings, env)
        security.py      # JWT, bcrypt, хеширование API-ключей
        deps.py          # Зависимости FastAPI (auth, сессия БД)
    db/
        session.py       # Асинхронный движок и сессия SQLAlchemy
    models/
        base.py          # Declarative base (UUID pk, timestamps)
        user.py          # Модель пользователя
        prediction.py    # Модель предсказания
        api_key.py       # Модель API-ключа
    schemas/
        user.py          # UserCreate, UserRead, Token
        prediction.py    # PredictionCreate, PredictionRead, PaginatedPredictions
        api_key.py       # ApiKeyCreate, ApiKeyRead, ApiKeyCreated
    routers/
        auth.py          # Регистрация и логин
        users.py         # Профиль пользователя
        predictions.py   # CRUD предсказаний
        external.py      # Доступ для внешних сервисов (read-only)
        admin.py         # Управление API-ключами
alembic/                 # Миграции БД
tests/                   # Набор pytest-тестов (21 тест)
```

## Аутентификация

**Пользователи (desktop-приложение):** регистрируются через `/auth/register`, логин через `/auth/login` для получения JWT-токена. Передавайте его в заголовке `Authorization: Bearer <token>`.

**Внешние сервисы:** получают API-ключ от администратора. Передавайте его в заголовке `X-API-Key: <key>`. Ключи хранятся в БД в виде хеша (SHA-256); исходный ключ показывается только один раз при создании.

## Мониторинг

Prometheus забирает `/metrics` — как автоматические HTTP-метрики, так и кастомные счётчики:

- `adouga_prediction_uploads_total` (по предсказанному классу)
- `adouga_active_users` (уникальные пользователи за последние 5 минут)
- `adouga_external_requests_total` (по имени сервиса)

Grafana при старте автоматически получает provisioned-дашборд.
