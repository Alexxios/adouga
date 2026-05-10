# Adouga Backend API

REST API that connects the Adouga desktop application (gaming activity classifier) with external reward service providers.

## Features

- **User registration & JWT authentication**
- **Gaming activity predictions CRUD** -- desktop app uploads inference results, users manage their data
- **External service API** -- reward providers retrieve predictions via API keys
- **Admin panel** -- API key lifecycle management
- **Prometheus + Grafana monitoring** -- HTTP metrics, active users, prediction rates, external service distribution

## Tech Stack

- Python 3.14, FastAPI, Uvicorn
- PostgreSQL, SQLAlchemy 2.x (async), Alembic
- JWT (python-jose), bcrypt
- Prometheus (prometheus-fastapi-instrumentator)

## API Endpoints

| Method | Path | Auth | Description |
|--------|------|------|-------------|
| POST | `/auth/register` | -- | Register a new user |
| POST | `/auth/login` | -- | Get JWT access token |
| GET | `/users/me` | JWT | Current user profile |
| POST | `/predictions` | JWT | Upload a prediction |
| GET | `/predictions` | JWT | List own predictions (paginated, filterable) |
| GET | `/predictions/{id}` | JWT | Get single prediction |
| DELETE | `/predictions/{id}` | JWT | Delete own prediction |
| GET | `/external/predictions` | API Key | Query predictions by user_id |
| GET | `/external/predictions/{id}` | API Key | Get prediction by ID |
| POST | `/admin/api-keys` | JWT (admin) | Create API key |
| GET | `/admin/api-keys` | JWT (admin) | List API keys |
| DELETE | `/admin/api-keys/{id}` | JWT (admin) | Revoke API key |

## Quick Start

### With Docker (recommended)

From the repository root:

```bash
docker compose up --build
```

This starts four services:

| Service | URL |
|---------|-----|
| FastAPI | http://localhost:8008 |
| Swagger UI | http://localhost:8008/docs |
| Prometheus | http://localhost:9090 |
| Grafana | http://localhost:3000 (admin / admin) |

The entrypoint runs `alembic upgrade head` automatically before starting the server.

### Local Development

```bash
cd backend
poetry install
```

Set environment variables in `.env`:

```
DATABASE_URL=postgresql+asyncpg://adouga:adouga@localhost:5432/adouga
SECRET_KEY=your-secret-key
ACCESS_TOKEN_EXPIRE_MINUTES=60
```

Run migrations and start the server:

```bash
poetry run alembic upgrade head
poetry run uvicorn src.main:app --host 0.0.0.0 --port 8008 --reload
```

## Running Tests

Tests use an in-memory SQLite database and do not require PostgreSQL.

```bash
cd backend
poetry run pytest tests/ -v
```

## Project Structure

```
src/
    main.py              # App entry point, Prometheus metrics, router wiring
    core/
        config.py        # Settings (env-based via pydantic-settings)
        security.py      # JWT, bcrypt, API key hashing
        deps.py          # FastAPI dependencies (auth, DB session)
    db/
        session.py       # Async SQLAlchemy engine and session
    models/
        base.py          # Declarative base (UUID pk, timestamps)
        user.py          # User model
        prediction.py    # Prediction model
        api_key.py       # API key model
    schemas/
        user.py          # UserCreate, UserRead, Token
        prediction.py    # PredictionCreate, PredictionRead, PaginatedPredictions
        api_key.py       # ApiKeyCreate, ApiKeyRead, ApiKeyCreated
    routers/
        auth.py          # Registration and login
        users.py         # User profile
        predictions.py   # Prediction CRUD
        external.py      # External service read-only access
        admin.py         # API key management
alembic/                 # Database migrations
tests/                   # Pytest suite (21 tests)
```

## Authentication

**Users (desktop app):** Register via `/auth/register`, login via `/auth/login` to get a JWT token. Pass it as `Authorization: Bearer <token>`.

**External services:** Receive an API key from an admin. Pass it as `X-API-Key: <key>` header. Keys are hashed (SHA-256) in the database; the raw key is shown only once at creation.

## Monitoring

Prometheus scrapes `/metrics` for both automatic HTTP metrics and custom counters:

- `adouga_prediction_uploads_total` (by predicted class)
- `adouga_active_users` (distinct users in last 5 minutes)
- `adouga_external_requests_total` (by service name)

Grafana is auto-provisioned with a dashboard at startup.
