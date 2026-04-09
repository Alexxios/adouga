import time

from fastapi import FastAPI, Request
from prometheus_client import Counter, Gauge
from prometheus_fastapi_instrumentator import Instrumentator

from src.routers import admin, auth, external, predictions, users

app = FastAPI(title="Adouga API", version="0.1.0")

# Custom Prometheus metrics
PREDICTION_UPLOADS = Counter(
    "adouga_prediction_uploads_total",
    "Total prediction uploads",
    ["predicted_class"],
)
ACTIVE_USERS = Gauge(
    "adouga_active_users",
    "Distinct users active in last 5 minutes",
)
EXTERNAL_REQUESTS = Counter(
    "adouga_external_requests_total",
    "External service API requests",
    ["service_name"],
)

# Track active users: {user_id: last_seen_timestamp}
_active_users: dict[str, float] = {}
ACTIVE_WINDOW = 300  # 5 minutes


@app.middleware("http")
async def track_active_users(request: Request, call_next):
    response = await call_next(request)
    user = getattr(request.state, "user", None)
    if user:
        _active_users[str(user.id)] = time.time()
    return response


def _update_active_users_gauge():
    cutoff = time.time() - ACTIVE_WINDOW
    expired = [uid for uid, ts in _active_users.items() if ts < cutoff]
    for uid in expired:
        del _active_users[uid]
    ACTIVE_USERS.set(len(_active_users))


# Routers
app.include_router(auth.router)
app.include_router(users.router)
app.include_router(predictions.router)
app.include_router(external.router)
app.include_router(admin.router)

# Prometheus instrumentation
Instrumentator().instrument(app).expose(app, endpoint="/metrics")


@app.get("/")
async def health():
    _update_active_users_gauge()
    return {"status": "ok"}
