# monitoring

Prometheus scrape config and Grafana dashboard provisioning for the Adouga backend. Everything here is mounted into the Prometheus and Grafana containers by the repo-root `docker-compose.yml`.

## Contents

```
monitoring/
├── prometheus.yml                    # Scrape config (targets fastapi_app:8000)
└── grafana/
    ├── provisioning/
    │   ├── datasources/datasource.yml  # Registers Prometheus as a datasource
    │   └── dashboards/dashboard.yml    # Tells Grafana where to find dashboards
    └── dashboards/
        └── adouga.json                 # Pre-built Adouga dashboard
```

## How it is wired up

In `docker-compose.yml`:

- **Prometheus** mounts `monitoring/prometheus.yml` to `/etc/prometheus/prometheus.yml` and exposes `http://localhost:9090`.
- **Grafana** mounts `monitoring/grafana/provisioning` to `/etc/grafana/provisioning` and `monitoring/grafana/dashboards` to `/var/lib/grafana/dashboards`, then exposes `http://localhost:3000` (default credentials `admin` / `admin`).

On first boot Grafana auto-registers the Prometheus datasource and loads `adouga.json`; no manual clicking required.

## Metrics scraped

The backend exposes `/metrics` (via `prometheus-fastapi-instrumentator`). The dashboard plots both the default HTTP metrics and these custom counters (defined in `backend/src/main.py`):

- `adouga_prediction_uploads_total{predicted_class="..."}` — predictions uploaded by the desktop app
- `adouga_active_users` — gauge, distinct users active in the last 5 minutes
- `adouga_external_requests_total{service="..."}` — external-service API calls

## Editing the dashboard

Edit `adouga.json` directly, or tweak in Grafana UI then **Share → Export → Save to file** and overwrite `adouga.json`. Changes are picked up on container restart.

## Changing the scrape target

The Prometheus config targets `fastapi_app:8000` — the compose service name. If you run the backend outside Docker, edit `prometheus.yml` and point it at `host.docker.internal:8008` (or whichever port you bind).
