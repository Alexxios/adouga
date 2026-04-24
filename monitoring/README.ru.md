# monitoring

Конфиг Prometheus и provisioning Grafana-дашборда для backend Adouga. Всё, что здесь лежит, монтируется в контейнеры Prometheus и Grafana через корневой `docker-compose.yml`.

## Содержимое

```
monitoring/
├── prometheus.yml                    # Scrape-конфиг (таргет fastapi_app:8000)
└── grafana/
    ├── provisioning/
    │   ├── datasources/datasource.yml  # Регистрирует Prometheus как datasource
    │   └── dashboards/dashboard.yml    # Где Grafana искать дашборды
    └── dashboards/
        └── adouga.json                 # Готовый дашборд Adouga
```

## Как это подключено

В `docker-compose.yml`:

- **Prometheus** монтирует `monitoring/prometheus.yml` в `/etc/prometheus/prometheus.yml` и слушает `http://localhost:9090`.
- **Grafana** монтирует `monitoring/grafana/provisioning` в `/etc/grafana/provisioning` и `monitoring/grafana/dashboards` в `/var/lib/grafana/dashboards`, слушает `http://localhost:3000` (логин по умолчанию `admin` / `admin`).

При первом запуске Grafana автоматически регистрирует Prometheus как datasource и загружает `adouga.json` — вручную ничего кликать не нужно.

## Собираемые метрики

Backend отдаёт `/metrics` (через `prometheus-fastapi-instrumentator`). Дашборд рисует как стандартные HTTP-метрики, так и эти кастомные счётчики (определены в `backend/src/main.py`):

- `adouga_prediction_uploads_total{predicted_class="..."}` — предсказания, загруженные desktop-приложением
- `adouga_active_users` — gauge, уникальные пользователи, активные за последние 5 минут
- `adouga_external_requests_total{service="..."}` — запросы внешних сервисов

## Правка дашборда

Правьте `adouga.json` напрямую либо меняйте дашборд в UI Grafana, затем **Share → Export → Save to file** и перезаписывайте `adouga.json`. Изменения подхватываются при перезапуске контейнера.

## Смена цели scrape

Prometheus-конфиг нацелен на `fastapi_app:8000` — имя сервиса в compose. Если запускаете backend вне Docker, поправьте `prometheus.yml` на `host.docker.internal:8008` (или тот порт, на котором backend слушает).
