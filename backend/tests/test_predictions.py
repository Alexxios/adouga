import uuid
from datetime import datetime, timezone

import pytest
from httpx import AsyncClient

from src.models.user import User

SAMPLE_PREDICTION = {
    "predicted_class": "Gaming",
    "confidence": 0.95,
    "probabilities": {"Gaming": 0.95, "Not Gaming": 0.05},
    "timestamp": "2026-01-15T12:00:00Z",
}


async def test_create_prediction(client: AsyncClient, auth_headers):
    resp = await client.post("/predictions", json=SAMPLE_PREDICTION, headers=auth_headers)
    assert resp.status_code == 201
    data = resp.json()
    assert data["predicted_class"] == "Gaming"
    assert data["confidence"] == 0.95


async def test_list_predictions_empty(client: AsyncClient, auth_headers):
    resp = await client.get("/predictions", headers=auth_headers)
    assert resp.status_code == 200
    data = resp.json()
    assert data["items"] == []
    assert data["total"] == 0


async def test_list_predictions(client: AsyncClient, auth_headers):
    await client.post("/predictions", json=SAMPLE_PREDICTION, headers=auth_headers)
    resp = await client.get("/predictions", headers=auth_headers)
    assert resp.status_code == 200
    data = resp.json()
    assert data["total"] == 1
    assert len(data["items"]) == 1


async def test_filter_by_class(client: AsyncClient, auth_headers):
    await client.post("/predictions", json=SAMPLE_PREDICTION, headers=auth_headers)
    not_gaming = {**SAMPLE_PREDICTION, "predicted_class": "Not Gaming", "confidence": 0.8}
    await client.post("/predictions", json=not_gaming, headers=auth_headers)

    resp = await client.get("/predictions", params={"predicted_class": "Gaming"}, headers=auth_headers)
    data = resp.json()
    assert data["total"] == 1
    assert data["items"][0]["predicted_class"] == "Gaming"


async def test_get_single_prediction(client: AsyncClient, auth_headers):
    create_resp = await client.post("/predictions", json=SAMPLE_PREDICTION, headers=auth_headers)
    pred_id = create_resp.json()["id"]

    resp = await client.get(f"/predictions/{pred_id}", headers=auth_headers)
    assert resp.status_code == 200
    assert resp.json()["id"] == pred_id


async def test_get_other_users_prediction(client: AsyncClient, auth_headers, admin_headers):
    create_resp = await client.post("/predictions", json=SAMPLE_PREDICTION, headers=admin_headers)
    pred_id = create_resp.json()["id"]

    resp = await client.get(f"/predictions/{pred_id}", headers=auth_headers)
    assert resp.status_code == 404


async def test_delete_prediction(client: AsyncClient, auth_headers):
    create_resp = await client.post("/predictions", json=SAMPLE_PREDICTION, headers=auth_headers)
    pred_id = create_resp.json()["id"]

    resp = await client.delete(f"/predictions/{pred_id}", headers=auth_headers)
    assert resp.status_code == 204

    resp = await client.get(f"/predictions/{pred_id}", headers=auth_headers)
    assert resp.status_code == 404


async def test_delete_other_users_prediction(client: AsyncClient, auth_headers, admin_headers):
    create_resp = await client.post("/predictions", json=SAMPLE_PREDICTION, headers=admin_headers)
    pred_id = create_resp.json()["id"]

    resp = await client.delete(f"/predictions/{pred_id}", headers=auth_headers)
    assert resp.status_code == 404
