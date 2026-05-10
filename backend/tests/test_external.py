import pytest
from httpx import AsyncClient

from src.models.user import User

SAMPLE_PREDICTION = {
    "predicted_class": "Gaming",
    "confidence": 0.95,
    "probabilities": {"Gaming": 0.95, "Not Gaming": 0.05},
    "timestamp": "2026-01-15T12:00:00Z",
}


async def test_external_list_predictions(client: AsyncClient, auth_headers, test_user: User, test_api_key):
    raw_key, _ = test_api_key
    await client.post("/predictions", json=SAMPLE_PREDICTION, headers=auth_headers)

    resp = await client.get(
        "/external/predictions",
        params={"user_id": str(test_user.id)},
        headers={"X-API-Key": raw_key},
    )
    assert resp.status_code == 200
    data = resp.json()
    assert data["total"] == 1


async def test_external_invalid_key(client: AsyncClient):
    resp = await client.get("/external/predictions", headers={"X-API-Key": "invalid-key"})
    assert resp.status_code == 401


async def test_external_get_single(client: AsyncClient, auth_headers, test_api_key):
    raw_key, _ = test_api_key
    create_resp = await client.post("/predictions", json=SAMPLE_PREDICTION, headers=auth_headers)
    pred_id = create_resp.json()["id"]

    resp = await client.get(f"/external/predictions/{pred_id}", headers={"X-API-Key": raw_key})
    assert resp.status_code == 200
    assert resp.json()["id"] == pred_id
