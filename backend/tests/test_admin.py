import pytest
from httpx import AsyncClient


async def test_create_api_key(client: AsyncClient, admin_headers):
    resp = await client.post(
        "/admin/api-keys", json={"service_name": "reward-provider"}, headers=admin_headers
    )
    assert resp.status_code == 201
    data = resp.json()
    assert "raw_key" in data
    assert data["service_name"] == "reward-provider"
    assert data["is_active"] is True


async def test_list_api_keys(client: AsyncClient, admin_headers, test_api_key):
    resp = await client.get("/admin/api-keys", headers=admin_headers)
    assert resp.status_code == 200
    data = resp.json()
    assert len(data) >= 1


async def test_revoke_api_key(client: AsyncClient, admin_headers, test_api_key):
    raw_key, api_key = test_api_key
    resp = await client.delete(f"/admin/api-keys/{api_key.id}", headers=admin_headers)
    assert resp.status_code == 204

    # Revoked key should not work
    resp = await client.get("/external/predictions", headers={"X-API-Key": raw_key})
    assert resp.status_code == 401


async def test_non_admin_access(client: AsyncClient, auth_headers):
    resp = await client.post(
        "/admin/api-keys", json={"service_name": "test"}, headers=auth_headers
    )
    assert resp.status_code == 403
