import pytest
from httpx import AsyncClient


async def test_register_success(client: AsyncClient):
    resp = await client.post("/auth/register", json={"username": "newuser", "password": "securepass"})
    assert resp.status_code == 201
    data = resp.json()
    assert data["username"] == "newuser"
    assert "id" in data


async def test_register_duplicate(client: AsyncClient):
    await client.post("/auth/register", json={"username": "dup", "password": "securepass"})
    resp = await client.post("/auth/register", json={"username": "dup", "password": "securepass"})
    assert resp.status_code == 409


async def test_register_short_password(client: AsyncClient):
    resp = await client.post("/auth/register", json={"username": "user", "password": "short"})
    assert resp.status_code == 422


async def test_login_success(client: AsyncClient, test_user):
    resp = await client.post("/auth/login", data={"username": "testuser", "password": "testpassword"})
    assert resp.status_code == 200
    data = resp.json()
    assert "access_token" in data
    assert data["token_type"] == "bearer"


async def test_login_wrong_password(client: AsyncClient, test_user):
    resp = await client.post("/auth/login", data={"username": "testuser", "password": "wrong"})
    assert resp.status_code == 401


async def test_protected_route_no_token(client: AsyncClient):
    resp = await client.get("/users/me")
    assert resp.status_code == 401
