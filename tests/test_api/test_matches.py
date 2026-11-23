"""Tests for matches API."""

import pytest


@pytest.mark.asyncio
async def test_get_football_matches(client):
    response = await client.get("/api/matches/football")
    assert response.status_code == 200
    data = response.json()
    assert data["sport"] == "football"
    assert "matches" in data


@pytest.mark.asyncio
async def test_get_basketball_matches(client):
    response = await client.get("/api/matches/basketball")
    assert response.status_code == 200
    data = response.json()
    assert data["sport"] == "basketball"


@pytest.mark.asyncio
async def test_get_unknown_sport(client):
    response = await client.get("/api/matches/cricket")
    assert response.status_code == 200
    data = response.json()
    assert "error" in data
