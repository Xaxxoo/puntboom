"""Tests for predictions API."""

import pytest
from datetime import datetime, timezone

from puntboom.models.prediction import Prediction


@pytest.mark.asyncio
async def test_get_predictions_empty(client):
    response = await client.get("/api/predictions/football")
    assert response.status_code == 200
    data = response.json()
    assert data["predictions"] == []
    assert data["sport"] == "football"


@pytest.mark.asyncio
async def test_get_predictions_with_data(client, session):
    pred = Prediction(
        sport="football",
        match_external_id="123",
        match_date=datetime.now(timezone.utc),
        home_team="Team A",
        away_team="Team B",
        prob_home=0.5,
        prob_draw=0.3,
        prob_away=0.2,
        predicted_outcome="H",
        confidence=0.5,
    )
    session.add(pred)
    await session.commit()

    response = await client.get("/api/predictions/football")
    assert response.status_code == 200
    data = response.json()
    assert len(data["predictions"]) == 1
    assert data["predictions"][0]["home_team"] == "Team A"


@pytest.mark.asyncio
async def test_get_all_predictions(client):
    response = await client.get("/api/predictions/")
    assert response.status_code == 200
