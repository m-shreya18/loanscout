from fastapi.testclient import TestClient

from app.main import app


def test_api_end_to_end(scenarios):
    with TestClient(app) as c:
        h = c.get("/health").json()
        assert h["status"] == "ok" and h["schemes"] == 10
        assert c.get("/schemes/cgtmse/fees").json()["guarantee_fee_slabs"]
        assert c.get("/schemes/nope/fees").status_code == 404
        r = c.post("/plan", json=scenarios["rural_food_processing"])
        assert r.status_code == 200
        body = r.json()
        assert body["status"].startswith("approved") and body["plan"]["subsidy"] > 0
        bad = dict(scenarios["rural_food_processing"], owner_contribution=9_999_999)
        assert c.post("/plan", json=bad).status_code == 422
        assert c.get("/").status_code == 200
