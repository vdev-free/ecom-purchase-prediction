from fastapi.testclient import TestClient

from ecom.api.main import app

client = TestClient(app)


def test_health() -> None:
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}

def test_predict_smoke(monkeypatch) -> None:
    # fake bundle with a fixed threshold/version
    fake_bundle = {"threshold": 0.1427, "model_version": "v1", "model": object()}

    # fake inference result regardless of input
    def fake_predict_one(payload: dict, bundle: dict) -> dict:
        return {
            "probability": 0.123,
            "prediction": 0,
            "threshold": bundle["threshold"],
            "model_version": bundle["model_version"],
        }

    # patch get_bundle + predict_one inside predict router module
    import ecom.api.routers.predict as predict_router

    monkeypatch.setattr(predict_router, "get_bundle", lambda: fake_bundle)
    monkeypatch.setattr(predict_router, "predict_one", fake_predict_one)

    payload = {
        "Administrative": 0,
        "Administrative_Duration": 0.0,
        "Informational": 0,
        "Informational_Duration": 0.0,
        "ProductRelated": 1,
        "ProductRelated_Duration": 10.0,
        "BounceRates": 0.2,
        "ExitRates": 0.3,
        "PageValues": 100.0,
        "SpecialDay": 0.0,
        "Month": "May",
        "OperatingSystems": 2,
        "Browser": 2,
        "Region": 1,
        "TrafficType": 1,
        "VisitorType": "Returning_Visitor",
        "Weekend": False,
    }

    response = client.post("/predict", json=payload)
    assert response.status_code == 200
    assert response.json() == {
        "probability": 0.123,
        "prediction": 0,
        "threshold": 0.1427,
        "model_version": "v1",
    }