from __future__ import annotations

import pandas as pd

from collector.supabase_repository import SupabaseConfig, SupabaseRepository


class FakeResponse:
    def __init__(self, payload: list | None = None) -> None:
        self._payload = payload or []

    def raise_for_status(self) -> None:
        return None

    def json(self) -> list:
        return self._payload


class FakeSession:
    def __init__(self, get_responses: list[FakeResponse], post_responses: list[FakeResponse]) -> None:
        self.get_responses = get_responses
        self.post_responses = post_responses
        self.get_calls: list[dict] = []
        self.post_calls: list[dict] = []

    def get(self, url: str, headers: dict, params: dict, timeout: int) -> FakeResponse:
        self.get_calls.append({"url": url, "headers": headers, "params": params, "timeout": timeout})
        return self.get_responses.pop(0)

    def post(self, url: str, headers: dict, json: list | dict, timeout: int) -> FakeResponse:
        self.post_calls.append({"url": url, "headers": headers, "json": json, "timeout": timeout})
        return self.post_responses.pop(0)


def make_repository(session: FakeSession) -> SupabaseRepository:
    return SupabaseRepository(
        SupabaseConfig(url="https://example.supabase.co", key="test-key"),
        session=session,
    )


def test_get_or_create_asset_returns_existing_asset() -> None:
    session = FakeSession(get_responses=[FakeResponse([{"id": "asset-1"}])], post_responses=[])
    repository = make_repository(session)

    asset_id = repository.get_or_create_asset("aapl", name="Apple", asset_class="stock")

    assert asset_id == "asset-1"
    assert session.get_calls[0]["params"] == {"ticker": "eq.AAPL", "select": "id"}
    assert session.post_calls == []


def test_get_or_create_asset_creates_missing_asset() -> None:
    session = FakeSession(
        get_responses=[FakeResponse([])],
        post_responses=[FakeResponse([{"id": "asset-2"}])],
    )
    repository = make_repository(session)

    asset_id = repository.get_or_create_asset("btcusdt", asset_class="crypto")

    assert asset_id == "asset-2"
    assert session.post_calls[0]["json"] == {
        "ticker": "BTCUSDT",
        "name": "BTCUSDT",
        "asset_class": "crypto",
    }
    assert session.post_calls[0]["headers"]["Prefer"] == "return=representation"


def test_upsert_prices_batches_payload() -> None:
    session = FakeSession(
        get_responses=[],
        post_responses=[FakeResponse(), FakeResponse()],
    )
    repository = make_repository(session)
    prices = pd.DataFrame(
        {
            "timestamp": pd.date_range("2024-01-01", periods=3, freq="D", tz="UTC"),
            "open": [10, 11, 12],
            "high": [12, 13, 14],
            "low": [9, 10, 11],
            "close": [11, 12, 13],
            "volume": [1000, 1100, 1200],
        }
    )

    inserted = repository.upsert_prices("asset-1", prices, batch_size=2)

    assert inserted == 3
    assert len(session.post_calls) == 2
    assert session.post_calls[0]["headers"]["Prefer"] == "resolution=merge-duplicates"
    assert session.post_calls[0]["json"][0]["asset_id"] == "asset-1"
    assert session.post_calls[0]["json"][0]["close"] == 11.0
    assert session.post_calls[1]["json"][0]["volume"] == 1200


def test_get_asset_id_raises_when_missing() -> None:
    session = FakeSession(get_responses=[FakeResponse([])], post_responses=[])
    repository = make_repository(session)

    try:
        repository.get_asset_id("missing")
    except ValueError as error:
        assert "MISSING" in str(error)
    else:
        raise AssertionError("Expected missing asset to raise ValueError")


def test_get_prices_returns_dataframe() -> None:
    session = FakeSession(
        get_responses=[
            FakeResponse(
                [
                    {
                        "timestamp": "2024-01-01T00:00:00+00:00",
                        "open": 10,
                        "high": 12,
                        "low": 9,
                        "close": 11,
                        "volume": 1000,
                    }
                ]
            )
        ],
        post_responses=[],
    )
    repository = make_repository(session)

    prices = repository.get_prices("asset-1", limit=100)

    assert len(prices) == 1
    assert prices.loc[0, "close"] == 11
    assert session.get_calls[0]["params"]["order"] == "timestamp.asc"
    assert session.get_calls[0]["params"]["limit"] == "100"


def test_get_features_filters_by_feature_set() -> None:
    session = FakeSession(
        get_responses=[
            FakeResponse(
                [
                    {
                        "timestamp": "2024-01-01T00:00:00+00:00",
                        "features": {"return_1d": 0.01},
                    }
                ]
            )
        ],
        post_responses=[],
    )
    repository = make_repository(session)

    features = repository.get_features("asset-1", "technical_v1", limit=50)

    assert len(features) == 1
    assert session.get_calls[0]["params"]["feature_set"] == "eq.technical_v1"
    assert session.get_calls[0]["params"]["limit"] == "50"


def test_get_features_can_order_descending() -> None:
    session = FakeSession(get_responses=[FakeResponse([])], post_responses=[])
    repository = make_repository(session)

    repository.get_features("asset-1", "technical_v1", ascending=False)

    assert session.get_calls[0]["params"]["order"] == "timestamp.desc"


def test_get_labels_filters_by_method_and_horizon() -> None:
    session = FakeSession(
        get_responses=[
            FakeResponse(
                [
                    {
                        "timestamp": "2024-01-01T00:00:00+00:00",
                        "label": "BUY",
                        "outcome_return": 0.03,
                    }
                ]
            )
        ],
        post_responses=[],
    )
    repository = make_repository(session)

    labels = repository.get_labels("asset-1", "triple_barrier", 5)

    assert len(labels) == 1
    assert session.get_calls[0]["params"]["label_method"] == "eq.triple_barrier"
    assert session.get_calls[0]["params"]["horizon"] == "eq.5"


def test_upsert_features_stores_json_payload() -> None:
    session = FakeSession(get_responses=[], post_responses=[FakeResponse()])
    repository = make_repository(session)
    features = pd.DataFrame(
        {
            "timestamp": pd.date_range("2024-01-01", periods=2, freq="D", tz="UTC"),
            "return_1d": [0.01, None],
            "rsi_14": [55.5, 60.0],
        }
    )

    inserted = repository.upsert_features(
        asset_id="asset-1",
        features=features,
        feature_columns=["return_1d", "rsi_14"],
        feature_set="technical_test",
    )

    assert inserted == 1
    payload = session.post_calls[0]["json"][0]
    assert payload["feature_set"] == "technical_test"
    assert payload["features"] == {"return_1d": 0.01, "rsi_14": 55.5}


def test_upsert_labels_stores_method_and_horizon() -> None:
    session = FakeSession(get_responses=[], post_responses=[FakeResponse()])
    repository = make_repository(session)
    labels = pd.DataFrame(
        {
            "timestamp": pd.date_range("2024-01-01", periods=2, freq="D", tz="UTC"),
            "label": ["BUY", None],
            "outcome_return": [0.03, None],
            "label_exit_timestamp": ["2024-01-03T00:00:00+00:00", None],
        }
    )

    inserted = repository.upsert_labels(
        asset_id="asset-1",
        labels=labels,
        label_method="triple_barrier",
        horizon=5,
    )

    assert inserted == 1
    payload = session.post_calls[0]["json"][0]
    assert payload["label_method"] == "triple_barrier"
    assert payload["horizon"] == 5
    assert payload["label"] == "BUY"
    assert payload["outcome_return"] == 0.03


def test_create_model_run_reuses_existing_run() -> None:
    session = FakeSession(get_responses=[FakeResponse([{"id": "run-1"}])], post_responses=[])
    repository = make_repository(session)

    run_id = repository.create_model_run(
        model_name="baseline",
        model_version="v1",
        feature_set="technical_v1",
        label_method="triple_barrier",
        horizon=5,
    )

    assert run_id == "run-1"
    assert session.post_calls == []


def test_create_model_run_posts_payload_when_missing() -> None:
    session = FakeSession(
        get_responses=[FakeResponse([])],
        post_responses=[FakeResponse([{"id": "run-2"}])],
    )
    repository = make_repository(session)

    run_id = repository.create_model_run(
        model_name="baseline",
        model_version="v2",
        feature_set="technical_v1",
        label_method="fixed_horizon",
        horizon=10,
        params={"splits": 3},
        metrics={"summary": {"mean_f1_macro": 0.5}},
        artifact_uri="models/baseline.joblib",
    )

    assert run_id == "run-2"
    payload = session.post_calls[0]["json"]
    assert payload["model_name"] == "baseline"
    assert payload["model_version"] == "v2"
    assert payload["metrics"]["summary"]["mean_f1_macro"] == 0.5
    assert session.post_calls[0]["headers"]["Prefer"] == "return=representation"


def test_get_model_run_returns_matching_run() -> None:
    session = FakeSession(
        get_responses=[
            FakeResponse(
                [
                    {
                        "id": "run-1",
                        "model_name": "baseline",
                        "model_version": "v1",
                        "feature_set": "technical_v1",
                        "artifact_uri": "models/model.joblib",
                    }
                ]
            )
        ],
        post_responses=[],
    )
    repository = make_repository(session)

    model_run = repository.get_model_run("baseline", "v1")

    assert model_run["id"] == "run-1"
    assert session.get_calls[0]["params"]["select"] == "*"


def test_upsert_predictions_stores_probabilities_and_metadata() -> None:
    session = FakeSession(get_responses=[], post_responses=[FakeResponse()])
    repository = make_repository(session)
    predictions = pd.DataFrame(
        {
            "timestamp": pd.date_range("2024-01-01", periods=1, freq="D", tz="UTC"),
            "action": ["BUY"],
            "confidence": [0.72],
            "expected_return": [None],
            "expected_risk": [None],
            "probabilities": [{"BUY": 0.72, "HOLD": 0.2, "SELL": 0.08}],
            "metadata": [{"raw_action": "BUY"}],
        }
    )

    inserted = repository.upsert_predictions("asset-1", "run-1", predictions)

    assert inserted == 1
    payload = session.post_calls[0]["json"][0]
    assert payload["asset_id"] == "asset-1"
    assert payload["model_run_id"] == "run-1"
    assert payload["action"] == "BUY"
    assert payload["probabilities"]["BUY"] == 0.72
    assert payload["metadata"]["raw_action"] == "BUY"


def test_get_prediction_feedback_filters_view() -> None:
    session = FakeSession(
        get_responses=[
            FakeResponse(
                [
                    {
                        "prediction_id": 1,
                        "model_name": "baseline",
                        "model_version": "v1",
                        "predicted_action": "BUY",
                        "actual_label": "BUY",
                        "is_correct": True,
                    }
                ]
            )
        ],
        post_responses=[],
    )
    repository = make_repository(session)

    feedback = repository.get_prediction_feedback(
        model_name="baseline",
        model_version="v1",
        asset_id="asset-1",
        limit=25,
    )

    assert len(feedback) == 1
    params = session.get_calls[0]["params"]
    assert params["model_name"] == "eq.baseline"
    assert params["model_version"] == "eq.v1"
    assert params["asset_id"] == "eq.asset-1"
    assert params["actual_label"] == "not.is.null"
    assert params["limit"] == "25"


def test_create_backtest_posts_summary_payload() -> None:
    session = FakeSession(
        get_responses=[],
        post_responses=[FakeResponse([{"id": "backtest-1"}])],
    )
    repository = make_repository(session)

    backtest_id = repository.create_backtest(
        name="baseline:v1:AAPL",
        model_run_id="run-1",
        asset_id="asset-1",
        metrics={"total_return": 0.12},
        params={"fee_bps": 5},
        started_at="2024-01-01T00:00:00+00:00",
        ended_at="2024-01-10T00:00:00+00:00",
    )

    assert backtest_id == "backtest-1"
    payload = session.post_calls[0]["json"]
    assert payload["name"] == "baseline:v1:AAPL"
    assert payload["metrics"]["total_return"] == 0.12
    assert session.post_calls[0]["headers"]["Prefer"] == "return=representation"


def test_insert_backtest_trades_batches_payload() -> None:
    session = FakeSession(get_responses=[], post_responses=[FakeResponse()])
    repository = make_repository(session)
    trades = pd.DataFrame(
        {
            "timestamp": pd.date_range("2024-01-01", periods=1, freq="D", tz="UTC"),
            "action": ["BUY"],
            "confidence": [0.8],
            "gross_return": [0.03],
            "net_return": [0.028],
            "cost": [0.002],
            "equity": [1028],
            "metadata": [{"actual_label": "BUY"}],
        }
    )

    inserted = repository.insert_backtest_trades("backtest-1", "asset-1", trades)

    assert inserted == 1
    payload = session.post_calls[0]["json"][0]
    assert payload["backtest_id"] == "backtest-1"
    assert payload["asset_id"] == "asset-1"
    assert payload["net_return"] == 0.028
    assert payload["metadata"]["actual_label"] == "BUY"
