from __future__ import annotations

import os
from dataclasses import dataclass
from datetime import datetime
from typing import Any

import pandas as pd
import requests
from dotenv import load_dotenv


@dataclass(frozen=True)
class SupabaseConfig:
    url: str
    key: str

    @classmethod
    def from_env(cls) -> "SupabaseConfig":
        load_dotenv()
        url = os.getenv("SUPABASE_URL")
        key = os.getenv("SUPABASE_KEY")
        if not url or not key:
            raise RuntimeError("SUPABASE_URL and SUPABASE_KEY must be set")
        return cls(url=url.rstrip("/"), key=key)


@dataclass
class SupabaseRepository:
    config: SupabaseConfig
    session: requests.Session | None = None

    def __post_init__(self) -> None:
        self._session = self.session or requests.Session()

    @property
    def headers(self) -> dict[str, str]:
        return {
            "apikey": self.config.key,
            "Authorization": f"Bearer {self.config.key}",
            "Content-Type": "application/json",
        }

    def get_assets(self) -> list[dict[str, Any]]:
        response = self._session.get(
            f"{self.config.url}/rest/v1/assets",
            headers=self.headers,
            params={"select": "*", "order": "ticker.asc"},
            timeout=30,
        )
        response.raise_for_status()
        return response.json()

    def get_auth_user(self, access_token: str) -> dict[str, Any]:
        response = self._session.get(
            f"{self.config.url}/auth/v1/user",
            headers={
                "apikey": self.config.key,
                "Authorization": f"Bearer {access_token}",
            },
            timeout=30,
        )
        response.raise_for_status()
        user = response.json()
        if not user.get("id"):
            raise RuntimeError("Supabase Auth did not return a user id")
        return user

    def get_or_create_asset(
        self,
        ticker: str,
        name: str | None = None,
        asset_class: str | None = None,
    ) -> str:
        normalized_ticker = ticker.upper()
        lookup_url = f"{self.config.url}/rest/v1/assets"
        lookup = self._session.get(
            lookup_url,
            headers=self.headers,
            params={"ticker": f"eq.{normalized_ticker}", "select": "id"},
            timeout=30,
        )
        lookup.raise_for_status()
        existing = lookup.json()
        if existing:
            return existing[0]["id"]

        create_headers = self.headers | {"Prefer": "return=representation"}
        payload = {
            "ticker": normalized_ticker,
            "name": name or normalized_ticker,
            "asset_class": asset_class or "unknown",
        }
        created = self._session.post(
            lookup_url,
            headers=create_headers,
            json=payload,
            timeout=30,
        )
        created.raise_for_status()
        data = created.json()
        if not data:
            raise RuntimeError(f"Supabase did not return created asset for {normalized_ticker}")
        return data[0]["id"]

    def get_asset_id(self, ticker: str) -> str:
        normalized_ticker = ticker.upper()
        url = f"{self.config.url}/rest/v1/assets"
        response = self._session.get(
            url,
            headers=self.headers,
            params={"ticker": f"eq.{normalized_ticker}", "select": "id"},
            timeout=30,
        )
        response.raise_for_status()
        data = response.json()
        if not data:
            raise ValueError(f"Asset not found in Supabase: {normalized_ticker}")
        return data[0]["id"]

    def get_asset(self, ticker: str) -> dict[str, Any]:
        normalized_ticker = ticker.upper()
        url = f"{self.config.url}/rest/v1/assets"
        response = self._session.get(
            url,
            headers=self.headers,
            params={"ticker": f"eq.{normalized_ticker}", "select": "id,ticker,name,asset_class"},
            timeout=30,
        )
        response.raise_for_status()
        data = response.json()
        if not data:
            raise ValueError(f"Asset not found in Supabase: {normalized_ticker}")
        return data[0]

    def get_prices(self, asset_id: str, limit: int | None = None, ascending: bool = True) -> pd.DataFrame:
        params = {
            "asset_id": f"eq.{asset_id}",
            "select": "timestamp,open,high,low,close,volume",
            "order": "timestamp.asc" if ascending else "timestamp.desc",
        }
        return pd.DataFrame(self._get_rows("prices", params, limit=limit))

    def get_features(
        self,
        asset_id: str,
        feature_set: str,
        limit: int | None = None,
        ascending: bool = True,
    ) -> pd.DataFrame:
        params = {
            "asset_id": f"eq.{asset_id}",
            "feature_set": f"eq.{feature_set}",
            "select": "timestamp,features",
            "order": "timestamp.asc" if ascending else "timestamp.desc",
        }
        return pd.DataFrame(self._get_rows("features_daily", params, limit=limit))

    def get_labels(
        self,
        asset_id: str,
        label_method: str,
        horizon: int,
        limit: int | None = None,
    ) -> pd.DataFrame:
        params = {
            "asset_id": f"eq.{asset_id}",
            "label_method": f"eq.{label_method}",
            "horizon": f"eq.{horizon}",
            "select": "timestamp,label,outcome_return,label_exit_timestamp",
            "order": "timestamp.asc",
        }
        return pd.DataFrame(self._get_rows("labels_daily", params, limit=limit))

    def upsert_prices(self, asset_id: str, prices: pd.DataFrame, batch_size: int = 500) -> int:
        if prices.empty:
            return 0
        required = {"timestamp", "open", "high", "low", "close", "volume"}
        missing = required - set(prices.columns)
        if missing:
            raise ValueError(f"prices DataFrame missing columns: {sorted(missing)}")

        rows = []
        clean = prices.sort_values("timestamp").drop_duplicates("timestamp", keep="last")
        for _, row in clean.iterrows():
            rows.append(
                {
                    "asset_id": asset_id,
                    "timestamp": pd.Timestamp(row["timestamp"]).isoformat(),
                    "open": float(row["open"]),
                    "high": float(row["high"]),
                    "low": float(row["low"]),
                    "close": float(row["close"]),
                    "volume": int(row["volume"]) if pd.notna(row["volume"]) else 0,
                }
            )

        return self._post_batches(
            "prices",
            rows,
            batch_size,
            on_conflict="asset_id,timestamp",
        )

    def upsert_features(
        self,
        asset_id: str,
        features: pd.DataFrame,
        feature_columns: list[str],
        feature_set: str,
        batch_size: int = 500,
    ) -> int:
        required = {"timestamp", *feature_columns}
        missing = required - set(features.columns)
        if missing:
            raise ValueError(f"features DataFrame missing columns: {sorted(missing)}")

        clean = features.dropna(subset=feature_columns)
        rows = [
            {
                "asset_id": asset_id,
                "timestamp": pd.Timestamp(row["timestamp"]).isoformat(),
                "feature_set": feature_set,
                "features": {column: _json_value(row[column]) for column in feature_columns},
            }
            for _, row in clean.iterrows()
        ]
        return self._post_batches(
            "features_daily",
            rows,
            batch_size,
            on_conflict="asset_id,timestamp,feature_set",
        )

    def upsert_labels(
        self,
        asset_id: str,
        labels: pd.DataFrame,
        label_method: str,
        horizon: int,
        batch_size: int = 500,
    ) -> int:
        required = {"timestamp", "label"}
        missing = required - set(labels.columns)
        if missing:
            raise ValueError(f"labels DataFrame missing columns: {sorted(missing)}")

        clean = labels.dropna(subset=["label"])
        rows = []
        for _, row in clean.iterrows():
            rows.append(
                {
                    "asset_id": asset_id,
                    "timestamp": pd.Timestamp(row["timestamp"]).isoformat(),
                    "label_method": label_method,
                    "horizon": horizon,
                    "label": str(row["label"]),
                    "outcome_return": _json_value(
                        row["outcome_return"] if "outcome_return" in row else row.get("future_return")
                    ),
                    "label_exit_timestamp": _timestamp_or_none(row.get("label_exit_timestamp")),
                    "metadata": {},
                }
            )
        return self._post_batches(
            "labels_daily",
            rows,
            batch_size,
            on_conflict="asset_id,timestamp,label_method,horizon",
        )

    def create_model_run(
        self,
        model_name: str,
        model_version: str,
        feature_set: str,
        label_method: str,
        horizon: int,
        train_start: str | datetime | None = None,
        train_end: str | datetime | None = None,
        params: dict[str, Any] | None = None,
        metrics: dict[str, Any] | None = None,
        artifact_uri: str | None = None,
    ) -> str:
        existing = self._session.get(
            f"{self.config.url}/rest/v1/model_runs",
            headers=self.headers,
            params={
                "model_name": f"eq.{model_name}",
                "model_version": f"eq.{model_version}",
                "select": "id",
            },
            timeout=30,
        )
        existing.raise_for_status()
        data = existing.json()
        if data:
            return data[0]["id"]

        payload = {
            "model_name": model_name,
            "model_version": model_version,
            "feature_set": feature_set,
            "label_method": label_method,
            "horizon": horizon,
            "train_start": _timestamp_or_none(train_start),
            "train_end": _timestamp_or_none(train_end),
            "params": _json_safe(params or {}),
            "metrics": _json_safe(metrics or {}),
            "artifact_uri": artifact_uri,
        }
        created = self._session.post(
            f"{self.config.url}/rest/v1/model_runs",
            headers=self.headers | {"Prefer": "return=representation"},
            json=payload,
            timeout=30,
        )
        created.raise_for_status()
        created_data = created.json()
        if not created_data:
            raise RuntimeError(f"Supabase did not return created model run for {model_name}:{model_version}")
        return created_data[0]["id"]

    def get_model_run(self, model_name: str, model_version: str) -> dict[str, Any]:
        response = self._session.get(
            f"{self.config.url}/rest/v1/model_runs",
            headers=self.headers,
            params={
                "model_name": f"eq.{model_name}",
                "model_version": f"eq.{model_version}",
                "select": "*",
            },
            timeout=30,
        )
        response.raise_for_status()
        data = response.json()
        if not data:
            raise ValueError(f"Model run not found: {model_name}:{model_version}")
        return data[0]

    def get_model_runs(
        self,
        model_name: str | None = None,
        model_version: str | None = None,
        limit: int | None = None,
        ascending: bool = False,
    ) -> list[dict[str, Any]]:
        params = {
            "select": "*",
            "order": "created_at.asc" if ascending else "created_at.desc",
        }
        if model_name:
            params["model_name"] = f"eq.{model_name}"
        if model_version:
            params["model_version"] = f"eq.{model_version}"
        return self._get_rows("model_runs", params, limit=limit)

    def get_default_user_risk_profile(self, user_id: str) -> dict[str, Any] | None:
        rows = self._get_rows(
            "user_risk_profiles",
            {
                "user_id": f"eq.{user_id}",
                "is_default": "eq.true",
                "select": "*",
                "order": "updated_at.desc",
            },
            limit=1,
        )
        return rows[0] if rows else None

    def get_scoped_user_risk_profile(
        self,
        user_id: str,
        scope_type: str,
        scope_value: str = "",
    ) -> dict[str, Any] | None:
        rows = self._get_rows(
            "user_risk_profiles",
            {
                "user_id": f"eq.{user_id}",
                "scope_type": f"eq.{scope_type}",
                "scope_value": f"eq.{scope_value}",
                "select": "*",
                "order": "updated_at.desc",
            },
            limit=1,
        )
        return rows[0] if rows else None

    def get_user_risk_profile_for_asset(
        self,
        user_id: str,
        ticker: str | None = None,
        asset_class: str | None = None,
    ) -> dict[str, Any] | None:
        if ticker:
            profile = self.get_scoped_user_risk_profile(user_id, "ticker", ticker.upper())
            if profile:
                return profile
        if asset_class:
            profile = self.get_scoped_user_risk_profile(user_id, "asset_class", asset_class.lower())
            if profile:
                return profile
        return self.get_default_user_risk_profile(user_id)

    def upsert_default_user_risk_profile(
        self,
        user_id: str,
        profile: dict[str, Any],
    ) -> dict[str, Any]:
        return self.upsert_user_risk_profile(user_id, profile, scope_type="default", scope_value="")

    def upsert_user_risk_profile(
        self,
        user_id: str,
        profile: dict[str, Any],
        scope_type: str = "default",
        scope_value: str = "",
    ) -> dict[str, Any]:
        payload = {
            **profile,
            "user_id": user_id,
            "name": profile.get("name") or "default",
            "scope_type": scope_type,
            "scope_value": scope_value,
            "is_default": scope_type == "default",
        }
        response = self._session.post(
            f"{self.config.url}/rest/v1/user_risk_profiles",
            headers=self.headers | {"Prefer": "resolution=merge-duplicates,return=representation"},
            params={"on_conflict": "user_id,scope_type,scope_value"},
            json=payload,
            timeout=30,
        )
        response.raise_for_status()
        data = response.json()
        if not data:
            raise RuntimeError(f"Supabase did not return updated risk profile for user: {user_id}")
        return data[0]

    def update_model_run_artifact_uri(self, model_run_id: str, artifact_uri: str) -> dict[str, Any]:
        response = self._session.patch(
            f"{self.config.url}/rest/v1/model_runs",
            headers=self.headers | {"Prefer": "return=representation"},
            params={"id": f"eq.{model_run_id}"},
            json={"artifact_uri": artifact_uri},
            timeout=30,
        )
        response.raise_for_status()
        data = response.json()
        if not data:
            raise RuntimeError(f"Supabase did not return updated model run: {model_run_id}")
        return data[0]

    def upsert_predictions(
        self,
        asset_id: str,
        model_run_id: str,
        predictions: pd.DataFrame,
        batch_size: int = 500,
    ) -> int:
        required = {"timestamp", "action", "confidence", "probabilities"}
        missing = required - set(predictions.columns)
        if missing:
            raise ValueError(f"predictions DataFrame missing columns: {sorted(missing)}")

        rows = []
        for _, row in predictions.iterrows():
            rows.append(
                {
                    "asset_id": asset_id,
                    "model_run_id": model_run_id,
                    "timestamp": pd.Timestamp(row["timestamp"]).isoformat(),
                    "action": str(row["action"]),
                    "confidence": _json_value(row["confidence"]),
                    "expected_return": _json_value(row.get("expected_return")),
                    "expected_risk": _json_value(row.get("expected_risk")),
                    "probabilities": _json_safe(row["probabilities"]),
                    "metadata": _json_safe(row.get("metadata", {})),
                }
            )

        return self._post_batches(
            "predictions",
            rows,
            batch_size,
            on_conflict="asset_id,model_run_id,timestamp",
        )

    def get_prediction_feedback(
        self,
        model_name: str | None = None,
        model_version: str | None = None,
        asset_id: str | None = None,
        only_evaluated: bool = True,
        limit: int | None = None,
        ascending: bool = True,
    ) -> pd.DataFrame:
        params = {
            "select": "*",
            "order": "timestamp.asc" if ascending else "timestamp.desc",
        }
        if model_name:
            params["model_name"] = f"eq.{model_name}"
        if model_version:
            params["model_version"] = f"eq.{model_version}"
        if asset_id:
            params["asset_id"] = f"eq.{asset_id}"
        if only_evaluated:
            params["actual_label"] = "not.is.null"
        return pd.DataFrame(self._get_rows("prediction_feedback", params, limit=limit))

    def get_latest_prediction(
        self,
        asset_id: str,
        model_name: str | None = None,
        model_version: str | None = None,
    ) -> dict[str, Any] | None:
        params = {
            "asset_id": f"eq.{asset_id}",
            "select": "*",
            "order": "timestamp.desc",
            "limit": "1",
        }
        if model_name:
            params["model_name"] = f"eq.{model_name}"
        if model_version:
            params["model_version"] = f"eq.{model_version}"

        response = self._session.get(
            f"{self.config.url}/rest/v1/prediction_feedback",
            headers=self.headers,
            params=params,
            timeout=30,
        )
        response.raise_for_status()
        data = response.json()
        return data[0] if data else None

    def create_backtest(
        self,
        name: str,
        model_run_id: str | None,
        asset_id: str | None,
        metrics: dict[str, Any],
        params: dict[str, Any] | None = None,
        started_at: str | datetime | None = None,
        ended_at: str | datetime | None = None,
    ) -> str:
        payload = {
            "name": name,
            "model_run_id": model_run_id,
            "asset_id": asset_id,
            "started_at": _timestamp_or_none(started_at),
            "ended_at": _timestamp_or_none(ended_at),
            "params": _json_safe(params or {}),
            "metrics": _json_safe(metrics),
        }
        response = self._session.post(
            f"{self.config.url}/rest/v1/backtests",
            headers=self.headers | {"Prefer": "return=representation"},
            json=payload,
            timeout=30,
        )
        response.raise_for_status()
        data = response.json()
        if not data:
            raise RuntimeError(f"Supabase did not return created backtest: {name}")
        return data[0]["id"]

    def get_backtests(
        self,
        asset_id: str | None = None,
        model_run_id: str | None = None,
        limit: int | None = None,
        ascending: bool = False,
    ) -> pd.DataFrame:
        params = {
            "select": "*,model_runs(model_name,model_version,feature_set,label_method,horizon)",
            "order": "created_at.asc" if ascending else "created_at.desc",
        }
        if asset_id:
            params["asset_id"] = f"eq.{asset_id}"
        if model_run_id:
            params["model_run_id"] = f"eq.{model_run_id}"
        return pd.DataFrame(self._get_rows("backtests", params, limit=limit))

    def insert_backtest_trades(
        self,
        backtest_id: str,
        asset_id: str | None,
        trades: pd.DataFrame,
        batch_size: int = 500,
    ) -> int:
        if trades.empty:
            return 0
        required = {"timestamp", "action", "confidence", "gross_return", "net_return", "cost", "equity"}
        missing = required - set(trades.columns)
        if missing:
            raise ValueError(f"trades DataFrame missing columns: {sorted(missing)}")

        rows = []
        for _, row in trades.iterrows():
            rows.append(
                {
                    "backtest_id": backtest_id,
                    "asset_id": asset_id,
                    "timestamp": pd.Timestamp(row["timestamp"]).isoformat(),
                    "action": str(row["action"]),
                    "confidence": _json_value(row["confidence"]),
                    "gross_return": _json_value(row["gross_return"]),
                    "net_return": _json_value(row["net_return"]),
                    "cost": _json_value(row["cost"]),
                    "equity": _json_value(row["equity"]),
                    "metadata": _json_safe(row.get("metadata", {})),
                }
            )
        return self._post_batches("backtest_trades", rows, batch_size)

    def _post_batches(
        self,
        table: str,
        rows: list[dict[str, Any]],
        batch_size: int,
        on_conflict: str | None = None,
    ) -> int:
        if not rows:
            return 0

        url = f"{self.config.url}/rest/v1/{table}"
        params = {"on_conflict": on_conflict} if on_conflict else None
        headers = self.headers | {"Prefer": "resolution=merge-duplicates"}
        inserted = 0
        for start in range(0, len(rows), batch_size):
            batch = rows[start : start + batch_size]
            response = self._session.post(url, headers=headers, json=batch, params=params, timeout=30)
            response.raise_for_status()
            inserted += len(batch)
        return inserted

    def _get_rows(
        self,
        table: str,
        params: dict[str, str],
        limit: int | None = None,
        page_size: int = 1000,
    ) -> list[dict[str, Any]]:
        if limit is not None and limit <= page_size:
            response = self._session.get(
                f"{self.config.url}/rest/v1/{table}",
                headers=self.headers,
                params=params | {"limit": str(limit)},
                timeout=30,
            )
            response.raise_for_status()
            return response.json()

        rows: list[dict[str, Any]] = []
        offset = 0
        while True:
            remaining = None if limit is None else limit - len(rows)
            if remaining is not None and remaining <= 0:
                break
            current_page_size = min(page_size, remaining) if remaining is not None else page_size
            response = self._session.get(
                f"{self.config.url}/rest/v1/{table}",
                headers=self.headers | {"Range": f"{offset}-{offset + current_page_size - 1}"},
                params=params,
                timeout=30,
            )
            response.raise_for_status()
            batch = response.json()
            rows.extend(batch)
            if len(batch) < current_page_size:
                break
            offset += current_page_size
        return rows


def _json_value(value: Any) -> Any:
    if pd.isna(value):
        return None
    if hasattr(value, "item"):
        return value.item()
    return value


def _timestamp_or_none(value: Any) -> str | None:
    if value is None or pd.isna(value):
        return None
    return pd.Timestamp(value).isoformat()


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {key: _json_safe(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_json_safe(item) for item in value]
    if isinstance(value, tuple):
        return [_json_safe(item) for item in value]
    return _json_value(value)
