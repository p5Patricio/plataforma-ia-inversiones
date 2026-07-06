from __future__ import annotations

from collector.schema_check import check_relations
from collector.supabase_repository import SupabaseConfig, SupabaseRepository


class FakeResponse:
    def __init__(self, status_code: int, text: str = "") -> None:
        self.status_code = status_code
        self.text = text


class FakeSession:
    def __init__(self, statuses: dict[str, int]) -> None:
        self.statuses = statuses
        self.requested: list[str] = []

    def get(self, url: str, headers: dict, params: dict, timeout: int) -> FakeResponse:
        relation = url.rsplit("/", maxsplit=1)[-1]
        self.requested.append(relation)
        status = self.statuses.get(relation, 200)
        return FakeResponse(status, text="missing relation" if status >= 400 else "")


def make_repository(session: FakeSession) -> SupabaseRepository:
    return SupabaseRepository(SupabaseConfig(url="https://example.supabase.co", key="test-key"), session=session)


def test_check_relations_marks_available_tables() -> None:
    session = FakeSession({"features_daily": 200, "prediction_feedback": 200})
    repository = make_repository(session)

    statuses = check_relations(repository, relations=("features_daily", "prediction_feedback"))

    assert [status.available for status in statuses] == [True, True]
    assert session.requested == ["features_daily", "prediction_feedback"]


def test_check_relations_reports_missing_relation() -> None:
    session = FakeSession({"features_daily": 200, "model_runs": 404})
    repository = make_repository(session)

    statuses = check_relations(repository, relations=("features_daily", "model_runs"))

    assert statuses[0].available is True
    assert statuses[1].available is False
    assert statuses[1].status_code == 404
    assert statuses[1].error == "missing relation"
