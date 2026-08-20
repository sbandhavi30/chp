from datetime import datetime, timezone
from chp.schema.context_manifest import ContextManifest, ContextRequirements
from chp.schema.rationale_envelope import AnnotatedChunk
from chp.engine.embedder import StubEmbedder
from chp.ledger.lancedb_ledger import CHPLedger
from chp.adapters.crewai import CHPCrewTask


def test_chp_crew_task_filters_context():
    manifest = ContextManifest(
        chp_version="0.1", agent_id="researcher", task="summarize",
        requires=ContextRequirements(
            must_carry=["user_id"], domain_tags=["research"],
            history_depth="full", recency_window=None, exclude=["debug"],
        ),
        token_budget=300, on_missing="warn",
    )
    chunks = [
        AnnotatedChunk(chunk_id="c1", content="user_id is 7", token_cost=30,
                       source_agent="planner", source_turn=1,
                       timestamp=datetime(2026, 8, 19, tzinfo=timezone.utc)),
        AnnotatedChunk(chunk_id="c2", content="debug: verbose log", token_cost=30,
                       source_agent="planner", source_turn=2,
                       timestamp=datetime(2026, 8, 19, tzinfo=timezone.utc)),
    ]

    received = {}

    def my_task(context):
        received["context"] = context
        return "done"

    task = CHPCrewTask(my_task, manifest=manifest, ledger=CHPLedger(), embedder=StubEmbedder())
    task.run(chunks=chunks, session_id="sess_crew", hop=0)

    ids = [c.chunk_id for c in received["context"]]
    assert "c1" in ids
    assert "c2" not in ids
