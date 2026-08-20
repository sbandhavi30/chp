"""
DynamoDBLedger — CHP ledger backend backed by AWS DynamoDB via boto3.

Install:
    pip install "chp[dynamodb]"
    # which pulls in: boto3

Two-table design (mirrors SQLiteLedger / LanceDBLedger):
  {prefix}_chunks  — content store keyed on chunk_id (S)
  {prefix}_ledger  — provenance index keyed on session_id (S) + ledger_id (S)
                     with two GSIs for efficient agent and hop queries

GSIs:
  session_agent_idx  — HASH: session_id, RANGE: to_agent
  session_hop_idx    — HASH: session_id, RANGE: hop_number (N)

Usage — real AWS (credentials from environment / IAM role / ~/.aws):
    from chp.ledger.dynamodb_ledger import DynamoDBLedger

    ledger = DynamoDBLedger(table_prefix="chp", region_name="us-east-1")
    ledger.write(session_id, hop_number, from_agent, to_agent, envelope)
    envelopes = ledger.query(session_id)

Usage — LocalStack / DynamoDB Local (for tests and CI):
    import boto3
    session = boto3.Session(
        aws_access_key_id="test",
        aws_secret_access_key="test",
        region_name="us-east-1",
    )
    ledger = DynamoDBLedger(
        table_prefix="chp_test",
        endpoint_url="http://localhost:4566",   # LocalStack default
        boto3_session=session,
    )

Required IAM permissions:
    dynamodb:CreateTable
    dynamodb:DescribeTable
    dynamodb:PutItem
    dynamodb:GetItem
    dynamodb:Query
    dynamodb:Scan
    dynamodb:DeleteItem

Thread safety: threading.Lock() guards all writes.
Multi-node: DynamoDB is globally consistent for single-item reads/writes;
            the conditional put_item on chunks prevents duplicate chunk writes
            across concurrent writers.
"""
from __future__ import annotations

import json
import re as _re
import threading
import uuid
from datetime import datetime, timezone
from decimal import Decimal
from typing import TYPE_CHECKING

from chp.schema.rationale_envelope import RationaleEnvelope
from chp.ledger.base import LedgerBackend
from chp.observability import CHPEvent, emit, Timer

if TYPE_CHECKING:
    from chp.engine.embedder import Embedder


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _safe_id(value: str, field: str = "id") -> str:
    """Validate that an ID contains only safe characters."""
    if not isinstance(value, str) or not value:
        raise ValueError(f"CHP: {field} must be a non-empty string")
    if not _re.match(r'^[a-zA-Z0-9_\-.:]+$', value):
        raise ValueError(
            f"CHP: {field}={value!r} contains invalid characters. "
            "Only alphanumeric, hyphen, underscore, dot, and colon are allowed."
        )
    return value


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


# ---------------------------------------------------------------------------
# DynamoDBLedger
# ---------------------------------------------------------------------------

class DynamoDBLedger(LedgerBackend):
    """
    CHP ledger backend backed by AWS DynamoDB.

    Parameters
    ----------
    table_prefix:
        Prefix for both DynamoDB tables.  Tables created are
        ``{table_prefix}_ledger`` and ``{table_prefix}_chunks``.
    region_name:
        AWS region.  None → boto3 default (AWS_DEFAULT_REGION env var or
        ~/.aws/config).
    endpoint_url:
        Override service endpoint.  Set to ``http://localhost:4566`` for
        LocalStack or ``http://localhost:8000`` for DynamoDB Local.
    boto3_session:
        Pre-configured :class:`boto3.Session`.  Useful for assuming roles or
        injecting test credentials.  If None a fresh ``boto3.Session()`` is
        created.
    """

    def __init__(
        self,
        table_prefix: str = "chp",
        region_name: str | None = None,
        endpoint_url: str | None = None,
        boto3_session=None,
    ) -> None:
        try:
            import boto3
            from boto3.dynamodb.conditions import Key, Attr  # noqa: F401 — validate import
        except ImportError as exc:
            raise ImportError(
                "boto3 is required for DynamoDBLedger. "
                'Install it with: pip install "chp[dynamodb]"'
            ) from exc

        self._table_prefix = table_prefix
        self._ledger_table_name = f"{table_prefix}_ledger"
        self._chunks_table_name = f"{table_prefix}_chunks"
        self._lock = threading.Lock()

        session = boto3_session if boto3_session is not None else boto3.Session()

        kwargs: dict = {}
        if region_name is not None:
            kwargs["region_name"] = region_name
        if endpoint_url is not None:
            kwargs["endpoint_url"] = endpoint_url

        self._dynamodb = session.resource("dynamodb", **kwargs)
        self._client = session.client("dynamodb", **kwargs)

        self._chunks_table = self._ensure_chunks_table()
        self._ledger_table = self._ensure_ledger_table()

    # ── Table provisioning ─────────────────────────────────────────────────────

    def _ensure_chunks_table(self):
        """Create {prefix}_chunks if it does not exist; return the Table resource."""
        try:
            table = self._dynamodb.create_table(
                TableName=self._chunks_table_name,
                KeySchema=[
                    {"AttributeName": "chunk_id", "KeyType": "HASH"},
                ],
                AttributeDefinitions=[
                    {"AttributeName": "chunk_id", "AttributeType": "S"},
                ],
                BillingMode="PAY_PER_REQUEST",
            )
            table.wait_until_exists()
        except self._client.exceptions.ResourceInUseException:
            # Table already exists — just grab a reference.
            table = self._dynamodb.Table(self._chunks_table_name)
        return table

    def _ensure_ledger_table(self):
        """Create {prefix}_ledger with GSIs if it does not exist; return the Table resource."""
        try:
            table = self._dynamodb.create_table(
                TableName=self._ledger_table_name,
                KeySchema=[
                    {"AttributeName": "session_id", "KeyType": "HASH"},
                    {"AttributeName": "ledger_id",  "KeyType": "RANGE"},
                ],
                AttributeDefinitions=[
                    {"AttributeName": "session_id",  "AttributeType": "S"},
                    {"AttributeName": "ledger_id",   "AttributeType": "S"},
                    {"AttributeName": "to_agent",    "AttributeType": "S"},
                    {"AttributeName": "hop_number",  "AttributeType": "N"},
                ],
                GlobalSecondaryIndexes=[
                    {
                        "IndexName": "session_agent_idx",
                        "KeySchema": [
                            {"AttributeName": "session_id", "KeyType": "HASH"},
                            {"AttributeName": "to_agent",   "KeyType": "RANGE"},
                        ],
                        "Projection": {"ProjectionType": "ALL"},
                    },
                    {
                        "IndexName": "session_hop_idx",
                        "KeySchema": [
                            {"AttributeName": "session_id", "KeyType": "HASH"},
                            {"AttributeName": "hop_number", "KeyType": "RANGE"},
                        ],
                        "Projection": {"ProjectionType": "ALL"},
                    },
                ],
                BillingMode="PAY_PER_REQUEST",
            )
            table.wait_until_exists()
        except self._client.exceptions.ResourceInUseException:
            table = self._dynamodb.Table(self._ledger_table_name)
        return table

    # ── Write ──────────────────────────────────────────────────────────────────

    def write(
        self,
        session_id: str,
        hop_number: int,
        from_agent: str,
        to_agent: str,
        envelope: RationaleEnvelope,
        embedder: "Embedder | None" = None,
    ) -> str:
        session_id = _safe_id(session_id, "session_id")
        from_agent = _safe_id(from_agent, "from_agent")
        to_agent   = _safe_id(to_agent,   "to_agent")
        _safe_id(envelope.chunk_id, "chunk_id")

        from boto3.dynamodb.conditions import Key, Attr

        with Timer() as t:
            with self._lock:
                # ── Duplicate check ──────────────────────────────────────────
                # Query ledger by session_id (partition key) then filter by
                # chunk_id + hop_number to match SQLiteLedger's UNIQUE constraint.
                dup_response = self._ledger_table.query(
                    KeyConditionExpression=Key("session_id").eq(session_id),
                    FilterExpression=(
                        Attr("chunk_id").eq(envelope.chunk_id)
                        & Attr("hop_number").eq(Decimal(hop_number))
                    ),
                    Limit=1,
                )
                if dup_response.get("Count", 0) > 0:
                    raise RuntimeError(
                        f"duplicate entry: session={session_id} "
                        f"chunk={envelope.chunk_id} hop={hop_number}"
                    )

                # ── Chunk upsert (content-dedup) ─────────────────────────────
                # Only write if chunk_id does not already exist.
                try:
                    self._chunks_table.put_item(
                        Item={
                            "chunk_id":     envelope.chunk_id,
                            "content":      envelope.content,
                            "source_agent": envelope.source_agent,
                            "token_cost":   Decimal(envelope.token_cost),
                            "created_at":   _now_iso(),
                        },
                        ConditionExpression="attribute_not_exists(chunk_id)",
                    )
                except self._dynamodb.meta.client.exceptions.ConditionalCheckFailedException:
                    # Chunk already stored — this is fine, content is immutable.
                    pass

                # ── Ledger write ─────────────────────────────────────────────
                ledger_id = str(uuid.uuid4())
                self._ledger_table.put_item(
                    Item={
                        "ledger_id":       ledger_id,
                        "session_id":      session_id,
                        "hop_number":      Decimal(hop_number),
                        "from_agent":      from_agent,
                        "to_agent":        to_agent,
                        "chunk_id":        envelope.chunk_id,
                        "score":           str(envelope.score),   # stored as string to avoid Decimal precision issues
                        "must_carry":      envelope.must_carry,
                        "hop_sequence":    json.dumps(envelope.hop_sequence),
                        "selected_because": json.dumps(envelope.selected_because),
                        "source_turn":     Decimal(envelope.source_turn),
                        "timestamp":       _now_iso(),
                    },
                )

        emit(CHPEvent.LEDGER_WRITE, {
            "session_id":  session_id,
            "hop_number":  hop_number,
            "from_agent":  from_agent,
            "to_agent":    to_agent,
            "chunk_id":    envelope.chunk_id,
            "elapsed_ms":  round(t.elapsed_ms, 2),
        })
        return ledger_id

    # ── Query ──────────────────────────────────────────────────────────────────

    def query(
        self,
        session_id: str,
        agent_id: str | None = None,
    ) -> list[RationaleEnvelope]:
        session_id = _safe_id(session_id, "session_id")

        from boto3.dynamodb.conditions import Key

        with Timer() as t:
            if agent_id:
                agent_id = _safe_id(agent_id, "agent_id")
                response = self._ledger_table.query(
                    IndexName="session_agent_idx",
                    KeyConditionExpression=(
                        Key("session_id").eq(session_id)
                        & Key("to_agent").eq(agent_id)
                    ),
                )
            else:
                response = self._ledger_table.query(
                    KeyConditionExpression=Key("session_id").eq(session_id),
                )

            ledger_items = response.get("Items", [])
            # Handle pagination
            while "LastEvaluatedKey" in response:
                if agent_id:
                    response = self._ledger_table.query(
                        IndexName="session_agent_idx",
                        KeyConditionExpression=(
                            Key("session_id").eq(session_id)
                            & Key("to_agent").eq(agent_id)
                        ),
                        ExclusiveStartKey=response["LastEvaluatedKey"],
                    )
                else:
                    response = self._ledger_table.query(
                        KeyConditionExpression=Key("session_id").eq(session_id),
                        ExclusiveStartKey=response["LastEvaluatedKey"],
                    )
                ledger_items.extend(response.get("Items", []))

            result = []
            for ledger_item in ledger_items:
                chunk_response = self._chunks_table.get_item(
                    Key={"chunk_id": ledger_item["chunk_id"]}
                )
                chunk_item = chunk_response.get("Item")
                if chunk_item is None:
                    # Orphan ledger row — skip gracefully
                    continue
                result.append(self._hydrate(ledger_item, chunk_item))

        emit(CHPEvent.LEDGER_QUERY, {
            "session_id":    session_id,
            "agent_id":      agent_id,
            "rows_returned": len(result),
            "elapsed_ms":    round(t.elapsed_ms, 2),
        })
        return result

    def query_hop(
        self,
        session_id: str,
        hop_number: int,
    ) -> list[RationaleEnvelope]:
        session_id = _safe_id(session_id, "session_id")

        from boto3.dynamodb.conditions import Key

        response = self._ledger_table.query(
            IndexName="session_hop_idx",
            KeyConditionExpression=(
                Key("session_id").eq(session_id)
                & Key("hop_number").eq(Decimal(hop_number))
            ),
        )
        ledger_items = response.get("Items", [])
        while "LastEvaluatedKey" in response:
            response = self._ledger_table.query(
                IndexName="session_hop_idx",
                KeyConditionExpression=(
                    Key("session_id").eq(session_id)
                    & Key("hop_number").eq(Decimal(hop_number))
                ),
                ExclusiveStartKey=response["LastEvaluatedKey"],
            )
            ledger_items.extend(response.get("Items", []))

        result = []
        for ledger_item in ledger_items:
            chunk_response = self._chunks_table.get_item(
                Key={"chunk_id": ledger_item["chunk_id"]}
            )
            chunk_item = chunk_response.get("Item")
            if chunk_item is None:
                continue
            result.append(self._hydrate(ledger_item, chunk_item))
        return result

    # query_by_meaning not supported — base returns [] automatically

    # ── Maintenance ────────────────────────────────────────────────────────────

    def prune(self, session_id: str) -> int:
        """Delete all ledger rows for a session. Returns count deleted."""
        session_id = _safe_id(session_id, "session_id")

        from boto3.dynamodb.conditions import Key

        deleted = 0
        with self._lock:
            response = self._ledger_table.query(
                KeyConditionExpression=Key("session_id").eq(session_id),
                ProjectionExpression="session_id, ledger_id",
            )
            items = response.get("Items", [])
            while "LastEvaluatedKey" in response:
                response = self._ledger_table.query(
                    KeyConditionExpression=Key("session_id").eq(session_id),
                    ProjectionExpression="session_id, ledger_id",
                    ExclusiveStartKey=response["LastEvaluatedKey"],
                )
                items.extend(response.get("Items", []))

            # Batch delete in groups of 25 (DynamoDB batch_writer limit)
            with self._ledger_table.batch_writer() as batch:
                for item in items:
                    batch.delete_item(
                        Key={
                            "session_id": item["session_id"],
                            "ledger_id":  item["ledger_id"],
                        }
                    )
                    deleted += 1

        return deleted

    def prune_older_than(self, cutoff_iso: str) -> int:
        """Delete all ledger rows with timestamp < cutoff_iso. Returns count deleted."""
        from boto3.dynamodb.conditions import Attr

        deleted = 0
        with self._lock:
            # Full table scan with timestamp filter
            scan_kwargs: dict = {
                "FilterExpression": Attr("timestamp").lt(cutoff_iso),
                "ProjectionExpression": "session_id, ledger_id",
            }
            items: list = []
            response = self._ledger_table.scan(**scan_kwargs)
            items.extend(response.get("Items", []))
            while "LastEvaluatedKey" in response:
                response = self._ledger_table.scan(
                    **scan_kwargs,
                    ExclusiveStartKey=response["LastEvaluatedKey"],
                )
                items.extend(response.get("Items", []))

            with self._ledger_table.batch_writer() as batch:
                for item in items:
                    batch.delete_item(
                        Key={
                            "session_id": item["session_id"],
                            "ledger_id":  item["ledger_id"],
                        }
                    )
                    deleted += 1

        return deleted

    def prune_orphan_chunks(self) -> int:
        """Delete chunk rows that are not referenced by any ledger row. Returns count deleted."""
        # Collect all chunk_ids referenced in the ledger
        referenced_ids: set[str] = set()
        response = self._ledger_table.scan(
            ProjectionExpression="chunk_id",
        )
        for item in response.get("Items", []):
            referenced_ids.add(item["chunk_id"])
        while "LastEvaluatedKey" in response:
            response = self._ledger_table.scan(
                ProjectionExpression="chunk_id",
                ExclusiveStartKey=response["LastEvaluatedKey"],
            )
            for item in response.get("Items", []):
                referenced_ids.add(item["chunk_id"])

        # Collect all chunk_ids in the chunk store
        all_chunks: list[str] = []
        response = self._chunks_table.scan(
            ProjectionExpression="chunk_id",
        )
        for item in response.get("Items", []):
            all_chunks.append(item["chunk_id"])
        while "LastEvaluatedKey" in response:
            response = self._chunks_table.scan(
                ProjectionExpression="chunk_id",
                ExclusiveStartKey=response["LastEvaluatedKey"],
            )
            for item in response.get("Items", []):
                all_chunks.append(item["chunk_id"])

        orphans = [cid for cid in all_chunks if cid not in referenced_ids]

        deleted = 0
        with self._lock:
            with self._chunks_table.batch_writer() as batch:
                for chunk_id in orphans:
                    batch.delete_item(Key={"chunk_id": chunk_id})
                    deleted += 1

        return deleted

    def compact(self) -> None:
        """No-op: DynamoDB manages storage and indexes automatically."""

    # ── Stats ──────────────────────────────────────────────────────────────────

    def stats(self) -> dict:
        """Return row counts for ledger and chunk tables."""
        ledger_response = self._ledger_table.scan(Select="COUNT")
        # Accumulate across paginated scan counts
        ledger_rows = ledger_response.get("Count", 0)
        while "LastEvaluatedKey" in ledger_response:
            ledger_response = self._ledger_table.scan(
                Select="COUNT",
                ExclusiveStartKey=ledger_response["LastEvaluatedKey"],
            )
            ledger_rows += ledger_response.get("Count", 0)

        chunk_response = self._chunks_table.scan(Select="COUNT")
        chunk_rows = chunk_response.get("Count", 0)
        while "LastEvaluatedKey" in chunk_response:
            chunk_response = self._chunks_table.scan(
                Select="COUNT",
                ExclusiveStartKey=chunk_response["LastEvaluatedKey"],
            )
            chunk_rows += chunk_response.get("Count", 0)

        return {"ledger_rows": ledger_rows, "chunk_rows": chunk_rows}

    # ── Internal ───────────────────────────────────────────────────────────────

    def _hydrate(self, ledger_item: dict, chunk_item: dict) -> RationaleEnvelope:
        """Reconstruct a RationaleEnvelope from a DynamoDB ledger row and its chunk row."""
        return RationaleEnvelope(
            chunk_id=chunk_item["chunk_id"],
            content=chunk_item["content"],
            source_agent=chunk_item["source_agent"],
            source_turn=int(ledger_item.get("source_turn", 0)),
            hop_sequence=json.loads(ledger_item.get("hop_sequence", "[]")),
            selected_because=json.loads(ledger_item.get("selected_because", "[]")),
            score=float(ledger_item["score"]),
            must_carry=bool(ledger_item["must_carry"]),
            token_cost=int(chunk_item["token_cost"]),
            ledger_id=ledger_item["ledger_id"],
        )
