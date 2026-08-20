from chp.schema.context_manifest import ContextManifest, ContextRequirements
from chp.schema.rationale_envelope import RationaleEnvelope, AnnotatedChunk
from chp.templates import ManifestTemplates
from chp.inference import infer_manifest
from chp.observability import set_metrics_hook, CHPEvent
from chp.ledger.base import LedgerBackend
from chp.ledger.lancedb_ledger import LanceDBLedger, CHPLedger, WriteLimiter
from chp.ledger.sqlite_ledger import SQLiteLedger
from chp.ledger.memory_ledger import InMemoryLedger
from chp.ledger.locks import FileLockBackend, RedisLockBackend, NoOpLockBackend, AsyncRedisLockBackend
from chp.pii import RegexPIIFilter, PresidioPIIFilter, PIIFilter, set_pii_filter, get_pii_filter

# Optional backends — imported lazily so missing drivers don't break the package.
# Access via chp.PostgresLedger, chp.DynamoDBLedger, chp.MongoDBLedger after
# installing the relevant extra: pip install "chp[postgres|dynamodb|mongodb]"
def __getattr__(name: str):
    if name == "PostgresLedger":
        from chp.ledger.postgres_ledger import PostgresLedger
        return PostgresLedger
    if name == "DynamoDBLedger":
        from chp.ledger.dynamodb_ledger import DynamoDBLedger
        return DynamoDBLedger
    if name == "MongoDBLedger":
        from chp.ledger.mongodb_ledger import MongoDBLedger
        return MongoDBLedger
    raise AttributeError(f"module 'chp' has no attribute {name!r}")

__all__ = [
    # Schema
    "ContextManifest", "ContextRequirements",
    "RationaleEnvelope", "AnnotatedChunk",
    # Templates + inference
    "ManifestTemplates", "infer_manifest",
    # Observability
    "set_metrics_hook", "CHPEvent",
    # Ledger backends (always available)
    "LedgerBackend",
    "LanceDBLedger", "CHPLedger",   # CHPLedger = backward-compat alias
    "SQLiteLedger",
    "InMemoryLedger",
    # Ledger backends (optional — require extra install)
    "PostgresLedger",   # pip install "chp[postgres]"
    "DynamoDBLedger",   # pip install "chp[dynamodb]"
    "MongoDBLedger",    # pip install "chp[mongodb]"
    # Rate limiting
    "WriteLimiter",
    # Lock backends
    "FileLockBackend", "RedisLockBackend", "NoOpLockBackend", "AsyncRedisLockBackend",
    # PII filtering
    "PIIFilter", "RegexPIIFilter", "PresidioPIIFilter",
    "set_pii_filter", "get_pii_filter",
]
