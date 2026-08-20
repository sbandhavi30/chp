"""Tests for chp.ledger.locks and CHPLedger lock_backend injection."""
from __future__ import annotations
import threading
import time
import pytest
from chp.ledger.locks import FileLockBackend, NoOpLockBackend, RedisLockBackend
from chp.ledger.lancedb_ledger import CHPLedger
from chp.schema.rationale_envelope import AnnotatedChunk, RationaleEnvelope


# ── helpers ───────────────────────────────────────────────────────────────────

def _env(cid: str) -> RationaleEnvelope:
    return RationaleEnvelope(
        chunk_id=cid, content=f"content {cid}",
        source_agent="a", source_turn=0,
        hop_sequence=["a", "b"], selected_because=["test"],
        score=0.5, must_carry=False, token_cost=10, ledger_id=None,
    )


# ── NoOpLockBackend ───────────────────────────────────────────────────────────

def test_noop_lock_acquires_and_releases():
    backend = NoOpLockBackend()
    with backend.acquire("key"):
        pass  # must not raise


def test_noop_lock_reentrant():
    backend = NoOpLockBackend()
    with backend.acquire("key"):
        with backend.acquire("key"):   # no deadlock
            pass


# ── FileLockBackend ───────────────────────────────────────────────────────────

def test_filelock_creates_lock_file(tmp_path):
    backend = FileLockBackend(str(tmp_path))
    with backend.acquire("chp_write"):
        lock_file = tmp_path / "chp_write.lock"
        assert lock_file.exists()


def test_filelock_mutual_exclusion_threaded(tmp_path):
    backend = FileLockBackend(str(tmp_path))
    results = []
    counter = [0]

    def worker():
        with backend.acquire("chp_write"):
            val = counter[0]
            time.sleep(0.005)          # hold lock briefly
            counter[0] = val + 1
            results.append(counter[0])

    threads = [threading.Thread(target=worker) for _ in range(10)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    # If lock works correctly, each increment is atomic → final value == 10
    assert counter[0] == 10
    assert results == list(range(1, 11))


def test_filelock_timeout_raises(tmp_path):
    backend = FileLockBackend(str(tmp_path), timeout_s=0.1)
    results = []

    def holder():
        with backend.acquire("slow_key"):
            time.sleep(0.5)

    t = threading.Thread(target=holder)
    t.start()
    time.sleep(0.02)  # let holder grab it

    try:
        with backend.acquire("slow_key", timeout_s=0.05):
            pass
    except Exception as exc:
        results.append(type(exc).__name__)
    finally:
        t.join()

    assert results, "Expected timeout exception but none raised"


# ── CHPLedger lock_backend injection ─────────────────────────────────────────

def test_default_backend_is_filelock(tmp_path):
    ledger = CHPLedger(db_path=str(tmp_path))
    assert isinstance(ledger._lock_backend, FileLockBackend)


def test_noop_backend_injection():
    ledger = CHPLedger(lock_backend=NoOpLockBackend())
    assert isinstance(ledger._lock_backend, NoOpLockBackend)


def test_custom_backend_used_on_write(tmp_path):
    acquired = []

    class TrackingBackend(NoOpLockBackend):
        from contextlib import contextmanager
        @contextmanager
        def acquire(self, key, timeout_s=-1):
            acquired.append(key)
            yield

    ledger = CHPLedger(db_path=str(tmp_path), lock_backend=TrackingBackend())
    ledger.write("sess-1", 0, "a", "b", _env("c1"))
    assert acquired, "lock backend acquire() never called during write"
    assert "chp_write" in acquired


def test_concurrent_writes_with_filelock(tmp_path):
    """10 threads × 5 writes each — no duplicate errors, all 50 rows committed."""
    ledger = CHPLedger(db_path=str(tmp_path))
    errors = []

    def write_batch(thread_idx: int):
        for i in range(5):
            try:
                cid = f"t{thread_idx}-c{i}"
                ledger.write(f"sess-concurrent", thread_idx * 10 + i, "a", "b", _env(cid))
            except RuntimeError:
                pass   # duplicate — acceptable under heavy concurrency
            except Exception as exc:
                errors.append(str(exc))

    threads = [threading.Thread(target=write_batch, args=(i,)) for i in range(10)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert not errors, f"Unexpected errors: {errors}"
    # All 50 unique (session, chunk, hop) combos should be committed
    assert ledger.stats()["ledger_rows"] == 50


# ── RedisLockBackend (mocked — no real Redis needed) ─────────────────────────

class _MockRedis:
    """Minimal Redis mock for lock tests — in-memory SET NX."""
    def __init__(self):
        self._store: dict[str, str] = {}
        self._lock = threading.Lock()
        self._scripts: dict[str, str] = {}
        self._sha_counter = 0

    def set(self, key, value, nx=False, px=None):
        with self._lock:
            if nx and key in self._store:
                return None
            self._store[key] = value
            return True

    def get(self, key):
        return self._store.get(key)

    def delete(self, key):
        self._store.pop(key, None)
        return 1

    def script_load(self, script: str) -> str:
        self._sha_counter += 1
        sha = f"sha{self._sha_counter}"
        self._scripts[sha] = script
        return sha

    def evalsha(self, sha: str, numkeys: int, *args):
        # Simplified: if value matches, delete
        key, token = args[0], args[1]
        with self._lock:
            if self._store.get(key) == token:
                del self._store[key]
                return 1
        return 0


def test_redis_lock_acquires_and_releases():
    mock = _MockRedis()
    backend = RedisLockBackend(mock, ttl_ms=5000, retry_delay_ms=10)
    with backend.acquire("test_key"):
        assert "chp:lock:test_key" in mock._store
    assert "chp:lock:test_key" not in mock._store


def test_redis_lock_mutual_exclusion():
    mock = _MockRedis()
    backend = RedisLockBackend(mock, ttl_ms=5000, retry_delay_ms=5)
    results = []
    counter = [0]

    def worker():
        with backend.acquire("shared"):
            val = counter[0]
            time.sleep(0.005)
            counter[0] = val + 1
            results.append(counter[0])

    threads = [threading.Thread(target=worker) for _ in range(8)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert counter[0] == 8
    assert results == list(range(1, 9))


def test_redis_lock_timeout_raises():
    mock = _MockRedis()
    # Pre-occupy lock so acquire must spin
    mock._store["chp:lock:busy_key"] = "someone_else"
    backend = RedisLockBackend(mock, ttl_ms=5000, retry_delay_ms=10)

    with pytest.raises(TimeoutError):
        with backend.acquire("busy_key", timeout_s=0.05):
            pass


def test_redis_lock_different_keys_independent():
    mock = _MockRedis()
    backend = RedisLockBackend(mock, ttl_ms=5000)
    # Two different keys must not block each other
    acquired = []
    with backend.acquire("key-a"):
        acquired.append("a")
        with backend.acquire("key-b"):
            acquired.append("b")
    assert acquired == ["a", "b"]


# ── Scalar index ──────────────────────────────────────────────────────────────

def test_scalar_index_created_on_init(tmp_path):
    """create_scalar_index called at init — query() works without full scan error."""
    ledger = CHPLedger(db_path=str(tmp_path))
    for i in range(10):
        ledger.write(f"sess-idx", i, "a", "b", _env(f"idx-c{i}"))
    # query() must return all 10 rows with no error
    rows = ledger.query("sess-idx")
    assert len(rows) == 10


def test_create_index_smoke(tmp_path):
    """create_index() must not raise on a populated ledger."""
    ledger = CHPLedger(db_path=str(tmp_path))
    for i in range(5):
        ledger.write("sess-rebuild", i, "a", "b", _env(f"reb-c{i}"))
    ledger.create_index(replace=True)   # must not raise
    assert ledger.stats()["ledger_rows"] == 5
