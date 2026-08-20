"""
CHP distributed lock backends.

Single-node:   FileLockBackend  (default, uses filelock on local FS)
Multi-node:    RedisLockBackend (uses redis-py + Lua SET NX for K8s/EFS deployments)
No-op:         NoOpLockBackend  (testing only — no concurrency safety)

Usage:
    # single-node (default)
    ledger = CHPLedger("/data/chp")

    # multi-node / Kubernetes
    import redis
    from chp.ledger.locks import RedisLockBackend
    backend = RedisLockBackend(redis.Redis.from_url("redis://redis-svc:6379/0"))
    ledger = CHPLedger("/data/chp", lock_backend=backend)
"""
from __future__ import annotations

import time
import uuid
from contextlib import contextmanager
from typing import Generator, Protocol, runtime_checkable


@runtime_checkable
class LockBackend(Protocol):
    """Context-manager protocol for distributed write locks."""

    @contextmanager
    def acquire(self, key: str, timeout_s: float = -1) -> Generator[None, None, None]:
        ...


# ── No-op (testing) ───────────────────────────────────────────────────────────

class NoOpLockBackend:
    """No locking — single-threaded tests only."""

    @contextmanager
    def acquire(self, key: str, timeout_s: float = -1) -> Generator[None, None, None]:
        yield


# ── File-based (single-node, default) ────────────────────────────────────────

class FileLockBackend:
    """
    OS-level file lock via `filelock`.

    Safe for multiple processes on the same node.
    NOT safe on NFS/EFS — use RedisLockBackend for multi-node.

    timeout_s=-1 means wait indefinitely (correct for high-concurrency single-node).
    """

    def __init__(self, lock_dir: str, timeout_s: float = -1) -> None:
        try:
            from filelock import FileLock as _FileLock
        except ImportError as exc:
            raise ImportError("pip install filelock") from exc
        self._FileLock = _FileLock
        self._lock_dir = lock_dir
        self._timeout_s = timeout_s

    @contextmanager
    def acquire(self, key: str, timeout_s: float = -1) -> Generator[None, None, None]:
        # sanitize key to safe filename
        safe_key = key.replace("/", "_").replace(":", "_")
        path = f"{self._lock_dir}/{safe_key}.lock"
        effective_timeout = timeout_s if timeout_s >= 0 else self._timeout_s
        lock = self._FileLock(path, timeout=effective_timeout)
        with lock:
            yield


# ── Redis (multi-node / Kubernetes) ──────────────────────────────────────────

_REDIS_RELEASE_SCRIPT = """
if redis.call("get", KEYS[1]) == ARGV[1] then
    return redis.call("del", KEYS[1])
else
    return 0
end
"""


class RedisLockBackend:
    """
    Distributed write lock using Redis SET NX PX (standard Redlock lite pattern).

    Safe across multiple nodes, pods, NFS/EFS mounts.

    Args:
        redis_client:  redis.Redis instance (sync).  Must be pre-configured
                       (host, port, db, auth, SSL).
        ttl_ms:        Lock TTL in milliseconds. Auto-released after this even
                       if the holder crashes. Default 10_000 (10 s).
        retry_delay_ms: How long to sleep between lock attempts. Default 50 ms.

    Requires:  pip install redis

    Example:
        import redis
        from chp.ledger.locks import RedisLockBackend

        r = redis.Redis.from_url("redis://redis-svc:6379/0", decode_responses=True)
        backend = RedisLockBackend(r)
        ledger = CHPLedger("/mnt/efs/chp", lock_backend=backend)
    """

    def __init__(
        self,
        redis_client: "redis.Redis",  # type: ignore[name-defined]
        ttl_ms: int = 10_000,
        retry_delay_ms: int = 50,
    ) -> None:
        self._redis = redis_client
        self._ttl_ms = ttl_ms
        self._retry_delay_ms = retry_delay_ms
        # Cached script SHA for EVALSHA (avoids re-sending script body every call)
        self._release_sha: str | None = None

    def _ensure_script(self) -> str:
        if self._release_sha is None:
            self._release_sha = self._redis.script_load(_REDIS_RELEASE_SCRIPT)
        return self._release_sha

    @contextmanager
    def acquire(self, key: str, timeout_s: float = -1) -> Generator[None, None, None]:
        lock_key = f"chp:lock:{key}"
        token = str(uuid.uuid4())
        deadline = time.monotonic() + timeout_s if timeout_s >= 0 else None

        # SET NX PX — atomic acquire
        while True:
            acquired = self._redis.set(lock_key, token, nx=True, px=self._ttl_ms)
            if acquired:
                break
            if deadline is not None and time.monotonic() >= deadline:
                raise TimeoutError(
                    f"CHP: could not acquire Redis lock '{key}' within {timeout_s}s"
                )
            time.sleep(self._retry_delay_ms / 1000.0)

        try:
            yield
        finally:
            sha = self._ensure_script()
            try:
                self._redis.evalsha(sha, 1, lock_key, token)
            except Exception:
                # Best-effort release. TTL ensures cleanup even if this fails.
                pass


# ── Async Redis (optional — for high-throughput async deployments) ─────────────

_ASYNC_REDIS_RELEASE_SCRIPT = _REDIS_RELEASE_SCRIPT  # same Lua


class AsyncRedisLockBackend:
    """
    Async variant of RedisLockBackend using redis.asyncio.

    Use when CHPLedger lives inside a fully async FastAPI / LangGraph service
    and you want non-blocking lock acquisition.

    Requires:  pip install redis  (redis.asyncio is bundled with redis-py >= 4.2)

    Example:
        from redis.asyncio import Redis
        from chp.ledger.locks import AsyncRedisLockBackend

        r = Redis.from_url("redis://redis-svc:6379/0", decode_responses=True)
        backend = AsyncRedisLockBackend(r)
        ledger = CHPLedger("/mnt/efs/chp", lock_backend=backend)

    Note: CHPLedger.write() is synchronous (LanceDB constraint). This backend
    wraps async acquire in asyncio.run() so it's callable from sync write().
    For pure-async write paths, use ledger.awrite() which offloads to thread pool.
    """

    def __init__(
        self,
        redis_client: "redis.asyncio.Redis",  # type: ignore[name-defined]
        ttl_ms: int = 10_000,
        retry_delay_ms: int = 50,
    ) -> None:
        import asyncio as _asyncio
        self._redis = redis_client
        self._ttl_ms = ttl_ms
        self._retry_delay_ms = retry_delay_ms
        self._asyncio = _asyncio
        self._release_sha: str | None = None

    async def _ensure_script(self) -> str:
        if self._release_sha is None:
            self._release_sha = await self._redis.script_load(_ASYNC_REDIS_RELEASE_SCRIPT)
        return self._release_sha

    async def _acquire_async(self, key: str, timeout_s: float) -> str:
        lock_key = f"chp:lock:{key}"
        token = str(uuid.uuid4())
        deadline = self._asyncio.get_event_loop().time() + timeout_s if timeout_s >= 0 else None

        while True:
            acquired = await self._redis.set(lock_key, token, nx=True, px=self._ttl_ms)
            if acquired:
                return token
            if deadline is not None and self._asyncio.get_event_loop().time() >= deadline:
                raise TimeoutError(
                    f"CHP: could not acquire Redis lock '{key}' within {timeout_s}s"
                )
            await self._asyncio.sleep(self._retry_delay_ms / 1000.0)

    async def _release_async(self, key: str, token: str) -> None:
        lock_key = f"chp:lock:{key}"
        sha = await self._ensure_script()
        try:
            await self._redis.evalsha(sha, 1, lock_key, token)
        except Exception:
            pass

    @contextmanager
    def acquire(self, key: str, timeout_s: float = -1) -> Generator[None, None, None]:
        # Called from sync write() — bridge via asyncio.run() in a thread.
        # awrite() already runs in asyncio.to_thread, so this is safe.
        import asyncio
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            loop = None

        if loop and loop.is_running():
            # We're inside an event loop (called via to_thread) — use run_coroutine_threadsafe
            import concurrent.futures
            fut = asyncio.run_coroutine_threadsafe(
                self._acquire_async(key, timeout_s), loop
            )
            token = fut.result(timeout=timeout_s if timeout_s > 0 else None)
        else:
            token = asyncio.run(self._acquire_async(key, timeout_s))

        try:
            yield
        finally:
            if loop and loop.is_running():
                import concurrent.futures
                fut = asyncio.run_coroutine_threadsafe(
                    self._release_async(key, token), loop
                )
                try:
                    fut.result(timeout=5)
                except Exception:
                    pass
            else:
                asyncio.run(self._release_async(key, token))
