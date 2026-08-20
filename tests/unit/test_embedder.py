import numpy as np
from chp.engine.embedder import StubEmbedder


def test_stub_embedder_shape():
    emb = StubEmbedder()
    result = emb.embed(["hello world", "another chunk"])
    assert result.shape == (2, 384)
    assert result.dtype == np.float32


def test_stub_embedder_deterministic():
    emb = StubEmbedder()
    r1 = emb.embed(["hello"])
    r2 = emb.embed(["hello"])
    np.testing.assert_array_equal(r1, r2)


def test_stub_embedder_empty():
    emb = StubEmbedder()
    result = emb.embed([])
    assert result.shape == (0, 384)
