"""SemanticMemory — vector remember/recall (mocked VectorStore)."""
from app.memory.semantic import SemanticMemory


class _FakeVector:
    def __init__(self) -> None:
        self.upserts: list[tuple] = []
        self._hits: list[dict] = []

    async def upsert(self, collection, *, ids, documents, metadatas, tenant_id):
        self.upserts.append((collection, documents, metadatas, tenant_id))

    async def query(self, collection, text, *, top_k=None, where=None, tenant_id=None):
        self.last_query = {"collection": collection, "where": where, "tenant_id": tenant_id}
        return self._hits


async def test_remember_is_customer_scoped():
    v = _FakeVector()
    await SemanticMemory(v).remember(
        "prefers silk sarees", tenant_id="t1", customer_id="9198", kind="pref"
    )
    assert len(v.upserts) == 1
    _coll, docs, metas, tid = v.upserts[0]
    assert docs == ["prefers silk sarees"]
    assert metas[0] == {"customer_id": "9198", "kind": "pref"}
    assert tid == "t1"


async def test_remember_skips_empty():
    v = _FakeVector()
    await SemanticMemory(v).remember("   ", tenant_id="t1", customer_id="9198")
    assert v.upserts == []


async def test_recall_maps_hits_and_scopes_to_customer():
    v = _FakeVector()
    v._hits = [{"document": "prefers silk", "metadata": {"kind": "pref"}}]
    out = await SemanticMemory(v).recall("silk?", tenant_id="t1", customer_id="9198")
    assert out == [{"text": "prefers silk", "kind": "pref"}]
    assert v.last_query["where"] == {"customer_id": "9198"}
    assert v.last_query["tenant_id"] == "t1"


async def test_recall_empty_query_returns_nothing():
    assert await SemanticMemory(_FakeVector()).recall(
        "", tenant_id="t1", customer_id="9198"
    ) == []
