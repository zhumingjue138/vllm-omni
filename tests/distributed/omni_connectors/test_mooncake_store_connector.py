# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Unit tests for MooncakeStoreConnector (mocked Mooncake store)."""

import pytest

import vllm_omni.distributed.omni_connectors.connectors.mooncake_store_connector as mooncake_store_module
from vllm_omni.distributed.omni_connectors.connectors.mooncake_store_connector import (
    MooncakeStoreConnector,
)

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


@pytest.fixture
def fake_mooncake_store(monkeypatch):
    class FakeReplicateConfig:
        def __init__(self):
            self.with_soft_pin = False

    class FakeStore:
        def __init__(self):
            self._data: dict[str, bytes] = {}
            self.closed = False

        def setup(self, host, metadata, segment, localbuf, proto, rdma, master):
            self.setup_args = (host, metadata, segment, localbuf, proto, rdma, master)
            return 0

        def put(self, key, serialized_data, pin):
            self._data[key] = serialized_data
            return True

        def get(self, key):
            return self._data.get(key)

        def close(self):
            self.closed = True

    monkeypatch.setattr(mooncake_store_module, "MooncakeDistributedStore", FakeStore)
    monkeypatch.setattr(mooncake_store_module, "ReplicateConfig", FakeReplicateConfig)
    return FakeStore


def test_init_raises_when_mooncake_unavailable(monkeypatch):
    monkeypatch.setattr(mooncake_store_module, "MooncakeDistributedStore", None)
    monkeypatch.setattr(mooncake_store_module, "ReplicateConfig", None)
    with pytest.raises(ImportError, match="Mooncake components"):
        MooncakeStoreConnector({})


def test_put_get_roundtrip(fake_mooncake_store, monkeypatch):
    monkeypatch.setattr(mooncake_store_module.time, "sleep", lambda _s: None)
    connector = MooncakeStoreConnector({"host": "10.0.0.1"})
    payload = {"key": "value", "items": [1, 2, 3]}

    ok, size, metadata = connector.put("stage_a", "stage_b", "req-1", payload)
    assert ok is True
    assert size > 0
    assert metadata is None
    assert connector._metrics["puts"] == 1

    result = connector.get("stage_a", "stage_b", "req-1")
    assert result is not None
    data, ret_size = result
    assert data == payload
    assert ret_size == size
    assert connector._metrics["gets"] == 1


def test_put_failure_increments_errors(fake_mooncake_store, monkeypatch):
    class FailingStore(fake_mooncake_store):
        def put(self, key, serialized_data, pin):
            return False

    monkeypatch.setattr(mooncake_store_module, "MooncakeDistributedStore", FailingStore)
    connector = MooncakeStoreConnector({})
    ok, size, metadata = connector.put("s0", "s1", "req-fail", {"x": 1})
    assert ok is False
    assert size == 0
    assert metadata is None
    assert connector._metrics["errors"] == 1


def test_get_timeout_when_key_missing(fake_mooncake_store, monkeypatch):
    monkeypatch.setattr(mooncake_store_module.time, "sleep", lambda _s: None)
    connector = MooncakeStoreConnector({})
    assert connector.get("s0", "s1", "missing-key") is None
    assert connector._metrics["timeouts"] == 1


def test_health_and_close(fake_mooncake_store):
    connector = MooncakeStoreConnector({"host": "127.0.0.1"})
    health = connector.health()
    assert health["status"] == "healthy"
    assert health["host"] == "127.0.0.1"
    assert health["puts"] == 0

    connector.close()
    assert connector.store is None


def test_setup_failure_raises(fake_mooncake_store, monkeypatch):
    class BadSetupStore(fake_mooncake_store):
        def setup(self, *args, **kwargs):
            return 1

    monkeypatch.setattr(mooncake_store_module, "MooncakeDistributedStore", BadSetupStore)
    with pytest.raises(RuntimeError, match="Mooncake setup failed"):
        MooncakeStoreConnector({})


def test_put_without_store_returns_false(monkeypatch):
    connector = MooncakeStoreConnector.__new__(MooncakeStoreConnector)
    connector.store = None
    connector._metrics = {"errors": 0}
    ok, size, metadata = connector.put("s0", "s1", "k", {"a": 1})
    assert ok is False
    assert size == 0
    assert metadata is None


def test_cleanup_is_noop(fake_mooncake_store):
    connector = MooncakeStoreConnector({})
    connector.cleanup("req-1")  # should not raise
