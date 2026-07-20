# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Unit tests for MoriTransferEngineConnector (mocked Mori IOEngine)."""

from types import SimpleNamespace
from unittest.mock import patch

import msgspec
import pytest

import vllm_omni.distributed.omni_connectors.connectors.mori_transfer_engine_connector as mori_module
from vllm_omni.distributed.omni_connectors.connectors.mori_transfer_engine_connector import (
    MoriPullRequest,
    MoriTransferEngineConnector,
    QueryRequest,
    QueryResponse,
)

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


class TestMoriZmqMessageTypes:
    def test_mori_pull_request_roundtrip(self):
        req = MoriPullRequest(
            request_id="req-1",
            engine_desc_packed=b"engine",
            mem_desc_packed=b"mem",
            dst_offset=64,
            length=128,
        )
        decoded = msgspec.msgpack.decode(msgspec.msgpack.encode(req), type=MoriPullRequest)
        assert decoded.request_id == "req-1"
        assert decoded.dst_offset == 64
        assert decoded.length == 128

    def test_query_request_response_roundtrip(self):
        q = QueryRequest(request_id="req-2")
        r = QueryResponse(request_id="req-2", data_size=512, is_fast_path=True)
        decoded_q = msgspec.msgpack.decode(msgspec.msgpack.encode(q), type=QueryRequest)
        decoded_r = msgspec.msgpack.decode(msgspec.msgpack.encode(r), type=QueryResponse)
        assert decoded_q.request_id == "req-2"
        assert decoded_r.data_size == 512
        assert decoded_r.is_fast_path is True


class TestMoriConnectorValidation:
    def test_init_raises_when_mori_unavailable(self, monkeypatch):
        monkeypatch.setattr(mori_module, "IOEngine", None)
        with pytest.raises(ImportError, match="Mori is not available"):
            MoriTransferEngineConnector({})

    def test_invalid_backend_type_raises(self, mori_mocks):
        with pytest.raises(ValueError, match="Invalid backend_type"):
            MoriTransferEngineConnector({"backend_type": "invalid"})

    def test_xgmi_requires_cuda_pool(self, mori_mocks):
        with pytest.raises(ValueError, match="memory_pool_device='cuda'"):
            MoriTransferEngineConnector({"backend_type": "xgmi", "memory_pool_device": "cpu"})

    def test_invalid_role_raises(self, mori_mocks):
        with pytest.raises(ValueError, match="Invalid role"):
            MoriTransferEngineConnector({"role": "both"})


class TestMoriConnectorBehavior:
    def test_receiver_rejects_put(self, mori_receiver_connector):
        ok, size, metadata = mori_receiver_connector.put("s0", "s1", "req-1", {"a": 1})
        assert ok is False
        assert size == 0
        assert metadata is None

    def test_sender_put_serializes_dict(self, mori_sender_connector):
        ok, size, metadata = mori_sender_connector.put("s0", "s1", "req-1", {"hello": "world"})
        assert ok is True
        assert size > 0
        assert metadata == {
            "source_host": mori_sender_connector.host,
            "source_port": mori_sender_connector.zmq_port,
            "data_size": size,
            "is_fast_path": False,
        }
        assert mori_sender_connector._metrics["puts"] == 1

    def test_sender_put_rejects_empty_bytes(self, mori_sender_connector):
        ok, size, metadata = mori_sender_connector.put("s0", "s1", "req-empty", b"")
        assert ok is False
        assert size == 0
        assert metadata is None

    def test_get_connection_info(self, mori_receiver_connector):
        info = mori_receiver_connector.get_connection_info()
        assert info["can_put"] is False
        assert "zmq_port" in info
        assert "engine_key" in info

    def test_update_sender_info(self, mori_receiver_connector):
        mori_receiver_connector.update_sender_info("10.0.0.2", 60000)
        assert mori_receiver_connector.sender_host == "10.0.0.2"
        assert mori_receiver_connector.sender_zmq_port == 60000

    def test_health_reports_metrics(self, mori_receiver_connector):
        health = mori_receiver_connector.health()
        assert health["status"] == "healthy"
        assert health["pool_device"] == "cpu"
        assert health["puts"] == 0

    def test_close_is_idempotent(self, mori_receiver_connector):
        mori_receiver_connector.close()
        mori_receiver_connector.close()
        assert mori_receiver_connector._closed is True

    @patch.object(MoriTransferEngineConnector, "_get_local_ip", return_value="192.168.1.10")
    def test_auto_host_uses_detected_ip(self, _mock_ip, mori_mocks):
        connector = MoriTransferEngineConnector({"host": "auto", "role": "receiver"})
        assert connector.host == "192.168.1.10"
        connector.close()

    def test_get_local_ip_fallback(self):
        with patch("socket.socket") as mock_socket_cls:
            mock_socket_cls.side_effect = OSError("network down")
            with patch("socket.gethostbyname", side_effect=OSError("dns down")):
                assert MoriTransferEngineConnector._get_local_ip() == "127.0.0.1"


@pytest.fixture
def mori_mocks(monkeypatch):
    class FakeEngineDesc:
        key = "remote-engine"
        host = "127.0.0.1"
        port = 12345

        def pack(self):
            return b"engine-desc"

        @classmethod
        def unpack(cls, _data: bytes):
            return cls()

    class FakeMemoryDesc:
        def pack(self):
            return b"mem-desc"

        @classmethod
        def unpack(cls, _data: bytes):
            return cls()

    class FakeEngine:
        def __init__(self, key, _config):
            self.key = key

        def create_backend(self, *_args, **_kwargs):
            return None

        def get_engine_desc(self):
            return FakeEngineDesc()

        def register_torch_tensor(self, _tensor):
            return FakeMemoryDesc()

        def register_remote_engine(self, _desc):
            return None

        def deregister_memory(self, _desc):
            return None

        def deregister_remote_engine(self, _desc):
            return None

    fake_backend = SimpleNamespace(RDMA=0, XGMI=1)
    fake_poll = SimpleNamespace(POLLING=0)

    monkeypatch.setattr(mori_module, "IOEngine", FakeEngine)
    monkeypatch.setattr(mori_module, "IOEngineConfig", lambda **kwargs: SimpleNamespace(**kwargs))
    monkeypatch.setattr(mori_module, "EngineDesc", FakeEngineDesc)
    monkeypatch.setattr(mori_module, "MemoryDesc", FakeMemoryDesc)
    monkeypatch.setattr(mori_module, "BackendType", fake_backend)
    monkeypatch.setattr(mori_module, "PollCqMode", fake_poll)
    monkeypatch.setattr(
        mori_module,
        "RdmaBackendConfig",
        lambda *args, **kwargs: SimpleNamespace(
            qp_per_transfer=args[0] if args else 1,
            post_batch_size=args[1] if len(args) > 1 else -1,
            num_workers=args[2] if len(args) > 2 else 1,
        ),
    )
    monkeypatch.setattr(
        mori_module,
        "XgmiBackendConfig",
        lambda: SimpleNamespace(num_streams=64, num_events=64),
    )

    def _skip_listener(self):
        self._listener_ready.set()

    monkeypatch.setattr(MoriTransferEngineConnector, "_zmq_listener_loop", _skip_listener)
    return mori_module


@pytest.fixture
def mori_receiver_connector(mori_mocks):
    connector = MoriTransferEngineConnector(
        {
            "role": "receiver",
            "sender_host": "127.0.0.1",
            "sender_zmq_port": 50051,
            "memory_pool_size": 1024 * 1024,
        }
    )
    yield connector
    connector.close()


@pytest.fixture
def mori_sender_connector(mori_mocks):
    connector = MoriTransferEngineConnector(
        {
            "role": "sender",
            "zmq_port": 50123,
            "memory_pool_size": 1024 * 1024,
        }
    )
    yield connector
    connector.close()
