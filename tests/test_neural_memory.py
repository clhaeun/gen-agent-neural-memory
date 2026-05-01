"""
Neural Memory Module 단위 테스트.

테스트 대상:
1. NeuralMemoryModule - write/read 기본 동작, 정규화 수정, lr_gate 분리
2. NeuralMemoryAdapter - AssociativeMemory 호환 인터페이스
"""

import torch
import pytest
import json
import os
import sys
import datetime
import tempfile
import shutil

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from neural_memory.config import NeuralMemoryConfig
from neural_memory.memory_module import NeuralMemoryModule
from neural_memory.adapter import NeuralMemoryAdapter, ConceptNode


# === Helper ===

def make_config(**overrides) -> NeuralMemoryConfig:
    defaults = dict(device="cpu", embed_dim=32, memory_dim=64)
    defaults.update(overrides)
    return NeuralMemoryConfig(**defaults)


def make_bootstrap_memory(tmpdir, embed_dim=32, n_nodes=3):
    """테스트용 bootstrap memory 파일 생성."""
    mem_dir = os.path.join(tmpdir, "bootstrap_memory", "associative_memory")
    os.makedirs(mem_dir, exist_ok=True)

    nodes = {}
    embeddings = {}
    for i in range(1, n_nodes + 1):
        node_id = f"node_{i}"
        emb_key = f"event_{i}_embedding"
        embeddings[emb_key] = [float(x) for x in torch.randn(embed_dim).tolist()]
        nodes[node_id] = {
            "node_count": i,
            "type_count": i,
            "type": "event",
            "depth": 0,
            "created": "2023-02-13 08:00:00",
            "expiration": None,
            "subject": f"person_{i}",
            "predicate": "is",
            "object": f"doing_thing_{i}",
            "description": f"Person {i} is doing thing {i}",
            "embedding_key": emb_key,
            "poignancy": 5,
            "keywords": [f"person_{i}", "is", f"doing_thing_{i}"],
            "filling": []
        }

    with open(os.path.join(mem_dir, "nodes.json"), "w") as f:
        json.dump(nodes, f)
    with open(os.path.join(mem_dir, "embeddings.json"), "w") as f:
        json.dump(embeddings, f)
    with open(os.path.join(mem_dir, "kw_strength.json"), "w") as f:
        json.dump({"kw_strength_event": {}, "kw_strength_thought": {}}, f)

    return mem_dir


# === NeuralMemoryModule Tests ===

class TestNeuralMemoryModule:

    def test_write_returns_nonnegative_surprise(self):
        config = make_config()
        memory = NeuralMemoryModule(config)
        emb = torch.randn(config.embed_dim)
        surprise = memory.write(emb)
        assert surprise >= 0

    def test_read_returns_correct_shape(self):
        config = make_config()
        memory = NeuralMemoryModule(config)
        memory.write(torch.randn(config.embed_dim))
        ctx = memory.read(torch.randn(config.embed_dim))
        assert ctx.shape == (config.embed_dim,)

    def test_write_count_increments(self):
        config = make_config()
        memory = NeuralMemoryModule(config)
        for i in range(5):
            memory.write(torch.randn(config.embed_dim))
        assert memory.write_count == 5


class TestRegularizationFix:

    def test_wk_prev_not_updated_every_step(self):
        config = make_config(reg_snapshot_interval=5)
        memory = NeuralMemoryModule(config)
        wk_prev_initial = memory.Wk_prev.clone()

        for i in range(4):
            memory.write(torch.randn(config.embed_dim))
        assert torch.equal(memory.Wk_prev, wk_prev_initial)

    def test_wk_prev_updated_at_interval(self):
        config = make_config(reg_snapshot_interval=5)
        memory = NeuralMemoryModule(config)
        wk_prev_initial = memory.Wk_prev.clone()

        for i in range(6):
            memory.write(torch.randn(config.embed_dim))
        assert not torch.equal(memory.Wk_prev, wk_prev_initial)

    def test_regularization_produces_nonzero_penalty(self):
        config = make_config(reg_snapshot_interval=100, reg_lambda=0.1)
        memory = NeuralMemoryModule(config)
        for i in range(20):
            memory.write(torch.randn(config.embed_dim))
        drift = torch.norm(memory.Wk - memory.Wk_prev).item()
        assert drift > 0


class TestLrGateFix:

    def test_lr_gate_has_separate_optimizer(self):
        config = make_config()
        memory = NeuralMemoryModule(config)
        assert memory._lr_gate_optimizer is not None

    def test_lr_gate_params_change_after_writes(self):
        config = make_config()
        memory = NeuralMemoryModule(config)
        gate_params_before = [p.clone() for p in memory.lr_gate.parameters()]
        for i in range(10):
            memory.write(torch.randn(config.embed_dim))
        changed = any(
            not torch.equal(before, after.data)
            for before, after in zip(gate_params_before, memory.lr_gate.parameters())
        )
        assert changed


# === NeuralMemoryAdapter Tests ===

class TestAdapterInterface:

    @pytest.fixture
    def adapter_setup(self, tmp_path):
        embed_dim = 32
        mem_dir = make_bootstrap_memory(str(tmp_path), embed_dim=embed_dim, n_nodes=3)
        config = make_config(embed_dim=embed_dim)
        adapter = NeuralMemoryAdapter(mem_dir, config)
        return adapter, embed_dim

    def test_loads_bootstrap_nodes(self, adapter_setup):
        adapter, _ = adapter_setup
        assert len(adapter.id_to_node) == 3
        assert len(adapter.seq_event) == 3

    def test_add_event_returns_concept_node(self, adapter_setup):
        adapter, dim = adapter_setup
        emb = [float(x) for x in torch.randn(dim).tolist()]
        node = adapter.add_event(
            created=datetime.datetime(2023, 2, 13, 9, 0, 0),
            expiration=None,
            s="Alice", p="is", o="cooking",
            description="Alice is cooking breakfast",
            keywords={"alice", "is", "cooking"},
            poignancy=7,
            embedding_pair=("alice_cooking_emb", emb),
            filling=[]
        )
        assert isinstance(node, ConceptNode)
        assert node.type == "event"
        assert node.node_id == "node_4"
        assert adapter.seq_event[0] == node  # prepended

    def test_add_thought_returns_concept_node(self, adapter_setup):
        adapter, dim = adapter_setup
        emb = [float(x) for x in torch.randn(dim).tolist()]
        node = adapter.add_thought(
            created=datetime.datetime(2023, 2, 13, 10, 0, 0),
            expiration=None,
            s="Alice", p="feels", o="happy",
            description="Alice feels happy today",
            keywords={"alice", "feels", "happy"},
            poignancy=6,
            embedding_pair=("alice_happy_emb", emb),
            filling=None
        )
        assert isinstance(node, ConceptNode)
        assert node.type == "thought"

    def test_add_chat_returns_concept_node(self, adapter_setup):
        adapter, dim = adapter_setup
        emb = [float(x) for x in torch.randn(dim).tolist()]
        node = adapter.add_chat(
            created=datetime.datetime(2023, 2, 13, 11, 0, 0),
            expiration=None,
            s="Alice", p="chat with", o="Bob",
            description="Alice chatted with Bob",
            keywords={"alice", "bob"},
            poignancy=4,
            embedding_pair=("alice_bob_chat_emb", emb),
            filling=[("Alice", "Hi Bob"), ("Bob", "Hi Alice")]
        )
        assert isinstance(node, ConceptNode)
        assert node.type == "chat"

    def test_retrieve_relevant_events(self, adapter_setup):
        adapter, _ = adapter_setup
        # "person_1" keyword was added during bootstrap loading
        results = adapter.retrieve_relevant_events("person_1", "is", "doing_thing_1")
        assert len(results) > 0

    def test_retrieve_relevant_thoughts_empty(self, adapter_setup):
        adapter, _ = adapter_setup
        results = adapter.retrieve_relevant_thoughts("nonexistent", "", "")
        assert len(results) == 0

    def test_get_last_chat(self, adapter_setup):
        adapter, dim = adapter_setup
        emb = [float(x) for x in torch.randn(dim).tolist()]
        adapter.add_chat(
            created=datetime.datetime(2023, 2, 13, 11, 0, 0),
            expiration=None,
            s="Alice", p="chat with", o="Bob",
            description="Chat with Bob",
            keywords={"alice", "bob"},
            poignancy=4,
            embedding_pair=("chat_emb", emb),
            filling=[]
        )
        result = adapter.get_last_chat("Bob")
        assert result is not False
        assert result.type == "chat"
        assert adapter.get_last_chat("Nonexistent") is False

    def test_embeddings_stored(self, adapter_setup):
        adapter, dim = adapter_setup
        emb = [float(x) for x in torch.randn(dim).tolist()]
        adapter.add_event(
            created=datetime.datetime(2023, 2, 13, 9, 0, 0),
            expiration=None,
            s="X", p="is", o="Y",
            description="X is Y",
            keywords={"x", "y"},
            poignancy=3,
            embedding_pair=("test_emb_key", emb),
            filling=[]
        )
        assert "test_emb_key" in adapter.embeddings
        assert adapter.embeddings["test_emb_key"] == emb

    def test_neural_write_records_surprise(self, adapter_setup):
        adapter, dim = adapter_setup
        initial_count = len(adapter.surprise_log)
        emb = [float(x) for x in torch.randn(dim).tolist()]
        adapter.add_event(
            created=datetime.datetime(2023, 2, 13, 9, 0, 0),
            expiration=None,
            s="X", p="is", o="Y",
            description="Test",
            keywords={"x"},
            poignancy=3,
            embedding_pair=("s_emb", emb),
            filling=[]
        )
        assert len(adapter.surprise_log) == initial_count + 1

    def test_neural_read(self, adapter_setup):
        adapter, dim = adapter_setup
        emb = [0.1] * dim
        ctx = adapter.neural_read(emb)
        assert ctx.shape == (dim,)

    def test_save_and_reload(self, adapter_setup, tmp_path):
        adapter, dim = adapter_setup
        save_dir = str(tmp_path / "save_test")
        os.makedirs(save_dir)
        adapter.save(save_dir)

        assert os.path.exists(os.path.join(save_dir, "nodes.json"))
        assert os.path.exists(os.path.join(save_dir, "embeddings.json"))
        assert os.path.exists(os.path.join(save_dir, "kw_strength.json"))
        assert os.path.exists(os.path.join(save_dir, "neural_memory.pt"))

    def test_kw_strength_updated(self, adapter_setup):
        adapter, dim = adapter_setup
        emb = [float(x) for x in torch.randn(dim).tolist()]
        adapter.add_event(
            created=datetime.datetime(2023, 2, 13, 9, 0, 0),
            expiration=None,
            s="Chef", p="is", o="cooking",
            description="Chef is cooking",
            keywords={"chef", "cooking"},
            poignancy=5,
            embedding_pair=("chef_emb", emb),
            filling=[]
        )
        assert "chef" in adapter.kw_strength_event

    def test_get_neural_stats(self, adapter_setup):
        adapter, _ = adapter_setup
        stats = adapter.get_neural_stats()
        assert "write_count" in stats
        assert "Wk_drift" in stats
        assert stats["write_count"] == 3  # from bootstrap loading


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
