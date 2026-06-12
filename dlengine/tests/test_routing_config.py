from unittest.mock import MagicMock, patch

from conftest import require_dlengine_cpp


cpp = require_dlengine_cpp()


def _make_config(routing_strategy="RoundRobin"):
    from dlengine.config import Config

    with patch("transformers.AutoConfig.from_pretrained") as mock_conf:
        mock_conf.return_value = MagicMock(
            architectures=["Qwen3ForCausalLM"],
            max_position_embeddings=32768,
            num_hidden_layers=32,
            num_attention_heads=32,
            hidden_size=4096,
        )
        return Config(
            model="/tmp/dlengine-test-model",
            engine_id="test_engine",
            routing_strategy=routing_strategy,
            attention_dp=2,
            attention_sp=1,
            max_num_seqs=8,
            max_num_batched_tokens=4096,
            max_model_len=4096,
            num_kvcache_blocks=1024,
            kvcache_block_size=256,
        )


def test_routing_strategy_from_config():
    config = _make_config()
    scheduler = cpp.init_scheduler(config)
    assert scheduler.routing_strategy == cpp.RoutingStrategy.RoundRobin

    config = _make_config("LeastBatch")
    scheduler = cpp.init_scheduler(config)
    assert scheduler.routing_strategy == cpp.RoutingStrategy.LeastBatch
