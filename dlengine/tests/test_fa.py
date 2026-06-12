import os
import time

import pytest
import torch


pytestmark = pytest.mark.skipif(
    os.getenv("DLENGINE_RUN_BENCHMARKS") != "1",
    reason="benchmark disabled; set DLENGINE_RUN_BENCHMARKS=1 to run",
)


def test_flash_attn_gqa_decode_benchmark_smoke():
    if not torch.cuda.is_available():
        pytest.skip("CUDA required")

    flash_attn_interface = pytest.importorskip(
        "flash_attn_interface", reason="flash_attn_interface is required"
    )
    pytest.importorskip("einops", reason="einops is required")
    pytest.importorskip("triton", reason="triton is required")

    from einops import rearrange
    from triton.testing import do_bench

    device = "cuda"
    dtype = torch.bfloat16
    batch_size = 1
    seqlen_q = 1
    seqlen = int(os.getenv("DLENGINE_FA_BENCH_SEQLEN", "1024"))
    nheads_q = 64
    nheads_kv = 4
    headdim = 128
    page_size = 256
    num_splits = 0

    assert seqlen % page_size == 0

    torch.manual_seed(0)
    cache_seqlens = torch.tensor([seqlen] * batch_size, device=device, dtype=torch.int)
    q = torch.randn(batch_size, seqlen_q, nheads_q, headdim, dtype=dtype, device=device)
    k_cache = torch.randn(
        batch_size, seqlen, nheads_kv, headdim, dtype=dtype, device=device
    )
    v_cache = torch.randn(
        batch_size, seqlen, nheads_kv, headdim, dtype=dtype, device=device
    )
    k_cache, v_cache = [
        rearrange(x, "b (n p) h d -> (b n) p h d", p=page_size)
        for x in [k_cache, v_cache]
    ]
    page_table = rearrange(
        torch.arange(batch_size * seqlen // page_size, device=device, dtype=torch.int32),
        "(b s) -> b s",
        s=seqlen // page_size,
    )

    def fn():
        return flash_attn_interface.flash_attn_with_kvcache(
            q,
            k_cache,
            v_cache,
            cache_seqlens=cache_seqlens,
            num_splits=num_splits,
            qv=None,
            page_table=page_table,
            causal=True,
        )

    time.sleep(0.1)
    elapsed_ms = do_bench(fn, warmup=5, rep=10)
    assert elapsed_ms > 0


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
