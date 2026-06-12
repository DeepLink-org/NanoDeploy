import torch
from conftest import require_dlengine_cpp


cpp = require_dlengine_cpp()


def test_serialization_size_with_scheduler_state():
    cpp.Sequence.set_block_size(64)
    scheduler = cpp.Scheduler(
        "test_engine",
        0,  # num_speculative_tokens
        8,  # max_num_seqs
        4096,  # max_num_batched_tokens
        4096,  # max_model_len
        [],  # eos_ids
        1,  # attention_dp
        1,  # group_size
        1024,  # num_kvcache_blocks
        64,  # kvcache_block_size
        "hybrid",
    )

    seqs = []
    for i in range(8):
        token_ids = list(range(i * 256, (i + 1) * 256))
        seq = cpp.Sequence(token_ids, cpp.SamplingParams())
        scheduler.add(seq)
        seqs.append(seq)

    sch_res = scheduler.schedule()
    assert sch_res.dp_seqs and sch_res.dp_seqs[0]

    scheduled_seq = sch_res.dp_seqs[0][0]
    num_blocks = scheduled_seq.num_blocks(cpp.BlockContextSlot.ACTIVE, 0)
    assert num_blocks > 0

    buffer = torch.zeros(4 * 1024 * 1024, dtype=torch.uint8)
    buffer_ptr = buffer.data_ptr()

    size_prefill = cpp.serialize(buffer_ptr, buffer.numel(), sch_res.dp_seqs[0], True)
    assert 0 < size_prefill <= buffer.numel()

    size_decode = cpp.serialize(buffer_ptr, buffer.numel(), sch_res.dp_seqs[0], False)
    assert 0 < size_decode <= buffer.numel()
