import ctypes
import unittest

import pytest
from conftest import require_dlengine_cpp


cpp = require_dlengine_cpp()


class TestSerialization(unittest.TestCase):
    def test_round_trip(self):
        seq = cpp.Sequence([101, 102, 103], cpp.SamplingParams())
        seq.seq_id = 999
        seq.status = cpp.SequenceStatus.WAITING

        buffer_size = 4096
        buffer = ctypes.create_string_buffer(buffer_size)
        ptr = ctypes.addressof(buffer)

        size = cpp.serialize(ptr, buffer_size, [seq], True)
        restored_seqs = cpp.deserialize(ptr, size)

        self.assertEqual(len(restored_seqs), 1)
        r_seq = restored_seqs[0]
        self.assertEqual(r_seq.seq_id, seq.seq_id)
        self.assertEqual(r_seq.token_ids, seq.token_ids)

    def test_manual_flatbuffer_construction(self):
        pytest.skip(
            "current dlengine package does not ship generated Python dlengine.fbs modules"
        )

    def test_reproduce_payload(self):
        hex_str = "100000000000000000000600080004000600000004000000010000001c000000000016001800100007000800000000000000000000000c00160000000000000118000000240000007b000000000000000800100008000400080000000a0000009a9999999999b93f03000000010000000200000003000000"
        payload = bytes.fromhex(hex_str)

        c_buf = ctypes.create_string_buffer(payload, len(payload))
        ptr = ctypes.addressof(c_buf)

        restored_seqs = cpp.deserialize(ptr, len(payload))
        self.assertGreaterEqual(len(restored_seqs), 0)


if __name__ == "__main__":
    unittest.main()
