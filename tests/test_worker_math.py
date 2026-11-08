"""The GPU preflight oracle is also checked against an independent CPU oracle."""
import importlib.util
import math
from pathlib import Path
import random
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'python'))


@unittest.skipUnless(importlib.util.find_spec('torch'), 'CPU PyTorch not installed')
class FrameworkReferenceTests(unittest.TestCase):
    def test_all_remote_taco_and_local_bf16_for_all_four_arms(self):
        import torch
        from flux_5090.worker import reduce_reference_sources, independent_codec
        from flux_5090.mapping import ARMS
        generator = torch.Generator().manual_seed(43)
        sources = torch.randn((4, 2, 128), generator=generator).to(torch.bfloat16)
        decoded = [independent_codec(source) for source in sources]
        for rank in range(4):
            plain = torch.zeros_like(sources[0])
            compressed = torch.zeros_like(plain)
            for source in range(4):
                plain = (plain.float() + sources[source].float()).to(torch.bfloat16)
                contribution = sources[source] if source == rank else decoded[source]
                compressed = (compressed.float() + contribution.float()).to(torch.bfloat16)
            for arm in ARMS:
                actual_plain, actual = reduce_reference_sources(sources, rank, arm != 'flux')
                self.assertTrue(torch.equal(actual_plain, plain), (rank, arm))
                self.assertTrue(torch.equal(actual, plain if arm == 'flux' else compressed), (rank, arm))

    def test_local_source_is_never_quantized(self):
        import torch
        from flux_5090.worker import reduce_reference_sources, independent_codec
        generator = torch.Generator().manual_seed(309)
        local = torch.randn((2, 128), generator=generator).to(torch.bfloat16)
        self.assertFalse(torch.equal(local, independent_codec(local)))
        for rank in range(4):
            sources = torch.zeros((4, 2, 128), dtype=torch.bfloat16)
            sources[rank] = local
            plain, actual = reduce_reference_sources(sources, rank, True)
            self.assertTrue(torch.equal(plain, local))
            self.assertTrue(torch.equal(actual, local))

    def test_reference_keeps_source_order_and_bf16_rounding_after_each_add(self):
        import torch
        from flux_5090.worker import reduce_reference_sources
        # BF16: (256 + 1) rounds to 256, then -256 + 1 produces 1. A sum
        # in FP32 or arrival order [0,2,1,3] instead produces 2.
        sources = torch.tensor([256., 1., -256., 1.], dtype=torch.bfloat16)
        sources = sources[:, None, None].expand(4, 2, 128).contiguous()
        for rank in range(4):
            for quantized in (False, True):
                plain, actual = reduce_reference_sources(sources, rank, quantized)
                self.assertTrue(torch.equal(plain, torch.ones_like(plain)))
                self.assertTrue(torch.equal(actual, torch.ones_like(actual)))
        _, changed = reduce_reference_sources(sources[[0, 2, 1, 3]], 0, False)
        self.assertTrue(torch.equal(changed, torch.full_like(changed, 2.)))

    def test_model_receives_role_specific_frozen_plans(self):
        from unittest.mock import patch
        import torch
        import torch.distributed as dist
        from flux_5090.model import build_model
        from flux_5090.planning import make_role
        config = dict(layers=2, hidden=256, ffn=1024, heads=8, sequence=512,
                      micro_batch=1, accumulation_steps=1, vocab=1024, chunk_rows=128)
        bundle = dict(roles={
            'attn': make_role(512, 256, 128, 2, 128, order=[1, 2, 0, 3]),
            'mlp': make_role(512, 256, 512, 2, 128, order=[0, 3, 1, 2]),
        })
        received = []
        class Recorder:
            def __init__(self, **kwargs):
                received.append(kwargs)
        with patch.object(dist, 'is_initialized', return_value=True), \
             patch.object(dist, 'get_world_size', return_value=2), \
             patch.object(dist, 'get_rank', return_value=0):
            model = build_model(config, 'flux_taco_interleaved_arrival',
                                device='cpu', operator_factory=Recorder, execution_plan=bundle)
        self.assertEqual(len(model.operators), 4)
        for index, kwargs in enumerate(received):
            self.assertEqual(kwargs['plan'], bundle['roles']['attn' if index % 2 == 0 else 'mlp'])
            self.assertEqual(kwargs['arm'], 'flux_taco_interleaved_arrival')
            self.assertEqual(kwargs['m'], 512)
            self.assertEqual(kwargs['chunk_rows'], 128)

    def test_high_level_codec_matches_fp64_matrix_oracle(self):
        import torch
        from flux_5090.reference import encode, decode
        from flux_5090.worker import independent_codec, relative_error
        rng = random.Random(1197)
        cases = [[0.] * 256, [rng.gauss(0, 1) for _ in range(256)],
                 [64. if i == 0 else (-96. if i == 128 else .01) for i in range(256)]]
        for values in cases:
            data = torch.tensor(values, dtype=torch.bfloat16).reshape(2, 128)
            expected = torch.tensor(decode(encode(data.float().flatten().tolist()))).reshape(2, 128)
            actual = independent_codec(data)
            self.assertLessEqual(relative_error(actual, expected), .02)

    def test_tensor_hash_is_bit_sensitive(self):
        import torch
        from flux_5090.worker import tensor_sha
        a = torch.zeros(8, dtype=torch.bfloat16)
        b = a.clone(); b[0] = 1
        self.assertEqual(tensor_sha(a), tensor_sha(a.clone()))
        self.assertNotEqual(tensor_sha(a), tensor_sha(b))
        positive_zero = torch.tensor([0.], dtype=torch.bfloat16)
        negative_zero = torch.tensor([-0.], dtype=torch.bfloat16)
        self.assertTrue(torch.equal(positive_zero, negative_zero))
        self.assertNotEqual(tensor_sha(positive_zero), tensor_sha(negative_zero))

    def test_relative_error_rejects_nonfinite_actual_or_reference(self):
        import torch
        from flux_5090.worker import relative_error
        zero = torch.zeros(128, dtype=torch.bfloat16)
        self.assertEqual(relative_error(zero, zero), 0.)
        for value in (math.nan, math.inf, -math.inf):
            invalid = zero.clone(); invalid[31] = value
            self.assertEqual(relative_error(invalid, zero), math.inf)
            self.assertEqual(relative_error(zero, invalid), math.inf)


if __name__ == '__main__':
    unittest.main()
