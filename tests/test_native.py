"""CPU contract tests plus optional probes of the actual compiled C++ mapping.

Set FLUX_5090_NATIVE_TEST_LIBRARY to a freshly built library to exercise the
host/device shared Schedule::tile implementation without requiring a GPU.
"""
import ctypes
import hashlib
import json
import os
from pathlib import Path
import tempfile
import unittest

from flux_5090.native import mapping_reference, source_inputs, validate_schedule


def order_for(world, per_destination, interleaved, reverse_odd=False):
    def physical(destination, index):
        if reverse_odd and destination % 2:
            index = per_destination - 1 - index
        return destination * per_destination + index
    if interleaved:
        return [physical(d, i) for i in range(per_destination) for d in range(world)]
    return [physical(d, i) for d in range(world) for i in range(per_destination)]


class NativeScheduleTests(unittest.TestCase):
    def test_destination_major_full_k_tile_grid(self):
        mapping = mapping_reference(2048, 4096, 8, 128, list(range(16)), False)
        self.assertEqual(mapping, [(m, n) for m in range(16) for n in range(32)])

    def test_interleave_is_paper_tile_mapping_without_rank_rotation(self):
        order = order_for(8, 2, True)
        mapping = mapping_reference(2048, 4096, 8, 128, order, True)
        expected = []
        for logical in range(512):
            destination, within = logical % 8, logical // 8
            expected.append((destination * 2 + within // 32, within % 32))
        self.assertEqual(mapping, expected)
        self.assertEqual(set(mapping), {(m, n) for m in range(16) for n in range(32)})

    def test_destination_specific_arrival_permutations_do_not_change_coordinates(self):
        order = order_for(8, 4, True, reverse_odd=True)
        mapping = mapping_reference(4096, 256, 8, 128, order, True)
        self.assertEqual(len(set(mapping)), len(mapping))
        self.assertEqual(mapping[:8], [(d * 4 + (3 if d % 2 else 0), 0) for d in range(8)])
        self.assertEqual([m // 4 for m, _ in mapping], list(range(8)) * 8)

    def test_multi_tile_macroblock_publication_coverage(self):
        order = order_for(8, 2, True, reverse_odd=True)
        mapping = mapping_reference(4096, 512, 8, 256, order, True)
        counts = {block: 0 for block in range(16)}
        for row, _ in mapping:
            counts[row // 2] += 1
        self.assertEqual(set(counts.values()), {8})
        self.assertEqual(set(mapping), {(m, n) for m in range(32) for n in range(4)})

    def test_invalid_mapping_rejected_before_native_launch(self):
        valid = order_for(8, 2, True)
        invalid = [valid[:-1] + valid[:1], list(range(16)), valid[:-1], [True] + valid[1:]]
        for order in invalid:
            with self.subTest(order=order), self.assertRaises(ValueError):
                validate_schedule(2048, 4096, 8, 128, order, True)
        with self.assertRaises(ValueError):
            validate_schedule(1024, 4096, 8, 256, list(range(4)), False)

    def test_frozen_source_identity_is_complete(self):
        inputs, sources = source_inputs()
        self.assertEqual(sources["flux"]["revision"], "19831ca2d820e3e782ed1d15d8b52d0898b78b26")
        self.assertEqual(sources["cutlass"]["revision"], "df8a550d3917b0e97f416b2ed8c2d786f7f686a3")
        self.assertIn("third_party/flux/src/comm_none/gemm_v2_comm_none.hpp", inputs)
        self.assertIn("third_party/cutlass/include/cutlass/cutlass.h", inputs)
        self.assertIn("third_party/cutlass/LICENSE.txt", inputs)
        self.assertIn("third_party/cutlass.lock.json", inputs)
        self.assertEqual(sources["cutlass"]["manifest_sha256"], inputs["third_party/cutlass.lock.json"])


class NativeSourceLockTests(unittest.TestCase):
    def fixture(self, root):
        for name in ("flux", "cutlass"):
            folder = root / "third_party" / name
            folder.mkdir(parents=True)
            (folder / "header.h").write_bytes(name.encode())
            manifest = json.dumps(dict(revision="fixture", files={
                "header.h": hashlib.sha256(name.encode()).hexdigest()})).encode()
            (folder / "manifest.json").write_bytes(manifest)
            if name == "cutlass":
                (root / "third_party/cutlass.lock.json").write_bytes(manifest)
        for name in ("src/gemm/native.cu", "python/flux_5090/native.py"):
            file = root / name
            file.parent.mkdir(parents=True)
            file.write_text("fixture")

    def test_missing_checkout_has_setup_instruction(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self.fixture(root)
            # Only change this test's disposable fixture, never the real checkout.
            (root / "third_party/cutlass").rename(root / "not-configured")
            with self.assertRaisesRegex(RuntimeError, "bash env/setup_5090.sh"):
                source_inputs(root)

    def test_external_lock_is_authoritative_and_local_manifest_is_optional(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self.fixture(root)
            inputs, sources = source_inputs(root)
            self.assertEqual(inputs["third_party/cutlass.lock.json"], inputs["third_party/cutlass/manifest.json"])
            (root / "third_party/cutlass/manifest.json").unlink()
            without_local, provenance = source_inputs(root)
            self.assertNotIn("third_party/cutlass/manifest.json", without_local)
            self.assertEqual(provenance, sources)
            self.assertIn("third_party/cutlass/header.h", without_local)

    def test_changed_manifest_cannot_replace_lock_and_header_changes_fail(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self.fixture(root)
            local = root / "third_party/cutlass/manifest.json"
            local.write_text("{}")
            with self.assertRaisesRegex(RuntimeError, "manifest differs"):
                source_inputs(root)
            local.write_bytes((root / "third_party/cutlass.lock.json").read_bytes())
            (root / "third_party/cutlass/header.h").write_text("changed")
            with self.assertRaisesRegex(RuntimeError, "source missing/changed"):
                source_inputs(root)


@unittest.skipUnless(os.environ.get("FLUX_5090_NATIVE_TEST_LIBRARY"),
                     "Set FLUX_5090_NATIVE_TEST_LIBRARY after a native build")
class CompiledNativeMappingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        library = Path(os.environ["FLUX_5090_NATIVE_TEST_LIBRARY"])
        cls.native = ctypes.CDLL(str(library.resolve()))
        cls.native.fx_last_error.restype = ctypes.c_char_p
        cls.native.fx_map_host.argtypes = [ctypes.c_int] * 4 + [ctypes.c_void_p, ctypes.c_int, ctypes.c_void_p]
        cls.native.fx_map_host.restype = ctypes.c_int

    def test_actual_native_map_matches_spec_for_both_shapes_and_arrival_orders(self):
        for world, per_destination, chunk_rows, n in ((8, 2, 128, 4096), (2, 3, 256, 256),
                                                       (8, 1, 128, 128)):
            for interleaved in (False, True):
                for reverse_odd in (False, True):
                    with self.subTest(world=world, chunk_rows=chunk_rows, interleaved=interleaved,
                                      reverse_odd=reverse_odd):
                        m = world * per_destination * chunk_rows
                        order = order_for(world, per_destination, interleaved, reverse_odd)
                        args = (ctypes.c_int * len(order))(*order)
                        count = (m // 128) * (n // 128)
                        output = (ctypes.c_int * (2 * count))()
                        rc = self.native.fx_map_host(m, n, world, chunk_rows, args, interleaved, output)
                        self.assertEqual(rc, 0, self.native.fx_last_error())
                        actual = [(output[2 * i], output[2 * i + 1]) for i in range(count)]
                        self.assertEqual(actual, mapping_reference(m, n, world, chunk_rows, order, interleaved))

    def test_native_rejects_non_bijective_or_cross_destination_schedule(self):
        for order in ([0] * 16, list(range(16))):
            args = (ctypes.c_int * 16)(*order)
            output = (ctypes.c_int * 1024)()
            self.assertEqual(self.native.fx_map_host(2048, 4096, 8, 128, args, 1, output), -1)
            self.assertTrue(self.native.fx_last_error())


if __name__ == "__main__":
    unittest.main()
