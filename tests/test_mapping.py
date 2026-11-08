import copy
import sys
import unittest
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'python'))
from flux_5090.mapping import ARMS, BACKEND, cta_mapping, schedule, split_sizes, wire_bytes
from flux_5090.planning import make_role


class MappingTests(unittest.TestCase):
    def test_common_interleave_mapping_and_local_spacing(self):
        for world in (2, 4, 8):
            base = schedule(256 * world, 4096, 512, world, 128, ARMS[1])
            mixed = schedule(256 * world, 4096, 512, world, 128, ARMS[2])
            self.assertEqual(sorted(x['physical'] for x in mixed), list(range(2 * world)))
            self.assertEqual({(x['destination'], x['row']) for x in base},
                             {(x['destination'], x['row']) for x in mixed})
            for rank in range(world):
                self.assertEqual([x['slot'] for x in mixed if x['destination'] == rank], [rank, rank + world])

    def test_ctas_interleave_at_column_tile_granularity(self):
        base = cta_mapping(2048,4096,512,8,128,ARMS[1])
        mixed = cta_mapping(2048,4096,512,8,128,ARMS[2])
        self.assertEqual(base, [(row,col) for row in range(16) for col in range(32)])
        self.assertEqual(mixed[:16], [(2*d,col) for col in (0,1) for d in range(8)])
        self.assertEqual(set(base),set(mixed))
        self.assertEqual(len(set(mixed)),512)

    def test_collective_splits_match_every_pair(self):
        for world in (2, 4, 8):
            for destination in range(world):
                pairs = [split_sizes(world, rank, destination, 136) for rank in range(world)]
                for source in range(world):
                    for target in range(world):
                        self.assertEqual(pairs[source][0][target], pairs[target][1][source])
                self.assertEqual(sum(sum(s) for s, _ in pairs), (world - 1) * 136)

    def test_exact_byte_budget(self):
        self.assertEqual(wire_bytes(2048,4096,8,False),14680064)
        self.assertEqual(wire_bytes(2048,4096,8,True),7798784)

    def test_invalid_shapes_fail(self):
        for values in ((2049,4096,512,8,128),(2048,130,512,8,128),(2048,4096,0,8,128),
                       (2048,4096,512,8,512),(2048,4096,511,8,128),(2048,4096,512,8,64)):
            with self.subTest(values=values), self.assertRaises(ValueError):
                schedule(*values,'flux')

    def test_arrival_plan_preserves_destinations_and_physical_lanes(self):
        for world in (2,4,8):
            shape=(256*world,4096,512)
            original=schedule(*shape,world,128,ARMS[2])
            reverse=[d*2+(1-within) for within in range(2) for d in range(world)]
            plan=make_role(*shape,world,128,order=reverse)
            changed=schedule(*shape,world,128,ARMS[3],plan)
            self.assertEqual([row['destination'] for row in changed],[i%world for i in range(2*world)])
            self.assertEqual({row['physical']:row['lane'] for row in original},
                             {row['physical']:row['lane'] for row in changed})
            self.assertEqual(set(cta_mapping(*shape,world,128,ARMS[3],plan)),
                             set(cta_mapping(*shape,world,128,ARMS[2])))

    def test_only_arrival_arm_accepts_plan(self):
        shape=(512,128,64)
        plan=make_role(*shape,2,128)
        with self.assertRaises(ValueError): schedule(*shape,2,128,ARMS[3])
        for arm in ARMS[:3]:
            with self.assertRaises(ValueError): schedule(*shape,2,128,arm,plan)
        self.assertEqual(len(ARMS),4)
        self.assertEqual(BACKEND,'flux-cutlass-dp-ready-nccl-v1')

    def test_plan_mutation_rejected(self):
        shape=(512,128,64)
        original=make_role(*shape,2,128)
        for field,value in (('arrival_order',[0,0,1,3]),('arrival_order',[2,0,1,3]),
                            ('arrival_order',[False,2,1,3]),('selective_mask',[True]*4),
                            ('shape',[1024,128,64]),('chunk_rows',64),('diagnostics',None)):
            plan=copy.deepcopy(original);plan[field]=value
            with self.subTest(field=field,value=value), self.assertRaises(ValueError):
                schedule(*shape,2,128,ARMS[3],plan)


if __name__ == '__main__':
    unittest.main()
