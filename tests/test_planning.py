"""Arrival plans use full-operator held-out measurements and preserve destinations."""
import json
from pathlib import Path
import sys
import unittest
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'python'))
from flux_5090 import planning as p


class PlanContract(unittest.TestCase):
    shape = (512, 128, 16, 2, 128)

    def test_orders_are_legal_rank_independent_and_deduplicated(self):
        for world in (2, 4, 8):
            shape = (256 * world, 128, 16, world, 128)
            candidates = p.candidates(*shape)
            self.assertEqual(candidates[0]['arrival_order'], p.base_order(*shape))
            self.assertEqual(len({tuple(c['arrival_order']) for c in candidates}), len(candidates))
            for candidate in candidates:
                plan = p.make_role(*shape, order=candidate['arrival_order'])
                self.assertIs(p.validate_role_plan(plan, *shape), plan)
                self.assertEqual([block // 2 for block in plan['arrival_order']],
                                 [slot % world for slot in range(2 * world)])

    def test_keyword_signature_and_invalid_plans(self):
        plan = p.make_role(*self.shape)
        self.assertIs(p.validate_role_plan(plan, shape=[512, 128, 16], world_size=2, chunk_rows=128), plan)
        for key, value in [('shape', [512, 128, 17]), ('chunk_rows', 256),
                           ('arrival_order', [0, 1, 2, 3]), ('arrival_order', [0, 2, 0, 3]),
                           ('selective_mask', [True] * 4), ('diagnostics', None)]:
            with self.subTest(key=key), self.assertRaises(ValueError):
                p.validate_role_plan(dict(plan, **{key: value}), *self.shape)

    def test_digest_has_stable_json_and_catches_changes(self):
        plan = p.make_role(*self.shape)
        self.assertEqual(p.role_digest(plan), p.role_digest(json.loads(json.dumps(plan))))
        self.assertNotEqual(p.role_digest(plan), p.role_digest(dict(plan, diagnostics={'test': 1})))
        with self.assertRaises(ValueError):
            p.role_digest(dict(plan, diagnostics={'time': float('nan')}))

    def test_one_block_per_destination_has_identity_only(self):
        self.assertEqual(len(p.candidates(1024, 128, 16, 8, 128)), 1)


class MeasuredSelection(unittest.TestCase):
    shape = (512, 128, 16, 2, 128)
    cfg = dict(warmup=0, iterations=1, training_pairs=3, confirmation_pairs=3, min_gain=.002)

    def evaluator(self, *, reject=False, noise=False):
        calls = []
        reverse = p.candidates(*self.shape)[1]['arrival_order']
        def measure(plan, seed, arm, label):
            self.assertEqual(arm, p.ARRIVAL_ARM)
            calls.append(dict(plan=json.loads(json.dumps(plan)), seed=seed, label=label))
            latency = 90. if plan['arrival_order'] == reverse else 100.
            if reject and label.startswith('arrival_confirmation') and '/candidate' in label:
                latency = 110.
            if noise and 'confirmation' in label and '/pair1/candidate' in label:
                latency = 101.
            return latency
        return measure, calls

    def test_full_measurement_selects_order_with_ab_ba_pairs(self):
        evaluate, _ = self.evaluator()
        plan = p.fit_role(*self.shape, evaluate, config=self.cfg)
        self.assertTrue(plan['diagnostics']['arrival']['accepted'])
        self.assertEqual(plan['arrival_order'], p.candidates(*self.shape)[1]['arrival_order'])
        self.assertEqual(set(plan), {'shape', 'chunk_rows', 'arrival_order', 'diagnostics'})
        for record in plan['diagnostics']['arrival']['training_candidates']:
            self.assertEqual(record['paired_samples'][0]['execution_order'], ['reference', 'candidate'])
            self.assertEqual(record['paired_samples'][1]['execution_order'], ['candidate', 'reference'])

    def test_confirmation_does_not_reselect_and_uses_separate_inputs(self):
        evaluate, calls = self.evaluator(reject=True)
        plan = p.fit_role(*self.shape, evaluate, config=self.cfg)
        self.assertEqual(plan['arrival_order'], p.base_order(*self.shape))
        self.assertTrue(plan['diagnostics']['arrival']['identity_fallback'])
        candidates = [c for c in calls if 'confirmation' in c['label'] and '/candidate' in c['label']]
        self.assertEqual(len({p.role_digest(c['plan']) for c in candidates}), 1)
        fit = {c['seed'] for c in calls if 'confirmation' not in c['label']}
        heldout = {c['seed'] for c in calls if 'confirmation' in c['label']}
        self.assertFalse(fit & heldout)

    def test_unstable_confirmation_rejects_optimistic_average(self):
        evaluate, _ = self.evaluator(noise=True)
        plan = p.fit_role(*self.shape, evaluate, config=self.cfg)
        self.assertFalse(plan['diagnostics']['arrival']['accepted'])

    def test_no_speedup_retains_identity(self):
        plan = p.fit_role(*self.shape, lambda *args: 100., config=self.cfg)
        self.assertEqual(plan['arrival_order'], p.base_order(*self.shape))
        self.assertIsNone(plan['diagnostics']['arrival']['confirmation'])

    def test_invalid_measurements_and_numerical_failure_are_fatal(self):
        for value in (0., -1., float('nan'), float('inf'), {'elapsed_seconds': 1., 'correctness_passed': False}):
            with self.subTest(value=value), self.assertRaises(ValueError):
                p.fit_role(*self.shape, lambda *args: value, config=self.cfg)
        def bad_control(*args):
            raise RuntimeError('numerical tolerance')
        with self.assertRaisesRegex(RuntimeError, 'numerical tolerance'):
            p.fit_role(*self.shape, bad_control, config=self.cfg)

    def test_settings_validation(self):
        for kwargs in ({'quant_budget': .5}, {'min_gain': 1}, {'min_gain': -1},
                       {'training_pairs': 0}, {'confirmation_pairs': True}, {'random_field': 1}):
            with self.assertRaises(ValueError):
                p.settings(kwargs)


if __name__ == '__main__':
    unittest.main()
