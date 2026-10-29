"""Tests for decisions that change proof work, coverage and public trust."""
import copy
import unittest

import numpy as np

from commitjl.cost_aware import Planner, ProofCosts, exact_distances
from commitjl.group_change_bounds import exact_old_correction_bounds
from validation.verify_cost_aware import complete_queries, validate_pins, validate_routes


class CostAwareTests(unittest.TestCase):
    def setUp(self):
        self.rows = [[0, 0, 1], [4, 0, 1], [7, 7, 2], [1, 7, 2]]
        self.old = [[0, 0, 0], [8, 8, 0]]
        self.model = dict(version='bound-audit-v1', method='learned_energy', n=4, d=3, k=2,
                          input_bits=4, basis_scale=8, input_scale='1', centers=self.old,
                          basis=[[8, 0, 0], [0, 8, 0]], tail_width=0, tail_rows=[], tail_signs=[], seed='')
        self.costs = ProofCosts(100, 1, 1)

    def test_cached_bounds_identical_to_original_integer_reference(self):
        planner = Planner(self.model, self.rows, self.costs)
        for new in [self.old, [[1, 0, 0], [7, 8, 0]], [[0, 0, 8], [8, 8, 8]]]:
            a = planner.bounds(new)
            b = exact_old_correction_bounds(self.model, self.rows, planner.prepared, self.old, new)
            for key in ['assigned_scaled', 'cost_denominator', 'change_lower_scaled', 'change_upper_scaled']:
                self.assertEqual(a[key], b[key])
            np.testing.assert_array_equal(a['row_lower'], b['row_lower'])
            np.testing.assert_array_equal(a['row_upper'], b['row_upper'])

    def test_actual_integer_decisions_and_complete_routing(self):
        rng = np.random.default_rng(2921)
        centers = [self.old]+rng.integers(-6, 14, (80, 2, 3)).tolist()
        planner = Planner(self.model, self.rows, self.costs)
        plan = planner.batch(centers)
        validate_routes({k: plan[k] for k in ['certificate', 'full']}, len(centers))
        old_cost = int(exact_distances(self.rows, self.old).min(1).sum())
        for c, record in zip(centers, plan['records']):
            truth = int(exact_distances(self.rows, c).min(1).sum())-old_cost
            expected = 'worsens' if truth > 0 else 'improves' if truth < 0 else 'equal'
            if record['path'] == 'certificate':
                self.assertEqual(record['predicted_decision'], expected)
                self.assertLessEqual(len(record['refinement_indices']), 16)
        self.assertTrue(plan['certificate'])

    def test_expensive_or_unamortized_plans_go_directly_full(self):
        planner = Planner(self.model, self.rows, ProofCosts(1, 100, 100))
        plan = planner.batch([self.old])
        self.assertEqual(plan['full'], [0])
        self.assertEqual(plan['records'][0]['refinement_indices'], [])
        planner = Planner(self.model, self.rows, self.costs)
        plan = planner.batch([self.old], extra_setup_seconds=1000)
        self.assertEqual(plan['full'], [0])
        self.assertEqual(plan['records'][0]['reason'], 'batch_setup_not_amortized')

    def test_batch_key_load_cost_can_cancel_per_request_savings(self):
        # Every query is cheap on its own, yet loading a second proof system
        # costs more than this small batch saves. A larger batch amortizes it.
        planner = Planner(self.model, self.rows,
                          ProofCosts(10, 5, 1, certificate_batch_seconds=20))
        small = planner.batch([self.old]*2)
        self.assertEqual(small['certificate'], [])
        self.assertEqual(small['full'], [0, 1])
        self.assertEqual(small['records'][0]['reason'], 'batch_overhead_or_margin')
        large = planner.batch([self.old]*8)
        self.assertEqual(large['certificate'], list(range(8)))
        self.assertEqual(large['full'], [])

    def test_zero_correction_budget_keeps_only_conclusive_base(self):
        planner = Planner(self.model, self.rows, self.costs, max_corrections=0)
        for new in [self.old, [[1, 0, 0], [7, 8, 0]]]:
            record = planner.request(new, 0)
            self.assertFalse(record['refinement_indices'])
            if record['predicted_decision'] == 'inconclusive':
                self.assertEqual(record['path'], 'full')

    def test_overflow_safe_integer_distances(self):
        value = 2**40
        d = exact_distances([[value, -value]], [[-value, value], [value, value]])
        self.assertEqual(d.tolist(), [[8*value**2, 4*value**2]])

    def test_missing_duplicate_wrong_type_and_reordered_routes_rejected(self):
        validate_routes(dict(certificate=[0, 2], full=[1, 3]), 4)
        for routes in [dict(certificate=[0], full=[2, 3]),
                       dict(certificate=[0, 1], full=[1, 2, 3]),
                       dict(certificate=[True], full=[0, 2, 3]),
                       dict(certificate=[2, 0], full=[1, 3]),
                       dict(certificate=[0, 4], full=[1, 2, 3])]:
            with self.assertRaises(ValueError):
                validate_routes(routes, 4)

    def test_query_coverage_rejects_omission_and_duplication(self):
        complete_queries([[0, 1], [2, 3]], 4)
        for queries in [[[0, 1]], [[0, 1], [1, 2]], [[2, 3], [0, 1]], [[0, 1], []]]:
            with self.assertRaises(ValueError):
                complete_queries(queries, 4)

    def test_cross_branch_root_scale_reference_and_task_must_match(self):
        full = dict(model=dict(self.model, method='full'), original_root='17')
        cert = dict(version='group-compact-v3', owner=dict(model=self.model, original_root='17'))
        pin, task = dict(full=full, certificate=cert), dict(comparison_base=self.old, reuse_centers=[self.old])
        self.assertEqual(validate_pins(pin, task), 4)
        for field, value in [('original_root', '19'), ('input_scale', '2'), ('n', 3), ('centers', [[1, 0, 0], [8, 8, 0]])]:
            wrong = copy.deepcopy(pin)
            if field == 'original_root': wrong['full'][field] = value
            else: wrong['full']['model'][field] = value
            with self.assertRaises(ValueError): validate_pins(wrong, task)

    def test_bad_costs_and_shapes_rejected(self):
        for args in [(0, 1, 1), (1, float('nan'), 1), (1, 1, -1)]:
            with self.assertRaises(ValueError): ProofCosts(*args)
        with self.assertRaises(ValueError): ProofCosts(1, 1, 1, margin=1)
        with self.assertRaises(ValueError): Planner(self.model, self.rows, self.costs, 17)
        with self.assertRaises(ValueError): Planner(self.model, self.rows[:2], self.costs)


if __name__ == '__main__':
    unittest.main()
