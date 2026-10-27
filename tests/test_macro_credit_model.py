"""Economic/accounting invariants, independent of graphical presentation."""
import unittest
import numpy as np
from macro_credit_model import Parameters, dd_distress_bridge, liquidity_example, simulate


class ModelChecks(unittest.TestCase):
    def test_no_shock_preserves_baseline(self):
        result = simulate([0] * 5, [0] * 5)
        np.testing.assert_allclose(result["output_index"], 100)

    def test_both_markets_clear(self):
        p = Parameters()
        r = simulate([0.2, 0, 0], [-0.1, -0.05, 0], params=p, expected_inflation_gap=[-.01, 0, 0])
        np.testing.assert_allclose(r["output_gap"], r["consumption_gap"] + r["investment_gap"], atol=1e-12)
        np.testing.assert_allclose(r["real_money_gap"], p.k*r["output_gap"] - p.h*r["rate_gap"], atol=1e-12)

    def test_credit_has_effect_at_unchanged_money(self):
        r = simulate([0.2, 0], [0, 0], rho=0.5)
        self.assertLess(r["output_gap"][0], 0)
        self.assertAlmostEqual(r["output_gap"][1] / r["output_gap"][0], .5)

    def test_slower_recovery_extends_loss(self):
        fast = simulate([.2, 0, 0], [0, 0, 0], rho=.2)
        slow = simulate([.2, 0, 0], [0, 0, 0], rho=.8)
        self.assertAlmostEqual(fast["output_gap"][0], slow["output_gap"][0])
        self.assertLess(slow["output_gap"][2], fast["output_gap"][2])

    def test_liquidity_loan_does_not_create_equity(self):
        args = dict(withdrawals=45, reserves=10, liquidation_value=.7, assets=100, asset_value=80, deposits=100)
        no = liquidity_example(support=0, **args)
        yes = liquidity_example(support=20, **args)
        self.assertLess(yes["actual_liquidation"], no["actual_liquidation"])
        self.assertEqual(yes["equity_before"], -10)
        self.assertEqual(yes["equity_after_loan_before_payout"], -10)

    def test_finite_assets_leave_payment_shortfall(self):
        r = liquidity_example(200, 10, 20, .5, 100, 100, 100)
        self.assertEqual(r["actual_liquidation"], 100)
        self.assertEqual(r["payment_shortfall"], 120)

    def test_dd_bridge_normal_case_and_cap(self):
        self.assertEqual(dd_distress_bridge(m=0), 0)
        self.assertGreater(dd_distress_bridge(m=.4), 0)
        self.assertLessEqual(dd_distress_bridge(m=1), 1)

    def test_invalid_inputs_fail(self):
        with self.assertRaises(ValueError):
            simulate([.1], [0], rho=1)
        with self.assertRaises(ValueError):
            simulate([.1], [float('nan')])
        with self.assertRaises(ValueError):
            Parameters(c=1)


if __name__ == "__main__":
    unittest.main()
