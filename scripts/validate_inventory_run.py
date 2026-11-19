"""Independently verify a completed inventory bundle from saved row-level evidence."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd


def checksum(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def validate(root, path):
    manifest = json.loads(path.read_text())
    assert manifest["status"] == "optimized"
    bundle = path.parent
    for name, digest in manifest["artifact_sha256"].items():
        assert checksum(bundle / name) == digest, name
    for name, digest in manifest["code_sha256"].items():
        assert checksum(root / "src/supply_chain_opt_engine" / name) == digest, name
    source = root / "data/interim" / manifest["source"]["ingestion_run_id"]
    sales = pd.read_csv(source / "fact_daily_sales.csv", parse_dates=["date"])
    sales["unique_id"] = sales.store_id + "/" + sales.item_id
    economics = pd.read_csv(source / "dim_sku_economics.csv").set_index("item_id")
    ingest = json.loads((source / "manifest.json").read_text())
    assert checksum(source / "manifest.json") == manifest["source"]["manifest_sha256"]
    for filename in ["fact_daily_sales.csv", "dim_sku_economics.csv"]:
        assert checksum(source / filename) == ingest["artifact_sha256"][filename]
    all_metrics = pd.read_csv(bundle / "policy_metrics.csv")
    results = []
    total_orders = 0
    max_error = 0.0
    max_capacity_violation = 0.0
    for phase in manifest["phases"]:
        name = phase["phase"]
        directory = bundle / name
        scenario = root / manifest["scenario_directory"] / name
        sm = json.loads((scenario / "manifest.json").read_text())
        assert checksum(scenario / "manifest.json") == manifest["scenario_manifest_sha256"][name]
        for filename, digest in sm["artifact_sha256"].items():
            assert checksum(scenario / filename) == digest
        catalog = (
            pd.read_csv(scenario / "catalog.csv").sort_values("unique_id").reset_index(drop=True)
        )
        ids = catalog.unique_id.tolist()
        n = len(ids)
        origin = pd.Timestamp(sm["origin"])
        dates = pd.date_range(origin, periods=phase["evaluation_days"] + 1)[1:]
        cap = (
            pd.read_csv(scenario / "store_capacity.csv").set_index("store_id").capacity_m3.to_dict()
        )
        init = (
            pd.read_csv(scenario / "starting_inventory.csv")
            .set_index("unique_id")
            .reindex(ids)
            .units.to_numpy()
        )
        pipeline = pd.read_csv(scenario / "pipeline_receipts.csv")
        receipts = np.zeros((n, len(dates)))
        for row in pipeline.itertuples():
            assert pd.Timestamp(row.order_date) <= origin
            i = ids.index(row.unique_id)
            day = dates.get_loc(pd.Timestamp(row.arrival_date))
            assert (
                pd.Timestamp(row.arrival_date) - pd.Timestamp(row.order_date)
            ).days == catalog.lead_time_days[i]
            receipts[i, day] += row.units
        train = sales[sales.date.between(origin - pd.Timedelta(days=55), origin)]
        recent = train.pivot(index="unique_id", columns="date", values="sales").reindex(ids)
        assert recent.shape == (n, 56) and recent.notna().all().all()
        np.testing.assert_allclose(recent.mean(axis=1), catalog.recent_mean)
        np.testing.assert_allclose(recent.std(axis=1, ddof=1), catalog.recent_std)
        for col in [
            "unit_purchase_cost",
            "daily_holding_cost",
            "stockout_penalty",
            "unit_volume_m3",
            "fixed_order_cost",
            "minimum_order_quantity",
        ]:
            expected = economics.reindex(catalog.item_id)[col].to_numpy()
            if col == "stockout_penalty":
                expected = expected * manifest["config"].get("shortage_penalty_multiplier", 1.0)
            np.testing.assert_array_equal(catalog[col], expected)
        schedule = np.load(directory / "order_schedules.npz", allow_pickle=False)
        np.testing.assert_array_equal(schedule["node_ids"], ids)
        exported = pd.read_csv(directory / "purchase_orders.csv")
        assert not exported.duplicated(["policy", "unique_id", "order_date"]).any()
        total_orders += len(exported)
        trace = pd.read_parquet(directory / "inventory_trace.parquet")
        realized_actual = None
        common_forecast = None
        for (policy, evaluation), part in trace.groupby(["policy", "evaluation"]):
            assert len(part) == n * len(dates)
            assert not part.duplicated(["unique_id", "ds"]).any()
            part = part.set_index(["unique_id", "ds"]).reindex(
                pd.MultiIndex.from_product([ids, dates])
            )

            def matrix(col, part=part, n=n, dates=dates):
                a = part[col].to_numpy().reshape(n, len(dates))
                assert np.isfinite(a).all() and a.min() >= -1e-8
                return a

            demand = matrix("demand")
            orders = schedule[policy]
            assert orders.shape == (n, phase["decision_days"])
            assert np.equal(orders, np.floor(orders)).all() and orders.min() >= 0
            assert (
                (orders == 0) | (orders >= catalog.minimum_order_quantity.to_numpy()[:, None])
            ).all()
            arr = receipts.copy()
            for i, t in zip(*np.nonzero(orders), strict=True):
                arrival = t + int(catalog.lead_time_days[i])
                assert arrival < len(dates)
                arr[i, arrival] += orders[i, t]
            np.testing.assert_allclose(matrix("arrivals"), arr, rtol=0, atol=1e-8)
            padded = np.zeros((n, len(dates)))
            padded[:, : orders.shape[1]] = orders
            np.testing.assert_array_equal(matrix("order_quantity"), padded)
            fulfilled = np.empty_like(demand)
            inventory = np.empty_like(demand)
            stock = init.astype(float).copy()
            for day in range(len(dates)):
                available = stock + arr[:, day]
                fulfilled[:, day] = np.minimum(available, demand[:, day])
                stock = available - fulfilled[:, day]
                inventory[:, day] = stock
            lost = demand - fulfilled
            for col, a in [("inventory", inventory), ("fulfilled", fulfilled), ("unmet", lost)]:
                err = float(np.max(np.abs(a - matrix(col))))
                max_error = max(max_error, err)
                assert err < 1e-8, (policy, evaluation, col, err)
            holding = inventory * catalog.daily_holding_cost.to_numpy()[:, None]
            shortage = lost * catalog.stockout_penalty.to_numpy()[:, None]
            fees = (padded > 0) * catalog.fixed_order_cost.to_numpy()[:, None]
            capital = inventory * catalog.unit_purchase_cost.to_numpy()[:, None]
            for col, a in [
                ("holding_cost", holding),
                ("stockout_cost", shortage),
                ("fixed_order_cost", fees),
                ("inventory_value", capital),
            ]:
                np.testing.assert_allclose(matrix(col), a, rtol=1e-10, atol=1e-8)
            violation = 0.0
            for store, capacity in cap.items():
                mask = catalog.store_id.eq(store).to_numpy()
                violation = max(
                    violation,
                    float(
                        np.max(catalog.unit_volume_m3.to_numpy()[mask] @ inventory[mask] - capacity)
                    ),
                )
            max_capacity_violation = max(max_capacity_violation, violation)
            if evaluation == "forecast":
                if policy == "lightgbm_mint_milp" or not manifest.get("forecast_comparison"):
                    assert violation <= 1e-5, (policy, evaluation, violation)
                if manifest.get("forecast_comparison"):
                    if common_forecast is None:
                        common_forecast = demand.copy()
                    else:
                        np.testing.assert_array_equal(demand, common_forecast)
            if evaluation == "realized":
                expected = (
                    sales[sales.date.isin(dates)]
                    .pivot(index="unique_id", columns="date", values="sales")
                    .reindex(index=ids, columns=dates)
                    .to_numpy()
                )
                np.testing.assert_array_equal(demand, expected)
                if realized_actual is None:
                    realized_actual = demand.copy()
                else:
                    np.testing.assert_array_equal(demand, realized_actual)
            for scope, end in [
                ("decision_window", orders.shape[1]),
                ("including_tail", len(dates)),
            ]:
                row = all_metrics[
                    (all_metrics.phase == name)
                    & (all_metrics.policy == policy)
                    & (all_metrics.evaluation == evaluation)
                    & (all_metrics.scope == scope)
                ].iloc[0]
                scope_violation = 0.0
                for store, capacity in cap.items():
                    mask = catalog.store_id.eq(store).to_numpy()
                    used = catalog.unit_volume_m3.to_numpy()[mask] @ inventory[mask, :end]
                    scope_violation = max(scope_violation, float(np.max(used - capacity)))
                assert abs(row.max_capacity_violation_m3 - scope_violation) < 1e-7
                assert row.capacity_feasible == (scope_violation <= 1e-5)
                if "service_feasible" in row.index:
                    target = manifest["config"].get("minimum_fill_rate", 0.0)
                    fills = []
                    shortfall = 0.0
                    for store in cap:
                        mask = catalog.store_id.eq(store).to_numpy()
                        d = demand[mask, :end].sum()
                        u = lost[mask, :end].sum()
                        shortfall = max(shortfall, u - (1 - target) * d)
                        if d:
                            fills.append(1 - u / d)
                    assert row.minimum_fill_rate == target
                    assert row.service_feasible == (shortfall <= 1e-5)
                    assert abs(row.max_service_shortfall_units - shortfall) < 1e-7
                    if fills:
                        np.testing.assert_allclose(row.minimum_store_fill_rate, min(fills))
                    else:
                        assert pd.isna(row.minimum_store_fill_rate)
                    assert row.comparison_eligible == (
                        row.capacity_feasible and row.service_feasible
                    )
                    if policy == "lightgbm_mint_milp" and evaluation == "forecast":
                        assert row.service_feasible, (name, scope, target, fills)
                for key, value in {
                    "holding_cost": holding[:, :end].sum(),
                    "stockout_cost": shortage[:, :end].sum(),
                    "fixed_order_cost": fees[:, :end].sum(),
                    "total_cost": (holding[:, :end] + shortage[:, :end] + fees[:, :end]).sum(),
                    "average_inventory_value": capital[:, :end].sum() / end,
                    "fill_rate": (
                        fulfilled[:, :end].sum() / demand[:, :end].sum()
                        if demand[:, :end].sum()
                        else np.nan
                    ),
                }.items():
                    np.testing.assert_allclose(row[key], value, rtol=1e-10, atol=1e-8)
            if evaluation == "forecast":
                exp = exported[exported.policy == policy]
                assert len(exp) == (orders > 0).sum()
                for row in exp.itertuples():
                    assert row.scenario_id == sm["scenario_id"]
                    i = ids.index(row.unique_id)
                    day = dates.get_loc(pd.Timestamp(row.order_date))
                    assert orders[i, day] == row.quantity
                    assert (
                        pd.Timestamp(row.arrival_date)
                        == dates[day + int(catalog.lead_time_days[i])]
                    )
                    assert row.unit_purchase_cost == catalog.unit_purchase_cost[i]
                    assert abs(row.purchase_value - row.quantity * row.unit_purchase_cost) < 1e-7
        solver = json.loads((directory / "solver_diagnostics.json").read_text())
        primary = all_metrics[
            (all_metrics.phase == name)
            & (all_metrics.policy == "lightgbm_mint_milp")
            & (all_metrics.evaluation == "forecast")
            & (all_metrics.scope == "including_tail")
        ].iloc[0]
        incumbent = sum(d["incumbent_objective"] for d in solver)
        bounds = [d["objective_bound"] for d in solver]
        bound = sum(bounds) if all(b is not None and np.isfinite(b) for b in bounds) else None
        assert all(d["has_feasible_solution"] for d in solver)
        assert abs(incumbent - primary.total_cost) < 1e-7
        gap = (
            max(0.0, incumbent - bound) / max(abs(incumbent), 1e-10) if bound is not None else None
        )
        results.append(
            {
                "phase": name,
                "series": n,
                "policies": phase["policies"],
                "solver_objective": incumbent,
                "solver_bound": bound,
                "relative_gap": gap,
                "solve_seconds": sum(d["solve_seconds"] for d in solver),
                "phase_seconds": phase["seconds"],
                "exported_orders": len(exported),
            }
        )
    if "comparison_eligible" in all_metrics:
        for _, part in all_metrics.groupby(["phase", "evaluation", "scope"]):
            reference = part[part.policy.eq("lightgbm_p50_order_up_to")]
            if (
                len(reference)
                and reference.comparison_eligible.iloc[0]
                and reference.total_cost.iloc[0] > 0
            ):
                good = part.comparison_eligible
                np.testing.assert_allclose(
                    part.loc[good, "cost_saving_vs_global_p50"],
                    1 - part.loc[good, "total_cost"] / reference.total_cost.iloc[0],
                )
                assert part.loc[~good, "cost_saving_vs_global_p50"].isna().all()
            else:
                assert part.cost_saving_vs_global_p50.isna().all()
    return {
        "run_id": manifest["run_id"],
        "manifest_sha256": checksum(path),
        "artifacts_verified": len(manifest["artifact_sha256"]),
        "source_files_verified": len(manifest["code_sha256"]),
        "phases": results,
        "purchase_order_rows": total_orders,
        "capacity_infeasible_replays": all_metrics.loc[
            ~all_metrics.capacity_feasible,
            ["phase", "policy", "evaluation", "scope", "max_capacity_violation_m3"],
        ].to_dict("records"),
        "max_replay_error": max_error,
        "max_capacity_violation_m3": max_capacity_violation,
        "checks": [
            "All checksums and scenario lineage verified",
            "Pre-origin 56-day statistics and shared item economics verified",
            "Pipeline orders predate the origin; arrivals obey shocked lead times",
            "Integer orders, MOQs, receipt dates and purchase exports agree",
            "Every inventory/fulfillment row independently replayed",
            "All policies use identical realized demand and scenario inputs",
            "Cost, fill rate and inventory capital recomputed for both windows",
            "Solver objectives and certified lower bounds match exported plans",
        ],
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifest", type=Path)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    report = validate(root, args.manifest.resolve())
    output = root / "artifacts/metrics/optimization_validation.json"
    output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
