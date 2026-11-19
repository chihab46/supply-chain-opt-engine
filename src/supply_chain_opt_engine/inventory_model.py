"""Pyomo inventory MILP, bounded order quantities and guarded HiGHS extraction."""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass

import numpy as np
import pyomo.environ as pyo
from pyomo.contrib.appsi.base import TerminationCondition
from pyomo.contrib.appsi.solvers import Highs

from supply_chain_opt_engine.inventory_data import InventoryInputs


@dataclass
class InventorySolution:
    orders: np.ndarray | None
    inventory: np.ndarray | None
    unmet: np.ndarray | None
    diagnostics: dict


def order_bounds(data: InventoryInputs):
    """Finite bounds from remaining demand, MOQ and storage on the arrival day."""
    c = data.catalog
    lead = c.lead_time_days.to_numpy(dtype=int)
    moq = c.minimum_order_quantity.to_numpy(dtype=int)
    volume = c.unit_volume_m3.to_numpy()
    capacity = c.store_id.map(data.capacity).to_numpy()
    penalty = c.stockout_penalty.to_numpy()
    fee = c.fixed_order_cost.to_numpy()
    bounds = np.zeros((data.n, data.decision_days), dtype=int)
    for i in range(data.n):
        for t in range(data.decision_days):
            arrival = t + lead[i]
            if arrival >= data.h:
                continue
            remaining = float(data.demand[i, arrival:].sum())
            # Without terminal minimum stock, an order cannot save more than all
            # remaining lost-sales penalties. Holding and capacity only reduce its value.
            if remaining == 0 or (data.minimum_fill_rate == 0 and remaining * penalty[i] <= fee[i]):
                continue
            useful = max(moq[i], int(np.ceil(remaining)))
            storage = int(np.floor(capacity[i] / volume[i] + data.demand[i, arrival] + 1e-9))
            if storage >= moq[i]:
                bounds[i, t] = min(useful, storage)
    return bounds


def build_inventory_model(data: InventoryInputs):
    c = data.catalog
    lead = c.lead_time_days.to_numpy(dtype=int)
    moq = c.minimum_order_quantity.to_numpy(dtype=int)
    volume = c.unit_volume_m3.to_numpy()
    holding = c.daily_holding_cost.to_numpy()
    penalty = c.stockout_penalty.to_numpy()
    fee = c.fixed_order_cost.to_numpy()
    upper = order_bounds(data)
    m = pyo.ConcreteModel("lost_sales_replenishment")
    m.skus = pyo.RangeSet(0, data.n - 1)
    m.days = pyo.RangeSet(0, data.h - 1)
    m.order_index = pyo.Set(dimen=2, initialize=list(zip(*np.nonzero(upper), strict=True)))
    m.shortage_index = pyo.Set(
        dimen=2, initialize=list(zip(*np.nonzero(data.demand > 0), strict=True))
    )
    # Bound stock using both per-item capacity and reachable cumulative supply.
    imax = np.empty((data.n, data.h))
    for i in range(data.n):
        incoming = np.zeros(data.h)
        for t in range(data.decision_days):
            if upper[i, t]:
                incoming[t + lead[i]] += upper[i, t]
        reachable = data.initial[i] + np.cumsum(data.receipts[i] + incoming)
        imax[i] = np.minimum(data.capacity[c.store_id.iloc[i]] / volume[i], reachable)
    m.order = pyo.Var(
        m.order_index, domain=pyo.NonNegativeIntegers, bounds=lambda _, i, t: (0, int(upper[i, t]))
    )
    m.trigger = pyo.Var(m.order_index, domain=pyo.Binary)
    m.inventory = pyo.Var(
        m.skus, m.days, domain=pyo.NonNegativeReals, bounds=lambda _, i, t: (0, float(imax[i, t]))
    )
    m.unmet = pyo.Var(
        m.skus,
        m.days,
        domain=pyo.NonNegativeReals,
        bounds=lambda _, i, t: (0, float(data.demand[i, t])),
    )
    m.shortage = pyo.Var(m.shortage_index, domain=pyo.Binary)

    def balance(_, i, t):
        previous = data.initial[i] if t == 0 else m.inventory[i, t - 1]
        arrival = m.order[i, t - lead[i]] if (i, t - lead[i]) in m.order_index else 0
        return m.inventory[i, t] - m.unmet[i, t] == (
            previous + float(data.receipts[i, t]) + arrival - float(data.demand[i, t])
        )

    m.balance = pyo.Constraint(m.skus, m.days, rule=balance)
    m.moq = pyo.Constraint(
        m.order_index, rule=lambda _, i, t: m.order[i, t] >= int(moq[i]) * m.trigger[i, t]
    )
    m.order_limit = pyo.Constraint(
        m.order_index, rule=lambda _, i, t: m.order[i, t] <= int(upper[i, t]) * m.trigger[i, t]
    )
    m.no_rationing = pyo.Constraint(
        m.shortage_index,
        rule=lambda _, i, t: m.inventory[i, t] <= float(imax[i, t]) * (1 - m.shortage[i, t]),
    )
    m.lost_sales = pyo.Constraint(
        m.shortage_index,
        rule=lambda _, i, t: m.unmet[i, t] <= float(data.demand[i, t]) * m.shortage[i, t],
    )
    m.store_days = pyo.Set(
        dimen=2, initialize=[(s, t) for s in sorted(data.capacity) for t in range(data.h)]
    )
    groups = {s: np.flatnonzero(c.store_id.eq(s)) for s in data.capacity}
    m.capacity = pyo.Constraint(
        m.store_days,
        rule=lambda _, s, t: (
            sum(float(volume[i]) * m.inventory[i, t] for i in groups[s]) <= data.capacity[s]
        ),
    )
    # A store-wide floor permits the solver to allocate service across its items.
    # Enforce both windows so tail demand cannot hide poor service in days 1-28.
    m.service_index = pyo.Set(
        dimen=2,
        initialize=[
            (store, end)
            for store in sorted(data.capacity)
            for end in sorted({data.decision_days, data.h})
            if data.minimum_fill_rate > 0 and data.demand[groups[store], :end].sum() > 0
        ],
    )
    m.service = pyo.Constraint(
        m.service_index,
        rule=lambda _, store, end: (
            sum(m.unmet[i, t] for i in groups[store] for t in range(end))
            <= (1 - data.minimum_fill_rate) * float(data.demand[groups[store], :end].sum())
        ),
    )
    m.cost = pyo.Objective(
        expr=(
            sum(float(fee[i]) * m.trigger[i, t] for i, t in m.order_index)
            + sum(
                float(holding[i]) * m.inventory[i, t] + float(penalty[i]) * m.unmet[i, t]
                for i in range(data.n)
                for t in range(data.h)
            )
        ),
        sense=pyo.minimize,
    )
    return m


def solution_acceptable(termination, objective):
    return (
        termination
        in {
            TerminationCondition.optimal,
            TerminationCondition.maxTimeLimit,
            TerminationCondition.maxIterations,
            TerminationCondition.objectiveLimit,
        }
        and objective is not None
        and np.isfinite(objective)
    )


def validate_solution(data, orders, inventory, unmet, objective=None, atol=1e-5):
    """Independently enforce physical fulfillment, integrality and objective accounting."""
    from supply_chain_opt_engine.simulation import inventory_trace, service_summary

    if orders.shape != (data.n, data.decision_days) or not np.isfinite(orders).all():
        raise ValueError("Invalid extracted order schedule.")
    if (orders < -atol).any() or np.max(np.abs(orders - np.rint(orders))) > atol:
        raise ValueError("Order quantities must be nonnegative integers.")
    orders = np.rint(orders).astype(int)
    if (orders > order_bounds(data)).any():
        raise ValueError("Orders exceed their finite planning bounds.")
    moq = data.catalog.minimum_order_quantity.to_numpy()[:, None]
    if ((orders > 0) & (orders < moq)).any():
        raise ValueError("A positive order violates MOQ.")
    trace = inventory_trace(data, orders)
    if not np.allclose(trace["inventory"], inventory, rtol=1e-8, atol=atol):
        raise ValueError("Extracted inventory violates work-conserving flow balance.")
    if not np.allclose(trace["unmet"], unmet, rtol=1e-8, atol=atol):
        raise ValueError("Extracted lost sales violate physical fulfillment.")
    if trace["capacity_violation_m3"] > atol:
        raise ValueError("Extracted inventory violates store capacity.")
    service = {}
    for scope, end in [("decision_window", data.decision_days), ("including_tail", data.h)]:
        summary = service_summary(data, trace["unmet"], end, atol=atol)
        if not summary["service_feasible"]:
            raise ValueError("Extracted inventory violates the per-store minimum fill rate.")
        service[scope] = summary
    value = float(
        (trace["inventory"] * data.catalog.daily_holding_cost.to_numpy()[:, None]).sum()
        + (trace["unmet"] * data.catalog.stockout_penalty.to_numpy()[:, None]).sum()
        + ((orders > 0) * data.catalog.fixed_order_cost.to_numpy()[:, None]).sum()
    )
    if objective is not None and not np.isclose(value, objective, rtol=1e-7, atol=atol):
        raise ValueError("Solver objective differs from independently recomputed costs.")
    return {
        "objective_recomputed": value,
        "max_capacity_violation_m3": trace["capacity_violation_m3"],
        "max_inventory_error": float(np.max(np.abs(trace["inventory"] - inventory))),
        "max_unmet_error": float(np.max(np.abs(trace["unmet"] - unmet))),
        "service": service,
    }


def solve_inventory(
    data,
    *,
    time_limit=60.0,
    mip_gap=0.01,
    seed=42,
    threads=1,
    log_path=None,
    solver=None,
    start_orders=None,
):
    start = time.perf_counter()
    m = build_inventory_model(data)
    build_seconds = time.perf_counter() - start
    solver = Highs() if solver is None else solver
    solver.config.load_solution = False
    solver.config.log_level = logging.DEBUG
    solver.config.time_limit = time_limit
    solver.config.mip_gap = mip_gap
    solver.highs_options.update(threads=threads, random_seed=seed, mip_feasibility_tolerance=1e-7)
    if log_path is not None:
        solver.highs_options["log_file"] = str(log_path)
    from supply_chain_opt_engine.simulation import (
        inventory_trace,
        service_start_orders,
        service_summary,
    )

    if start_orders is None and data.minimum_fill_rate > 0:
        try:
            start_orders = service_start_orders(data)
        except ValueError:
            pass  # HiGHS must establish feasibility when no heuristic plan fits.
    seed_orders = (
        np.zeros((data.n, data.decision_days), dtype=int) if start_orders is None else start_orders
    )
    seed_trace = inventory_trace(data, seed_orders)
    if (
        seed_trace["capacity_violation_m3"] <= 1e-7
        and (seed_orders <= order_bounds(data)).all()
        and all(
            service_summary(data, seed_trace["unmet"], end)["service_feasible"]
            for end in {data.decision_days, data.h}
        )
    ):
        for i, t in m.order_index:
            m.order[i, t].value = int(seed_orders[i, t])
            m.trigger[i, t].value = int(seed_orders[i, t] > 0)
        for i in range(data.n):
            for t in range(data.h):
                m.inventory[i, t].value = float(seed_trace["inventory"][i, t])
                m.unmet[i, t].value = float(seed_trace["unmet"][i, t])
        for i, t in m.shortage_index:
            m.shortage[i, t].value = int(seed_trace["unmet"][i, t] > 0)
        solver.config.warmstart = True
    solve_start = time.perf_counter()
    result = solver.solve(m)
    incumbent = result.best_feasible_objective
    bound = result.best_objective_bound
    if bound is not None and not np.isfinite(bound):
        bound = None
    if incumbent is not None and not np.isfinite(incumbent):
        incumbent = None
    status = result.termination_condition
    gap = None
    if incumbent is not None and bound is not None and np.isfinite([incumbent, bound]).all():
        gap = max(0.0, float(incumbent - bound)) / max(abs(incumbent), 1e-10)
    diag = {
        "termination": status.name,
        "incumbent_objective": incumbent,
        "objective_bound": bound,
        "relative_gap": gap,
        "build_seconds": build_seconds,
        "solve_seconds": time.perf_counter() - solve_start,
        "variables": sum(1 for _ in m.component_data_objects(pyo.Var)),
        "constraints": sum(1 for _ in m.component_data_objects(pyo.Constraint)),
        "order_variables": len(m.order_index),
        "has_feasible_solution": False,
        "minimum_fill_rate": data.minimum_fill_rate,
    }
    if not solution_acceptable(status, incumbent):
        return InventorySolution(None, None, None, diag)
    result.solution_loader.load_vars()
    orders = np.zeros((data.n, data.decision_days))
    for i, t in m.order_index:
        orders[i, t] = pyo.value(m.order[i, t])
    inventory = np.array(
        [[pyo.value(m.inventory[i, t]) for t in range(data.h)] for i in range(data.n)]
    )
    unmet = np.array([[pyo.value(m.unmet[i, t]) for t in range(data.h)] for i in range(data.n)])
    diag.update(validate_solution(data, orders, inventory, unmet, incumbent))
    diag["has_feasible_solution"] = True
    diag["proven_optimal"] = status == TerminationCondition.optimal and (gap or 0) <= 1e-8
    # "optimal" is HiGHS termination within configured MIP tolerances. Preserve the bound/gap.
    return InventorySolution(np.rint(orders).astype(int), inventory, unmet, diag)


def repair_capacity(data, orders, *, max_iterations=20000):
    """Trim a schedule until every store respects end-of-day capacity.

    On the earliest overflowing day, the store item with the lowest stockout
    penalty per cubic metre loses units from its latest order that has already
    arrived. Trimmed orders below MOQ are dropped. Smaller orders never raise
    later inventory, so the loop ends at a feasible schedule; the zero-order
    schedule is returned if the iteration cap is reached.
    """
    from supply_chain_opt_engine.simulation import inventory_trace

    orders = np.array(orders, dtype=int, copy=True)
    c = data.catalog
    lead = c.lead_time_days.to_numpy(dtype=int)
    moq = c.minimum_order_quantity.to_numpy(dtype=int)
    volume = c.unit_volume_m3.to_numpy()
    value = c.stockout_penalty.to_numpy() / volume
    stores = c.store_id.to_numpy()
    for _ in range(max_iterations):
        trace = inventory_trace(data, orders)
        if trace["capacity_violation_m3"] <= 1e-7:
            return orders
        inventory = trace["inventory"]
        worst = None
        for store, cap in data.capacity.items():
            mask = stores == store
            used = volume[mask] @ inventory[mask]
            over = np.flatnonzero(used - cap > 1e-7)
            if over.size and (worst is None or over[0] < worst[1]):
                worst = (store, int(over[0]), float(used[over[0]] - cap))
        store, day, excess = worst
        candidates = [
            i
            for i in np.flatnonzero(stores == store)
            if inventory[i, day] > 1e-9
            and any(orders[i, s] > 0 and s + lead[i] <= day for s in range(data.decision_days))
        ]
        if not candidates:
            break
        i = min(candidates, key=lambda j: value[j])
        s = max(s for s in range(data.decision_days) if orders[i, s] > 0 and s + lead[i] <= day)
        cut = min(
            int(orders[i, s]),
            max(1, int(np.ceil(excess / volume[i] - 1e-9))),
            max(1, int(np.ceil(inventory[i, day] - 1e-9))),
        )
        orders[i, s] -= cut
        if 0 < orders[i, s] < moq[i]:
            orders[i, s] = 0
    return np.zeros_like(orders)


ITEM_BUDGET_SHARE = 0.5


def coupled_store_fallback(data, item_orders, item_bound, children, *, strategy, **solve_options):
    """Solve the coupled store MILP from the best available feasible schedule.

    Candidates are the item schedule after capacity repair and the batched
    order-up-to heuristic used for service floors. Either is a feasible
    incumbent, so a short budget can no longer collapse to the trivial
    zero-order start. The better of the MILP incumbent and the warm start is
    returned. ``item_bound`` is the sum of item-relaxation lower bounds, valid
    because it drops only the shared capacity constraint.
    """
    from supply_chain_opt_engine.simulation import inventory_trace, service_start_orders

    candidates = {"capacity_repaired_item_schedule": repair_capacity(data, item_orders)}
    try:
        heuristic = service_start_orders(data)
    except ValueError:
        heuristic = None
    if (
        heuristic is not None
        and (heuristic <= order_bounds(data)).all()
        and inventory_trace(data, heuristic)["capacity_violation_m3"] <= 1e-7
    ):
        candidates["batched_order_up_to_heuristic"] = heuristic
    scored = []
    for source, orders in candidates.items():
        trace = inventory_trace(data, orders)
        value = validate_solution(data, orders, trace["inventory"], trace["unmet"])
        scored.append((value["objective_recomputed"], source, orders, trace))
    start_objective, start_source, start_orders, start = min(scored, key=lambda s: s[0])
    result = solve_inventory(data, start_orders=start_orders, **solve_options)
    diag = result.diagnostics
    if result.orders is None or diag["incumbent_objective"] > start_objective + 1e-6:
        checks = validate_solution(
            data, start_orders, start["inventory"], start["unmet"], start_objective
        )
        diag = {
            **diag,
            **checks,
            "incumbent_objective": start_objective,
            "has_feasible_solution": True,
            "proven_optimal": False,
            "incumbent_source": start_source,
        }
        result = InventorySolution(start_orders, start["inventory"], start["unmet"], diag)
    else:
        diag["incumbent_source"] = "coupled_milp"
    bound = diag.get("objective_bound")
    if np.isfinite(item_bound) and (bound is None or item_bound > bound):
        diag["objective_bound"] = item_bound
    if diag["objective_bound"] is not None:
        incumbent = diag["incumbent_objective"]
        diag["relative_gap"] = max(0.0, incumbent - diag["objective_bound"]) / max(
            abs(incumbent), 1e-10
        )
        diag["proven_optimal"] = diag["relative_gap"] <= 1e-8
    diag.update(
        strategy=strategy,
        item_solves=children,
        item_relaxation_bound=item_bound if np.isfinite(item_bound) else None,
        warm_start_source=start_source,
        warm_start_objective=start_objective,
    )
    diag["solve_seconds"] += sum(c["solve_seconds"] for c in children)
    return result


def solve_store(data, *, time_limit=60.0, mip_gap=0.01, seed=42, threads=1, log_path=None):
    """Use separable item bounds only when the combined schedule respects capacity.

    Removing cross-item capacity is a relaxation. Its summed objective bounds
    remain valid for the coupled store problem. A joint feasible incumbent plus
    this lower bound certifies the reported gap. Capacity conflicts trigger the
    original coupled MILP instead, warm-started from the item schedule after
    capacity repair. Item solves may use at most ITEM_BUDGET_SHARE of the time
    limit, so the coupled fallback always keeps the rest. The summed item bounds
    stay valid for that coupled solve and are reported when tighter.
    """
    from dataclasses import replace

    from supply_chain_opt_engine.simulation import inventory_trace

    if data.minimum_fill_rate > 0:
        # Item decomposition would impose a different, stronger per-item floor.
        result = solve_inventory(
            data,
            time_limit=time_limit,
            mip_gap=mip_gap,
            seed=seed,
            threads=threads,
            log_path=log_path,
        )
        result.diagnostics["strategy"] = "coupled_store_service_floor"
        return result
    if data.n == 1:
        return solve_inventory(
            data,
            time_limit=time_limit,
            mip_gap=mip_gap,
            seed=seed,
            threads=threads,
            log_path=log_path,
        )
    zero = np.zeros((data.n, data.decision_days), dtype=int)
    trace = inventory_trace(data, zero)
    if trace["capacity_violation_m3"] > 1e-7:
        return solve_inventory(
            data,
            time_limit=time_limit,
            mip_gap=mip_gap,
            seed=seed,
            threads=threads,
            log_path=log_path,
        )
    forced_cost = (trace["inventory"] * data.catalog.daily_holding_cost.to_numpy()[:, None]).sum(
        axis=1
    ) + (trace["unmet"] * data.catalog.stockout_penalty.to_numpy()[:, None]).sum(axis=1)
    objectives = forced_cost.copy()
    bounds = forced_cost.copy()
    orders = zero.copy()
    active = np.flatnonzero(order_bounds(data).any(axis=1))
    children = []
    remaining = time_limit
    item_budget = time_limit * ITEM_BUDGET_SHARE
    for counter, i in enumerate(active):
        item = replace(
            data,
            catalog=data.catalog.iloc[[i]].reset_index(drop=True),
            demand=data.demand[[i]],
            initial=data.initial[[i]],
            receipts=data.receipts[[i]],
        )
        allowance = max(0.001, item_budget / (len(active) - counter))
        result = solve_inventory(
            item,
            time_limit=allowance,
            mip_gap=mip_gap,
            seed=seed,
            threads=threads,
            log_path=log_path,
        )
        children.append({"unique_id": data.catalog.unique_id.iloc[i], **result.diagnostics})
        remaining -= result.diagnostics["solve_seconds"]
        item_budget -= result.diagnostics["solve_seconds"]
        if result.orders is None:
            # Item bounds are incomplete here, so only the coupled bound counts.
            return coupled_store_fallback(
                data,
                orders,
                -np.inf,
                children,
                strategy="coupled_after_missing_item_incumbent",
                time_limit=max(0.001, remaining),
                mip_gap=mip_gap,
                seed=seed,
                threads=threads,
                log_path=log_path,
            )
        orders[i] = result.orders[0]
        objectives[i] = result.diagnostics["incumbent_objective"]
        lower = result.diagnostics["objective_bound"]
        bounds[i] = max(0.0, lower) if lower is not None and np.isfinite(lower) else 0.0
    combined = inventory_trace(data, orders)
    obj = float(objectives.sum())
    bound = float(bounds.sum())
    gap = max(0.0, obj - bound) / max(abs(obj), 1e-10)
    if combined["capacity_violation_m3"] > 1e-7 or gap > mip_gap + 1e-9:
        # The coupled MILP keeps the remaining budget and the item bounds.
        return coupled_store_fallback(
            data,
            orders,
            bound,
            children,
            strategy=(
                "coupled_after_capacity_conflict"
                if combined["capacity_violation_m3"] > 1e-7
                else "coupled_after_item_gap"
            ),
            time_limit=max(0.001, remaining),
            mip_gap=mip_gap,
            seed=seed,
            threads=threads,
            log_path=log_path,
        )
    checks = validate_solution(data, orders, combined["inventory"], combined["unmet"], obj)
    diagnostics = {
        "strategy": "independent_items_jointly_capacity_feasible",
        "termination": "within_gap_tolerance" if gap <= mip_gap else "feasible_limit",
        "incumbent_objective": obj,
        "objective_bound": bound,
        "relative_gap": gap,
        "has_feasible_solution": True,
        "proven_optimal": gap <= 1e-8,
        "solve_seconds": sum(c["solve_seconds"] for c in children),
        "build_seconds": sum(c["build_seconds"] for c in children),
        "active_items": len(active),
        "fixed_zero_order_items": data.n - len(active),
        "item_solves": children,
        **checks,
    }
    return InventorySolution(orders, combined["inventory"], combined["unmet"], diagnostics)
