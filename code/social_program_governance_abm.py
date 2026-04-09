"""
Agent-based model of governance arrangements in social program access.

This model compares hierarchical, delegated, and adaptive governance
under heterogeneous territorial conditions, learning, congestion,
screening frictions, and capture risks.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def clip(x, lo=0.0, hi=1.0):
    return np.minimum(np.maximum(x, lo), hi)


@dataclass
class Params:
    periods: int = 40
    households: int = 600
    territories: int = 6
    neighbors: int = 6
    seed: int = 42

    benefit: float = 1.0
    eligibility_threshold: float = 0.62

    base_capacity: int = 52
    base_effort_mean: float = 0.68
    base_effort_sd: float = 0.08
    opportunism_mean: float = 0.28
    opportunism_sd: float = 0.08

    specificity: float = 0.65
    uncertainty: float = 0.20
    omega: float = 0.80

    access_cost_weight: float = 0.60
    network_info_strength: float = 0.22
    network_trust_strength: float = 0.18
    application_inertia: float = 0.10

    brokerage_mean_a: float = 1.6
    brokerage_mean_b: float = 4.0
    hierarchy_capture_scale: float = 0.18
    delegated_capture_scale: float = 0.62
    hierarchy_patronage_bias: float = 0.10
    delegated_patronage_bias: float = 0.80
    hierarchy_capture_cap: float = 0.28
    delegated_capture_cap: float = 0.50

    trust_gain_success: float = 0.07
    trust_loss_reject: float = 0.09
    trust_loss_nonapply_eligible: float = 0.02
    info_gain_success: float = 0.08
    info_gain_neighbors: float = 0.05

    # Stylised governance-risk hypotheses: hierarchy is assumed to offer
    # tighter screening/monitoring at a higher fixed coordination cost, while
    # delegation is cheaper but more porous to capture and patronage.
    hierarchy_base_monitor: float = 0.74
    delegated_base_monitor: float = 0.56

    hierarchy_fixed_cost: float = 0.19
    delegated_fixed_cost: float = 0.08
    hierarchy_hazard_slope: float = 0.14
    delegated_hazard_slope: float = 0.26

    hierarchy_screen_base: float = 0.10
    delegated_screen_base: float = 0.18
    hierarchy_screen_hazard: float = 0.07
    delegated_screen_hazard: float = 0.16

    congestion_penalty: float = 0.18
    monitor_congestion_penalty: float = 0.10
    capacity_governance_penalty_scale: float = 0.35

    hierarchy_effort_bonus: float = 0.06
    delegated_effort_penalty: float = 0.02

    cross_territory_tie_prob: float = 0.03

    # Dynamic institutional adaptation
    review_interval: int = 5
    min_tenure: int = 10
    adaptive_metric_smoothing: float = 0.35
    switch_cost_margin: float = 0.082
    adaptive_initial_mode: str = "score"

    effort_learning_rate: float = 0.18
    opportunism_learning_rate: float = 0.16
    quality_persistence: float = 0.62
    quality_base_shift: float = 0.16
    quality_effort_weight: float = 0.62
    quality_monitor_weight: float = 0.10
    quality_opportunism_weight: float = 0.22
    quality_cost_weight: float = 0.50
    quality_congestion_weight: float = 0.08
    effort_target_base: float = 0.58
    effort_quality_weight: float = 0.18
    effort_monitor_weight: float = 0.09
    effort_congestion_weight: float = 0.03
    effort_capture_weight: float = 0.08
    opportunism_target_base: float = 0.18
    opportunism_capture_weight: float = 0.42
    opportunism_monitor_weight: float = 0.16
    opportunism_quality_weight: float = 0.10
    opportunism_congestion_weight: float = 0.05

    # Comparative institutional scoring
    init_switch_margin: float = 0.04
    hysteresis_bonus: float = 0.055
    score_cost_weight: float = 0.85
    score_h_inclusion_weight: float = 1.40
    score_h_opportunism_weight: float = 1.10
    score_h_trustdef_weight: float = 0.70
    score_h_dynhaz_weight: float = 0.42
    score_h_monitor_weight: float = 0.42
    score_d_congestion_weight: float = 0.78
    score_d_demandgap_weight: float = 0.65
    score_d_trust_weight: float = 0.76
    score_d_capacity_weight: float = 0.45
    score_d_quality_weight: float = 0.30
    score_d_opportunism_weight: float = 1.00
    score_d_inclusion_weight: float = 1.50
    score_d_dynhaz_weight: float = 0.35
    score_d_adminexcl_weight: float = 0.45
    score_d_screening_weight: float = 0.55
    hierarchy_screening_drag_scale: float = 0.03
    delegated_screening_drag_scale: float = 0.16


def build_households(params: Params, rng: np.random.Generator):
    n = params.households
    territories = np.repeat(np.arange(params.territories), n // params.territories)
    if len(territories) < n:
        extra = rng.integers(0, params.territories, size=n - len(territories))
        territories = np.concatenate([territories, extra])
    rng.shuffle(territories)

    vulnerability = clip(rng.beta(2.2, 1.8, size=n))
    eligible = (vulnerability >= params.eligibility_threshold).astype(int)
    access_cost = clip(
        0.15 + 0.55 * rng.random(n) + 0.18 * (1 - vulnerability) + 0.08 * (territories / max(1, params.territories - 1)),
        0.05,
        0.95,
    )
    trust = clip(0.35 + 0.35 * rng.random(n))
    info = clip(0.20 + 0.50 * rng.random(n))
    brokerage = clip(rng.beta(params.brokerage_mean_a, params.brokerage_mean_b, size=n))

    return {
        "territory": territories,
        "vulnerability": vulnerability,
        "eligible": eligible,
        "access_cost": access_cost,
        "trust": trust,
        "info": info,
        "brokerage": brokerage,
    }


def build_offices(params: Params, rng: np.random.Generator):
    m = params.territories
    base_effort = clip(
        rng.normal(params.base_effort_mean, params.base_effort_sd, size=m),
        0.35,
        0.95,
    )
    opportunism = clip(
        rng.normal(params.opportunism_mean, params.opportunism_sd, size=m),
        0.02,
        0.70,
    )
    territorial_complexity = clip(rng.beta(2.0, 2.0, size=m), 0.05, 0.95)
    hazard = clip(params.specificity * (1 + params.omega * territorial_complexity) + params.uncertainty, 0.0, 2.5)
    return {
        "base_effort": base_effort,
        "opportunism": opportunism,
        "territorial_complexity": territorial_complexity,
        "hazard": hazard,
    }


def build_neighbors(territories: np.ndarray, params: Params, rng: np.random.Generator) -> List[np.ndarray]:
    n = len(territories)
    neighbors: List[np.ndarray] = []
    territory_to_idx = {t: np.where(territories == t)[0] for t in np.unique(territories)}
    all_idx = np.arange(n)
    for i in range(n):
        same = territory_to_idx[territories[i]]
        same = same[same != i]
        k_same = min(len(same), max(1, params.neighbors - 1))
        same_neighbors = rng.choice(same, size=k_same, replace=False) if k_same > 0 else np.array([], dtype=int)

        cross = all_idx[territories != territories[i]]
        cross_neighbors = []
        if len(cross) > 0 and rng.random() < params.cross_territory_tie_prob:
            cross_neighbors = rng.choice(cross, size=1, replace=False)
        arr = np.unique(np.concatenate([same_neighbors, np.array(cross_neighbors, dtype=int)]))
        neighbors.append(arr.astype(int))
    return neighbors


def governance_cost(hazard: np.ndarray, params: Params, g: str) -> np.ndarray:
    if g == "H":
        return params.hierarchy_fixed_cost + params.hierarchy_hazard_slope * hazard
    if g == "D":
        return params.delegated_fixed_cost + params.delegated_hazard_slope * hazard
    raise ValueError(f"Unknown governance {g}")


def screen_noise(hazard: np.ndarray, params: Params, g: str) -> np.ndarray:
    if g == "H":
        return params.hierarchy_screen_base + params.hierarchy_screen_hazard * hazard
    if g == "D":
        return params.delegated_screen_base + params.delegated_screen_hazard * hazard
    raise ValueError(f"Unknown governance {g}")


def monitor_level(
    hazard: np.ndarray,
    congestion: np.ndarray,
    terr_trust: np.ndarray,
    demand_pressure: np.ndarray,
    params: Params,
    g: str,
) -> np.ndarray:
    base = params.hierarchy_base_monitor if g == "H" else params.delegated_base_monitor
    trust_signal = 0.06 * (1 - terr_trust)
    demand_signal = 0.04 * np.maximum(0.0, demand_pressure - 0.25)
    adjustment = -params.monitor_congestion_penalty * congestion + 0.03 * (1 / (1 + hazard)) + trust_signal + demand_signal
    if g == "D":
        adjustment -= 0.01
    return clip(base + adjustment)


def governance_scores(
    current_governance: np.ndarray,
    opportunism: np.ndarray,
    hazard: np.ndarray,
    complexity: np.ndarray,
    recent_admin_excl: np.ndarray,
    recent_demand_gap: np.ndarray,
    recent_incl: np.ndarray,
    recent_cong: np.ndarray,
    recent_trust: np.ndarray,
    base_effort: np.ndarray,
    service_quality: np.ndarray,
    params: Params,
) -> Tuple[np.ndarray, np.ndarray]:
    dyn_hazard = hazard + 0.18 * recent_cong + 0.12 * recent_admin_excl + 0.08 * recent_demand_gap + 0.10 * (1 - recent_trust)
    t_h = governance_cost(dyn_hazard, params, "H")
    t_d = governance_cost(dyn_hazard, params, "D")
    screening_adequacy = 1.0 - clip(1.55 * recent_incl + 0.65 * np.maximum(0.0, recent_admin_excl - 0.25), 0.0, 0.92)

    score_h = (
        -params.score_cost_weight * t_h
        + params.score_h_inclusion_weight * recent_incl
        + params.score_h_opportunism_weight * opportunism
        + params.score_h_trustdef_weight * (1 - recent_trust)
        + params.score_h_dynhaz_weight * dyn_hazard
        + params.score_h_monitor_weight * np.maximum(0.0, recent_admin_excl - 0.20)
    )

    score_d = (
        -params.score_cost_weight * t_d
        + params.score_d_congestion_weight * np.maximum(0.0, recent_cong - 0.90)
        + params.score_d_demandgap_weight * recent_demand_gap
        + params.score_d_trust_weight * recent_trust
        + params.score_d_capacity_weight * base_effort * screening_adequacy
        + params.score_d_quality_weight * service_quality * screening_adequacy
        + 0.22 * complexity
        - params.score_d_opportunism_weight * opportunism
        - params.score_d_inclusion_weight * recent_incl
        - params.score_d_dynhaz_weight * np.maximum(0.0, dyn_hazard - 1.55)
        - params.score_d_adminexcl_weight * np.maximum(0.0, recent_admin_excl - 0.55)
        - params.score_d_screening_weight * (1.0 - screening_adequacy)
    )

    score_h = score_h + np.where(current_governance == "H", params.hysteresis_bonus, 0.0)
    score_d = score_d + np.where(current_governance == "D", params.hysteresis_bonus, 0.0)
    return score_h, score_d


def choose_initial_adaptive_governance(offices: Dict[str, np.ndarray], params: Params) -> np.ndarray:
    """Choose the initial adaptive governance mode explicitly.

    The default uses comparative institutional scores with no hysteresis so the
    adaptive regime does not begin from an implicit hierarchical benchmark.
    Set ``adaptive_initial_mode`` to ``"hierarchy"`` only if a hierarchical
    benchmark is desired as a deliberate modelling choice.
    """
    m = len(offices["hazard"])
    if params.adaptive_initial_mode == "hierarchy":
        return np.array(["H"] * m)

    recent_admin_excl = np.full(m, 0.20)
    recent_demand_gap = np.full(m, 0.26)
    recent_incl = np.full(m, 0.03)
    recent_cong = np.full(m, 0.82)
    recent_trust = np.full(m, 0.56)
    base_effort = offices["base_effort"].copy()
    service_quality = np.full(m, 0.58)
    neutral_current = np.array(["N"] * m)
    score_h, score_d = governance_scores(
        current_governance=neutral_current,
        opportunism=offices["opportunism"],
        hazard=offices["hazard"],
        complexity=offices["territorial_complexity"],
        recent_admin_excl=recent_admin_excl,
        recent_demand_gap=recent_demand_gap,
        recent_incl=recent_incl,
        recent_cong=recent_cong,
        recent_trust=recent_trust,
        base_effort=base_effort,
        service_quality=service_quality,
        params=params,
    )
    return np.where(score_d > score_h + params.init_switch_margin, "D", "H")

def update_adaptive_governance(
    office_governance: np.ndarray,
    last_switch: np.ndarray,
    t: int,
    opportunism: np.ndarray,
    hazard: np.ndarray,
    complexity: np.ndarray,
    recent_admin_excl: np.ndarray,
    recent_demand_gap: np.ndarray,
    recent_incl: np.ndarray,
    recent_cong: np.ndarray,
    recent_trust: np.ndarray,
    base_effort: np.ndarray,
    service_quality: np.ndarray,
    params: Params,
) -> Tuple[np.ndarray, np.ndarray, List[Tuple[int, str, str]]]:
    switches: List[Tuple[int, str, str]] = []
    if (t + 1) % params.review_interval != 0:
        return office_governance, last_switch, switches

    score_h, score_d = governance_scores(
        current_governance=office_governance,
        opportunism=opportunism,
        hazard=hazard,
        complexity=complexity,
        recent_admin_excl=recent_admin_excl,
        recent_demand_gap=recent_demand_gap,
        recent_incl=recent_incl,
        recent_cong=recent_cong,
        recent_trust=recent_trust,
        base_effort=base_effort,
        service_quality=service_quality,
        params=params,
    )

    for terr in range(len(office_governance)):
        if (t + 1 - last_switch[terr]) < params.min_tenure:
            continue
        old = office_governance[terr]
        if score_d[terr] > score_h[terr] + params.switch_cost_margin:
            new_g = "D"
        elif score_h[terr] > score_d[terr] + params.switch_cost_margin:
            new_g = "H"
        else:
            new_g = old

        if new_g != old:
            office_governance[terr] = new_g
            last_switch[terr] = t + 1
            switches.append((terr, old, new_g))

    return office_governance, last_switch, switches

def simulate_scenario(
    name: str,
    households: Dict[str, np.ndarray],
    offices: Dict[str, np.ndarray],
    neighbors,
    params: Params,
    rng: np.random.Generator,
):
    territory = households["territory"].copy()
    vulnerability = households["vulnerability"].copy()
    eligible = households["eligible"].copy()
    access_cost = households["access_cost"].copy()
    trust = households["trust"].copy()
    info = households["info"].copy()
    brokerage = households["brokerage"].copy()

    base_effort = offices["base_effort"].copy()
    opportunism = offices["opportunism"].copy()
    hazard = offices["hazard"].copy()
    complexity = offices["territorial_complexity"].copy()

    if name == "hierarchy":
        office_governance = np.array(["H"] * params.territories)
    elif name == "delegated":
        office_governance = np.array(["D"] * params.territories)
    elif name == "adaptive":
        office_governance = choose_initial_adaptive_governance(offices, params)
    else:
        raise ValueError(f"Unknown scenario {name}")

    initial_governance = office_governance.copy()
    last_switch = np.zeros(params.territories, dtype=int)
    switch_count = np.zeros(params.territories, dtype=int)
    delegated_periods = (office_governance == "D").astype(int)

    perceived_quality = np.full(params.territories, 0.50)
    service_quality_state = np.full(params.territories, 0.58)
    prior_apply = np.zeros(params.households)

    recent_admin_excl = np.full(params.territories, 0.22)
    recent_demand_gap = np.full(params.territories, 0.30)
    recent_incl = np.full(params.territories, 0.04)
    recent_cong = np.full(params.territories, 0.80)
    recent_trust = np.array([trust[territory == terr].mean() for terr in range(params.territories)])

    rows = []
    governance_history_rows = []

    for t in range(params.periods):
        apply_prob = np.zeros(params.households)

        for i in range(params.households):
            terr = territory[i]
            neigh = neighbors[i]
            neigh_success_signal = 0.0
            neigh_trust = 0.0
            if len(neigh) > 0:
                neigh_trust = trust[neigh].mean()
                neigh_success_signal = info[neigh].mean()

            expected_gain = (
                params.benefit
                * (0.35 + 0.40 * perceived_quality[terr])
                * (0.45 + 0.55 * trust[i])
                * (0.35 + 0.65 * info[i])
            )
            network_bonus = (
                params.network_info_strength * neigh_success_signal
                + params.network_trust_strength * neigh_trust
            )
            inertia = params.application_inertia * prior_apply[i]
            utility = expected_gain + network_bonus + inertia - params.access_cost_weight * access_cost[i]
            p_apply = clip(1 / (1 + np.exp(-4 * (utility - 0.35))))
            apply_prob[i] = p_apply

        applied = rng.random(params.households) < apply_prob
        applicants_by_office = [np.where((territory == terr) & applied)[0] for terr in range(params.territories)]

        approvals = np.zeros(params.households, dtype=int)

        office_access_rate = np.zeros(params.territories)
        office_quality_out = np.zeros(params.territories)
        office_monitor = np.zeros(params.territories)
        office_congestion = np.zeros(params.territories)
        office_access_gap = np.zeros(params.territories)
        office_admin_exclusion = np.zeros(params.territories)
        office_demand_gap = np.zeros(params.territories)
        office_inclusion = np.zeros(params.territories)
        office_terr_trust = np.zeros(params.territories)
        office_capture_realized = np.zeros(params.territories)

        for terr in range(params.territories):
            apps = applicants_by_office[terr]
            gov = office_governance[terr]
            terr_idx = np.where(territory == terr)[0]
            office_terr_trust[terr] = trust[terr_idx].mean()

            if len(apps) == 0:
                perceived_quality[terr] = 0.92 * perceived_quality[terr] + 0.08 * 0.50
                office_quality_out[terr] = service_quality_state[terr]
                total_eligible_terr = eligible[terr_idx].sum()
                eligible_applicants_terr = 0
                office_access_rate[terr] = 0.0 if total_eligible_terr > 0 else 1.0
                office_access_gap[terr] = 1 - office_access_rate[terr]
                office_demand_gap[terr] = (total_eligible_terr - eligible_applicants_terr) / max(1, total_eligible_terr)
                office_admin_exclusion[terr] = 0.0
                office_inclusion[terr] = 0.0
                continue

            approx_congestion = len(apps) / max(1, params.base_capacity)
            office_congestion[terr] = approx_congestion
            g_cost = governance_cost(np.array([hazard[terr]]), params, gov)[0]
            terr_trust_now = office_terr_trust[terr]
            demand_pressure = ((eligible[terr_idx] == 1) & (applied[terr_idx] == 0)).sum() / max(1, eligible[terr_idx].sum())
            monitor = monitor_level(
                np.array([hazard[terr]]),
                np.array([approx_congestion]),
                np.array([terr_trust_now]),
                np.array([demand_pressure]),
                params,
                gov,
            )[0]
            office_monitor[terr] = monitor

            effort_bonus = params.hierarchy_effort_bonus if gov == "H" else -params.delegated_effort_penalty
            quality_target = clip(
                params.quality_base_shift
                + params.quality_effort_weight * (base_effort[terr] + effort_bonus)
                + params.quality_monitor_weight * monitor
                - params.quality_cost_weight * g_cost
                - params.quality_opportunism_weight * opportunism[terr] * (1 - monitor)
                - params.quality_congestion_weight * max(0.0, approx_congestion - 1.0),
                0.10,
                0.95,
            )
            service_quality = clip(
                params.quality_persistence * service_quality_state[terr]
                + (1 - params.quality_persistence) * quality_target,
                0.10,
                0.95,
            )
            service_quality_state[terr] = service_quality
            office_quality_out[terr] = service_quality

            raw_capacity = max(
                1,
                int(
                    round(
                        params.base_capacity
                        * (1 - params.capacity_governance_penalty_scale * g_cost)
                        * (0.55 + 0.75 * service_quality)
                    )
                ),
            )

            sigma = screen_noise(np.array([hazard[terr]]), params, gov)[0]
            if gov == "H":
                capture_scale = params.hierarchy_capture_scale * (0.80 + 0.50 * opportunism[terr])
                patronage_bias = params.hierarchy_patronage_bias * (0.80 + 0.60 * opportunism[terr])
                capture_cap = params.hierarchy_capture_cap
                screening_drag_scale = params.hierarchy_screening_drag_scale
            else:
                capture_scale = params.delegated_capture_scale * (0.35 + 1.30 * opportunism[terr])
                patronage_bias = params.delegated_patronage_bias * (0.40 + 1.20 * opportunism[terr])
                capture_cap = params.delegated_capture_cap
                screening_drag_scale = params.delegated_screening_drag_scale
            leakage_pressure = capture_scale * opportunism[terr] * (1 - monitor) * (1 + 0.40 * max(0.0, approx_congestion - 1.0))
            capture_share = clip(0.02 + 0.95 * leakage_pressure, 0.0, capture_cap)
            screening_drag = clip(screening_drag_scale * (sigma + 0.75 * leakage_pressure), 0.0, 0.35)
            capacity = max(1, int(round(raw_capacity * (1 - screening_drag))))
            capture_slots = min(capacity, int(round(capacity * capture_share)))

            honest_scores = (
                1.00 * eligible[apps]
                + 0.82 * vulnerability[apps]
                + 0.20 * trust[apps]
                - patronage_bias * brokerage[apps] * (1 - eligible[apps])
                + rng.normal(0.0, sigma, size=len(apps))
            )
            honest_order = apps[np.argsort(honest_scores)[::-1]]

            captured = np.array([], dtype=int)
            if capture_slots > 0:
                capture_scores = (
                    1.10 * brokerage[apps]
                    + 0.20 * info[apps]
                    + patronage_bias * (1 - eligible[apps])
                    + 0.35 * patronage_bias * (1 - vulnerability[apps])
                    + rng.normal(0.0, 0.35 + sigma, size=len(apps))
                )
                captured = apps[np.argsort(capture_scores)[::-1][:capture_slots]]

            remaining_capacity = max(0, capacity - len(captured))
            honest_candidates = honest_order[~np.isin(honest_order, captured)]
            noneligible_cap_share = 0.08 if gov == "H" else 0.26
            noneligible_cap = int(round(capacity * noneligible_cap_share))
            chosen_list = list(captured)
            current_noneligible = int((eligible[captured] == 0).sum()) if len(captured) > 0 else 0
            for cand in honest_candidates:
                if len(chosen_list) >= capacity:
                    break
                if eligible[cand] == 0 and current_noneligible >= noneligible_cap:
                    continue
                chosen_list.append(int(cand))
                if eligible[cand] == 0:
                    current_noneligible += 1
            chosen = np.array(chosen_list, dtype=int)
            approvals[chosen] = 1

            observed_success = approvals[apps].mean() if len(apps) > 0 else 0.0
            perceived_quality[terr] = 0.65 * perceived_quality[terr] + 0.35 * observed_success

            total_eligible_terr = eligible[terr_idx].sum()
            eligible_applicants_terr = ((applied[terr_idx] == 1) & (eligible[terr_idx] == 1)).sum()
            eligible_served_terr = ((approvals[terr_idx] == 1) & (eligible[terr_idx] == 1)).sum()
            total_approved_terr = approvals[terr_idx].sum()
            ineligible_approved_terr = ((approvals[terr_idx] == 1) & (eligible[terr_idx] == 0)).sum()
            office_access_rate[terr] = eligible_served_terr / max(1, total_eligible_terr)
            office_access_gap[terr] = 1 - office_access_rate[terr]
            office_demand_gap[terr] = (total_eligible_terr - eligible_applicants_terr) / max(1, total_eligible_terr)
            office_admin_exclusion[terr] = (eligible_applicants_terr - eligible_served_terr) / max(1, eligible_applicants_terr)
            office_inclusion[terr] = ineligible_approved_terr / max(1, total_approved_terr)
            office_capture_realized[terr] = ineligible_approved_terr / max(1, capacity)

        # Household learning
        for i in range(params.households):
            neigh = neighbors[i]
            neigh_trust = trust[neigh].mean() if len(neigh) > 0 else trust[i]
            neigh_info = info[neigh].mean() if len(neigh) > 0 else info[i]

            if approvals[i] == 1 and eligible[i] == 1:
                trust[i] = clip(trust[i] + params.trust_gain_success)
                info[i] = clip(info[i] + params.info_gain_success)
            elif applied[i] and eligible[i] == 1 and approvals[i] == 0:
                trust[i] = clip(trust[i] - params.trust_loss_reject)
            elif eligible[i] == 1 and applied[i] == 0:
                trust[i] = clip(trust[i] - params.trust_loss_nonapply_eligible)

            trust[i] = clip((1 - params.network_trust_strength) * trust[i] + params.network_trust_strength * neigh_trust)
            info[i] = clip((1 - params.info_gain_neighbors) * info[i] + params.info_gain_neighbors * neigh_info)

        prior_apply = applied.astype(float)

        # Office learning / coevolution
        for terr in range(params.territories):
            governance_shift = 0.025 if office_governance[terr] == "H" else -0.010
            effort_target = clip(
                params.effort_target_base
                + params.effort_quality_weight * (office_quality_out[terr] - 0.50)
                + params.effort_monitor_weight * (office_monitor[terr] - 0.50)
                - params.effort_congestion_weight * max(0.0, office_congestion[terr] - 1.0)
                - params.effort_capture_weight * office_capture_realized[terr]
                + governance_shift,
                0.42,
                0.92,
            )
            base_effort[terr] = clip(
                base_effort[terr] + params.effort_learning_rate * (effort_target - base_effort[terr]),
                0.35,
                0.95,
            )

            opportunism_shift = 0.04 if office_governance[terr] == "D" else -0.02
            opportunism_target = clip(
                params.opportunism_target_base
                + params.opportunism_capture_weight * office_capture_realized[terr]
                + params.opportunism_congestion_weight * max(0.0, office_congestion[terr] - 1.0)
                + opportunism_shift
                - params.opportunism_monitor_weight * office_monitor[terr]
                - params.opportunism_quality_weight * office_quality_out[terr],
                0.05,
                0.72,
            )
            opportunism[terr] = clip(
                opportunism[terr] + params.opportunism_learning_rate * (opportunism_target - opportunism[terr]),
                0.03,
                0.75,
            )

        alpha = params.adaptive_metric_smoothing
        recent_admin_excl = (1 - alpha) * recent_admin_excl + alpha * office_admin_exclusion
        recent_demand_gap = (1 - alpha) * recent_demand_gap + alpha * office_demand_gap
        recent_incl = (1 - alpha) * recent_incl + alpha * office_inclusion
        recent_cong = (1 - alpha) * recent_cong + alpha * office_congestion
        recent_trust = (1 - alpha) * recent_trust + alpha * office_terr_trust

        if name == "adaptive":
            office_governance, last_switch, switches = update_adaptive_governance(
                office_governance=office_governance,
                last_switch=last_switch,
                t=t,
                opportunism=opportunism,
                hazard=hazard,
                complexity=complexity,
                recent_admin_excl=recent_admin_excl,
                recent_demand_gap=recent_demand_gap,
                recent_incl=recent_incl,
                recent_cong=recent_cong,
                recent_trust=recent_trust,
                base_effort=base_effort,
                service_quality=service_quality_state,
                params=params,
            )
            for terr, old, new_g in switches:
                switch_count[terr] += 1
                governance_history_rows.append({
                    "scenario": name,
                    "period": t + 1,
                    "office": terr,
                    "from_governance": old,
                    "to_governance": new_g,
                    "recent_admin_exclusion": float(recent_admin_excl[terr]),
                    "recent_demand_gap": float(recent_demand_gap[terr]),
                    "recent_inclusion": float(recent_incl[terr]),
                    "recent_congestion": float(recent_cong[terr]),
                    "recent_trust": float(recent_trust[terr]),
                })

        delegated_periods += (office_governance == "D").astype(int)

        total_eligible = eligible.sum()
        eligible_approved = ((approvals == 1) & (eligible == 1)).sum()
        ineligible_approved = ((approvals == 1) & (eligible == 0)).sum()
        total_approved = approvals.sum()
        access_rate = eligible_approved / max(1, total_eligible)
        exclusion_error = 1 - access_rate
        inclusion_error = ineligible_approved / max(1, total_approved)
        demand_gap = ((eligible == 1) & (applied == 0)).sum() / max(1, total_eligible)
        admin_exclusion_error = ((eligible == 1) & (applied == 1) & (approvals == 0)).sum() / max(1, ((eligible == 1) & (applied == 1)).sum())
        application_rate = applied.mean()
        avg_trust = trust.mean()
        avg_info = info.mean()
        territorial_access = []
        delegated_share_current = float((office_governance == "D").mean())
        for terr in range(params.territories):
            idx = np.where(territory == terr)[0]
            eligible_terr = eligible[idx].sum()
            ea_terr = ((approvals[idx] == 1) & (eligible[idx] == 1)).sum()
            territorial_access.append(ea_terr / max(1, eligible_terr))
        territorial_inequality = float(np.std(territorial_access))

        rows.append(
            {
                "scenario": name,
                "period": t,
                "applications": int(applied.sum()),
                "approvals": int(total_approved),
                "eligible_approved": int(eligible_approved),
                "ineligible_approved": int(ineligible_approved),
                "application_rate": float(application_rate),
                "access_rate": float(access_rate),
                "exclusion_error": float(exclusion_error),
                "admin_exclusion_error": float(admin_exclusion_error),
                "demand_gap": float(demand_gap),
                "inclusion_error": float(inclusion_error),
                "avg_trust": float(avg_trust),
                "avg_info": float(avg_info),
                "avg_monitoring": float(office_monitor.mean()),
                "avg_service_quality": float(office_quality_out.mean()),
                "avg_congestion": float(office_congestion.mean()),
                "territorial_inequality": territorial_inequality,
                "delegated_share": delegated_share_current,
            }
        )

    df = pd.DataFrame(rows)
    office_df = pd.DataFrame(
        {
            "office": np.arange(params.territories),
            "initial_governance": initial_governance,
            "final_governance": office_governance,
            "switch_count": switch_count,
            "delegated_share_over_time": delegated_periods / params.periods,
            "hazard": hazard,
            "territorial_complexity": complexity,
            "base_effort_final": base_effort,
            "opportunism_final": opportunism,
            "service_quality_final": office_quality_out,
            "admin_exclusion_final": office_admin_exclusion,
            "demand_gap_final": office_demand_gap,
            "access_gap_final": office_access_gap,
        }
    )
    governance_history = pd.DataFrame(governance_history_rows)
    return df, office_df, governance_history


def comparative_plot(all_ts: pd.DataFrame, out_path: Path):
    scenarios = list(all_ts["scenario"].unique())
    fig, axes = plt.subplots(2, 3, figsize=(14, 8))
    metrics = [
        ("access_rate", "Access rate"),
        ("exclusion_error", "Exclusion error"),
        ("inclusion_error", "Inclusion error"),
        ("avg_trust", "Average trust"),
        ("avg_congestion", "Average congestion"),
        ("delegated_share", "Delegated share"),
    ]

    for ax, (metric, title) in zip(axes.flat, metrics):
        for scen in scenarios:
            df = all_ts[all_ts["scenario"] == scen]
            ax.plot(df["period"], df[metric], label=scen)
        ax.set_title(title)
        ax.set_xlabel("Period")
        ax.set_ylabel(metric)
        ax.grid(True, alpha=0.3)

    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=len(scenarios), frameon=False)
    plt.tight_layout(rect=[0, 0.05, 1, 1])
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def build_summary(all_ts: pd.DataFrame) -> pd.DataFrame:
    mean_cols = [
        "access_rate",
        "exclusion_error",
        "admin_exclusion_error",
        "demand_gap",
        "inclusion_error",
        "avg_trust",
        "avg_info",
        "avg_monitoring",
        "avg_service_quality",
        "avg_congestion",
        "territorial_inequality",
        "delegated_share",
    ]
    final = all_ts.sort_values("period").groupby("scenario").tail(1).copy()
    means = (
        all_ts.groupby("scenario")[mean_cols]
        .mean()
        .reset_index()
        .rename(columns={c: f"{c}_mean" for c in mean_cols})
    )

    final = final[
        [
            "scenario",
            "applications",
            "approvals",
            "eligible_approved",
            "ineligible_approved",
            "access_rate",
            "exclusion_error",
            "inclusion_error",
            "avg_trust",
            "avg_info",
            "avg_monitoring",
            "avg_service_quality",
            "avg_congestion",
            "territorial_inequality",
            "delegated_share",
        ]
    ].rename(
        columns={
            "applications": "applications_final",
            "approvals": "approvals_final",
            "eligible_approved": "eligible_approved_final",
            "ineligible_approved": "ineligible_approved_final",
            "access_rate": "access_rate_final",
            "exclusion_error": "exclusion_error_final",
            "inclusion_error": "inclusion_error_final",
            "avg_trust": "avg_trust_final",
            "avg_info": "avg_info_final",
            "avg_monitoring": "avg_monitoring_final",
            "avg_service_quality": "avg_service_quality_final",
            "avg_congestion": "avg_congestion_final",
            "territorial_inequality": "territorial_inequality_final",
            "delegated_share": "delegated_share_final",
        }
    )

    return final.merge(means, on="scenario", how="left")


def parse_args():
    p = argparse.ArgumentParser(description="Run an agent-based model of governance arrangements in social program access.")
    p.add_argument("--out-dir", type=str, default="outputs_governance_abm", help="Output directory.")
    p.add_argument("--seed", type=int, default=42, help="Random seed.")
    p.add_argument("--periods", type=int, default=40, help="Number of periods.")
    p.add_argument("--households", type=int, default=600, help="Number of households.")
    p.add_argument("--territories", type=int, default=6, help="Number of territories / local offices.")
    p.add_argument("--neighbors", type=int, default=6, help="Average within-territory neighbor count.")
    p.add_argument(
        "--scenario",
        type=str,
        default="compare",
        choices=["hierarchy", "delegated", "adaptive", "compare"],
        help="Scenario to simulate.",
    )
    return p.parse_args()


def main():
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    params = Params(
        periods=args.periods,
        households=args.households,
        territories=args.territories,
        neighbors=args.neighbors,
        seed=args.seed,
    )
    rng = np.random.default_rng(params.seed)

    households = build_households(params, rng)
    offices = build_offices(params, rng)
    neighbors = build_neighbors(households["territory"], params, rng)

    scenario_list = [args.scenario] if args.scenario != "compare" else ["hierarchy", "delegated", "adaptive"]
    all_ts = []
    office_files = []
    governance_histories = []

    for offset, scen in enumerate(scenario_list):
        scen_rng = np.random.default_rng(params.seed + 1000 + offset)
        ts, office_df, gov_hist = simulate_scenario(scen, households, offices, neighbors, params, scen_rng)
        all_ts.append(ts)
        office_path = out_dir / f"offices_{scen}.csv"
        office_df.to_csv(office_path, index=False)
        office_files.append(office_path)
        if not gov_hist.empty:
            governance_histories.append(gov_hist)

    all_ts = pd.concat(all_ts, ignore_index=True)
    ts_path = out_dir / "timeseries.csv"
    all_ts.to_csv(ts_path, index=False)

    summary = build_summary(all_ts)
    summary_path = out_dir / "summary.csv"
    summary.to_csv(summary_path, index=False)

    plot_path = out_dir / "comparative_trajectories.png"
    comparative_plot(all_ts, plot_path)

    if governance_histories:
        gov_hist_all = pd.concat(governance_histories, ignore_index=True)
        gov_hist_path = out_dir / "governance_history.csv"
        gov_hist_all.to_csv(gov_hist_path, index=False)
        print(f"Saved governance history to: {gov_hist_path}")

    print(f"Saved time series to: {ts_path}")
    print(f"Saved summary to: {summary_path}")
    print(f"Saved plot to: {plot_path}")
    for fp in office_files:
        print(f"Saved office data to: {fp}")
    print("\nSummary:")
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()