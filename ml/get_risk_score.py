"""
get_risk_score() — the single function Layer 3 (backend) calls.
Per mplads_technical_spec.md section 2.1: takes RAW project fields,
does its own feature engineering, returns a risk score dict.

Requires these files in the same directory:
  isolation_forest.pkl, state_benchmark.pkl, group_stats.pkl,
  utilization_benchmark.pkl

Usage:
    from get_risk_score import get_risk_score
    result = get_risk_score({
        "project_id": "NEW-001", "state": "Bihar",
        "work_category": "Railways, Roads, Pathways & Bridges",
        "mp_name": "...", "sanctioned_amount": 2500000,
        "actual_expenditure": 2400000, "start_date": "2023-06-01",
        "expected_completion": "2024-01-01", "actual_completion": None,
        "status": "In Progress", "has_tender_on_file": True,
        "has_mp_recommendation": True
    })
"""

import pandas as pd
import numpy as np
import joblib
import os

_DIR = os.path.dirname(os.path.abspath(__file__))
_model = joblib.load(os.path.join(_DIR, 'isolation_forest.pkl'))
_bench = joblib.load(os.path.join(_DIR, 'state_benchmark.pkl'))
_group_stats = joblib.load(os.path.join(_DIR, 'group_stats.pkl'))
_util = joblib.load(os.path.join(_DIR, 'utilization_benchmark.pkl'))

_STATUS_CATEGORIES = ['Completed', 'In Progress', 'Not Started', 'Stalled']


def _engineer_features(p: dict) -> dict:
    """Recompute the same engineered features used in training, from raw input."""
    state = p['state']
    cost_estimate = _bench['state_benchmark'].get(state, _bench['national_avg'])
    if pd.isna(cost_estimate):
        cost_estimate = _bench['national_avg']

    cost_deviation_ratio = p['sanctioned_amount'] / cost_estimate
    expenditure_utilization = p['actual_expenditure'] / p['sanctioned_amount'] if p['sanctioned_amount'] else 0

    start = pd.to_datetime(p['start_date'])
    expected = pd.to_datetime(p['expected_completion'])
    today = pd.Timestamp.now()
    if p.get('actual_completion'):
        actual = pd.to_datetime(p['actual_completion'])
        delay_days = (actual - expected).days
    elif p['status'] in ('In Progress', 'Stalled', 'Not Started'):
        delay_days = max(0, (today - expected).days)
    else:
        delay_days = 0

    key = (state, p['work_category'])
    if key in _group_stats.index:
        mean, std = _group_stats.loc[key, 'mean'], _group_stats.loc[key, 'std']
    else:
        mean, std = cost_estimate, cost_estimate * 0.3
    std = std if std and not pd.isna(std) and std > 0 else cost_estimate * 0.3
    district_cost_zscore = (p['sanctioned_amount'] - mean) / std

    return {
        'cost_deviation_ratio': round(cost_deviation_ratio, 2),
        'expenditure_utilization': round(min(max(expenditure_utilization, 0), 1), 2),
        'delay_days': int(delay_days),
        'district_cost_zscore': round(district_cost_zscore, 2),
        'planned_days': (expected - start).days,
    }


def _apply_rules(p: dict, feats: dict) -> list:
    flags = []
    if not p['has_mp_recommendation']:
        flags.append('no_mp_recommendation')
    if not p['has_tender_on_file'] and feats['cost_deviation_ratio'] > 1.1:
        flags.append('no_tender_high_value')
    if feats['cost_deviation_ratio'] > 1.5:
        flags.append('cost_overrun')
    if feats['expenditure_utilization'] > 0.9 and p['status'] == 'Not Started':
        flags.append('ghost_project')
    planned_safe = max(abs(feats['planned_days']), 30)
    if p['status'] in ('In Progress', 'Stalled', 'Not Started') and \
       feats['delay_days'] > 2.0 * planned_safe and feats['delay_days'] > 200:
        flags.append('delayed_stalled')
    return flags


def get_risk_score(project: dict) -> dict:
    feats = _engineer_features(project)
    rule_flags = _apply_rules(project, feats)

    status_row = {f'status_{s}': (project['status'] == s) for s in _STATUS_CATEGORIES}
    ml_input = pd.DataFrame([{
        'cost_deviation_ratio': feats['cost_deviation_ratio'],
        'expenditure_utilization': feats['expenditure_utilization'],
        'delay_days': feats['delay_days'],
        'district_cost_zscore': feats['district_cost_zscore'],
        **status_row,
    }])
    # align columns with training order; fill any missing status dummy with False
    for col in ['status_Completed', 'status_In Progress', 'status_Not Started', 'status_Stalled']:
        if col not in ml_input.columns:
            ml_input[col] = False
    ml_input = ml_input[['cost_deviation_ratio', 'expenditure_utilization', 'delay_days',
                          'district_cost_zscore', 'status_Completed', 'status_In Progress',
                          'status_Not Started', 'status_Stalled']]

    ml_flagged = _model.predict(ml_input)[0] == -1
    ml_score = -_model.score_samples(ml_input)[0]  # higher = more anomalous

    all_flags = list(rule_flags)
    if ml_flagged and not all_flags:
        all_flags.append('statistical_anomaly')

    # Combine into 0-100 score: rule hits are strong signals, ML score adds nuance
    base = min(len(rule_flags) * 35, 90)
    ml_component = min(max(ml_score, 0) * 40, 40)
    risk_score = int(min(base + ml_component if rule_flags else ml_component, 100))
    if not rule_flags and not ml_flagged:
        risk_score = int(min(risk_score, 30))

    severity = 'high' if risk_score >= 70 else 'medium' if risk_score >= 40 else 'low' if risk_score >= 15 else 'none'

    reason = f"Flags: {', '.join(all_flags)}" if all_flags else "No significant risk indicators detected"

    return {
        'project_id': project.get('project_id', 'UNKNOWN'),
        'risk_score': risk_score,
        'severity': severity,
        'flags': all_flags,
        'reason': reason,
    }
