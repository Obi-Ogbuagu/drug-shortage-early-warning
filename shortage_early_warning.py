"""
Manufacturing Warning Time: Two-Layer Predictive Early-Warning Model
for Drug Shortages in U.S. Hospital Pharmacies and Distributors

Author: Oluchukwu Obinna Ogbuagu, B.Pharm., MBA
Date: 2026
License: MIT
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.calibration import CalibratedClassifierCV


def engineer_layer1_features(df: pd.DataFrame) -> pd.DataFrame:
    """Extracts structural market fragility and regulatory risk features."""
    X = pd.DataFrame(index=df.index)
    X["n_active_labelers"] = df["n_active_labelers"]
    X["is_sole_or_dual_source"] = (df["n_active_labelers"] <= 2).astype(int)
    X["is_sterile_injectable"] = df["is_sterile_injectable"]
    X["is_controlled_subst"] = df["is_controlled_subst"]
    X["oai_count_12m"] = df["oai_count_12m"]
    X["form483_velocity_6m"] = df["form483_velocity_6m"]
    X["cares_3112_flag"] = df["cares_3112_flag"]
    X["nadac_pct_chg_90d"] = df["nadac_pct_chg_90d"]
    X["prior_shortage_24m"] = df["prior_shortage_24m"]
    return X


def compute_layer1_market_risk(df: pd.DataFrame) -> pd.Series:
    """
    Computes calibrated 8-week market shortage probability P(S_{i, t+8})
    using gradient-boosted trees with interpretable feature weights.
    """
    X = engineer_layer1_features(df)
    y = df["entered_shortage_8w"]

    base_clf = GradientBoostingClassifier(
        max_depth=3,
        n_estimators=50,
        learning_rate=0.08,
        random_state=42
    )
    # Fit on historical training window and compute calibrated risk scores
    base_clf.fit(X, y)
    raw_probs = base_clf.predict_proba(X)[:, 1]

    # Blend with deterministic domain risk index for small-sample stability
    domain_score = (
        0.22 * X["is_sole_or_dual_source"]
        + 0.20 * X["is_sterile_injectable"]
        + 0.20 * (X["oai_count_12m"] > 0).astype(int)
        + 0.15 * X["cares_3112_flag"]
        + 0.13 * (X["nadac_pct_chg_90d"] > 10.0).astype(int)
        + 0.10 * (X["prior_shortage_24m"] >= 2).astype(int)
    )
    calibrated_prob = np.clip(0.5 * raw_probs + 0.5 * domain_score, 0.01, 0.99)
    return pd.Series(calibrated_prob, index=df.index, name="prob_shortage_8w")


def compute_layer2_exposure(df: pd.DataFrame, alpha: float = 0.75) -> pd.DataFrame:
    """
    Combines Layer 1 market probabilities with facility-specific utilization,
    days on hand (DOH), clinical criticality, and substitution difficulty
    to generate the Weekly Ranked Watch List.
    """
    result = df[["ndc", "drug_name", "avg_weekly_units", "days_on_hand"]].copy()
    result["prob_shortage_8w"] = compute_layer1_market_risk(df).round(3)

    burn_ratio = (df["avg_weekly_units"] / np.maximum(df["days_on_hand"], 1)) ** alpha
    clinical_weight = df["clinical_crit_score"] * df["sub_difficulty_score"]

    raw_orps = result["prob_shortage_8w"] * burn_ratio * clinical_weight
    result["orps_score"] = (100 * raw_orps / raw_orps.max()).round(1)

    def assign_tier(row):
        if row["prob_shortage_8w"] >= 0.65 and row["orps_score"] >= 35.0:
            return "Tier 1 - Immediate Action"
        elif row["prob_shortage_8w"] >= 0.40 or row["orps_score"] >= 20.0:
            return "Tier 2 - Active Monitoring"
        return "Tier 3 - Routine Surveillance"

    def explain_drivers(row):
        drivers = []
        if row["oai_count_12m"] > 0:
            drivers.append("Recent FDA OAI Inspection")
        if row["n_active_labelers"] <= 2:
            drivers.append("Sole/Dual Supplier Concentration")
        if row["cares_3112_flag"] == 1:
            drivers.append("CARES Act 3112 Notice")
        if row["nadac_pct_chg_90d"] > 10.0:
            drivers.append(f"NADAC Price +{row['nadac_pct_chg_90d']}%")
        if row["days_on_hand"] <= 14:
            drivers.append(f"Low Buffer ({row['days_on_hand']} DOH)")
        return "; ".join(drivers[:3]) if drivers else "Baseline Market Risk"

    result["action_tier"] = result.apply(assign_tier, axis=1)
    result["top_risk_drivers"] = df.apply(explain_drivers, axis=1)

    return result.sort_values(by="orps_score", ascending=False).reset_index(drop=True)


if __name__ == "__main__":
    data = pd.read_csv("sample_supply_signals.csv")
    watch_list = compute_layer2_exposure(data)
    print("=== WEEKLY PREDICTIVE DRUG SHORTAGE WATCH LIST ===")
    print(watch_list[["ndc", "drug_name", "prob_shortage_8w", "orps_score", "action_tier", "top_risk_drivers"]].to_string(index=False))
    watch_list.to_csv("weekly_watch_list_output.csv", index=False)
