# Manufacturing Warning Time: Predictive Early-Warning Model for Drug Shortages

**Author:** Oluchukwu Obinna Ogbuagu, B.Pharm., MBA  

## Overview
This repository provides the open reference implementation for a two-layer predictive early-warning framework designed to forecast drug shortages in U.S. hospital pharmacies, health systems, and regional pharmaceutical distributors **4 to 12 weeks prior to official public shortage announcements**.

- **Layer 1 (Market Supply Risk):** Estimates product-level shortage probabilities using public federal data streams (FDA Drug Shortages Database, ASHP archives, FDA Orange Book, openFDA enforcement reports, FDA Data Dashboard inspection classifications, CARES Act § 3112 notifications, and CMS NADAC pricing).
- **Layer 2 (Organizational Exposure):** Integrates Layer 1 probabilities with local hospital utilization velocity, inventory days on hand (DOH), clinical criticality, and therapeutic substitution friction to produce a prioritized **Weekly Watch List** with explainable risk drivers.

## Repository Contents
- `shortage_early_warning.py` — Python implementation of the Layer 1 feature engineering, calibrated risk scoring, and Layer 2 Operational Risk Priority Score (`ORPS`) ranking engine.
- `sample_supply_signals.csv` — Sample dataset combining public regulatory/market indicators with representative hospital pharmacy inventory parameters.
- `LICENSE` — MIT Open-Source License enabling free adoption by health systems, researchers, and public health agencies.

## Quickstart
1. Ensure Python 3.9+ with `pandas`, `numpy`, and `scikit-learn` installed:
   ```bash
   pip install pandas numpy scikit-learn
   ```
2. Run the predictive watch-list generator:
   ```bash
   python shortage_early_warning.py
   ```
3. Review the ranked console output and generated `weekly_watch_list_output.csv`.
