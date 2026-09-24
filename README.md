# EV Battery Health Predictor using Machine Learning

**Module 1 Project** — a machine-learning classifier that predicts electric vehicle (EV) battery health status — **Healthy / Degrading / Critical** — from real Li-ion battery cycle-test data, without requiring a full capacity discharge test.

## Objective

Train one classification model to predict battery health using measurements collected during routine charge–discharge–impedance test cycles, and evaluate it with standard classification metrics (accuracy, confusion matrix).

## Dataset

`Battery_Data_Cleaned.csv` — a cleaned export of the public **NASA Li-ion Battery Dataset** (NASA Prognostics Center of Excellence), covering **34 battery cells** (18650 Li-ion) cycled under different ambient temperatures (4°C, 24°C, and higher-stress conditions up to 44°C) until capacity fade. 7,368 raw test records.

| Column | Description |
|---|---|
| `type` | Test type: `1` = charge, `0` = discharge, `-1` = electrochemical impedance spectroscopy (EIS) |
| `ambient_temperature` | Chamber temperature during the test (°C) |
| `battery_id`, `test_id`, `uid` | Cell and test-sequence identifiers |
| `Capacity` | Normalized discharge capacity (fraction of the cell's rated capacity) |
| `Re` | Electrolyte resistance (Ohms), from the EIS test |
| `Rct` | Charge-transfer resistance (Ohms), from the EIS test |

## Approach

1. **Cleaning** — dropped exact duplicate rows and 50 records with `Capacity == 0` (invalid readings, not real dead cells). Result: 7,318 valid records.
2. **Modeling table** — kept only the impedance-test rows (`type == -1`), which carry genuine, directly-measured `Re`/`Rct` values (2,705 records after removing exact feature duplicates).
3. **Labeling** — each battery's own maximum observed `Capacity` is treated as its rated (fresh-cell) capacity. State of Health is computed as `SOH % = Capacity / rated_capacity * 100`, then bucketed:
   - **Healthy:** SOH ≥ 85%
   - **Degrading:** 65% ≤ SOH < 85%
   - **Critical:** SOH < 65%
4. **Features** — `Re`, `Rct`, `ambient_temperature`, `test_id` (cycle-order proxy). `Capacity`/`SOH_percent` are excluded from the model inputs since they define the label — including them would leak the answer.
5. **Model** — Random Forest Classifier (300 trees, max depth 10, class-balanced weighting), 75/25 stratified train/test split.

## Results

| Metric | Score |
|---|---|
| Accuracy | **90.8%** |
| Macro F1 | 0.889 |

**Confusion matrix** (rows = actual, columns = predicted):

| Actual \ Predicted | Critical | Degrading | Healthy |
|---|---|---|---|
| Critical | 61 | 5 | 2 |
| Degrading | 4 | 228 | 9 |
| Healthy | 13 | 29 | 326 |

`Rct` and cycle progression (`test_id`) were the strongest predictors, followed by `Re`; ambient temperature contributed least — consistent with internal-resistance growth being the dominant electrical fingerprint of battery aging.

## Repository Contents

```
├── Battery_Data_Cleaned.csv          # Raw input dataset
├── cleaned_battery_data.csv          # Cleaned dataset (post dedup/filtering)
├── modeling_table.csv                # Final feature + label table used for training
├── EV_Battery_Health_Predictor.ipynb # Full Colab/Jupyter notebook (cleaning → EDA → training → evaluation)
├── EV_Battery_Health_Report.docx     # Written project report
├── chart1_capacity_vs_cycle.png      # Capacity vs. cycle progression by health status
├── chart2_health_distribution.png    # Class distribution
├── chart3_confusion_matrix.png       # Confusion matrix heatmap
├── chart4_resistance_vs_soh.png      # Re/Rct vs. State of Health
├── chart5_feature_importance.png     # Random Forest feature importance
└── README.md
```

## How to Run

**Google Colab:**
1. Open `EV_Battery_Health_Predictor.ipynb` in [Colab](https://colab.research.google.com).
2. Upload `Battery_Data_Cleaned.csv` via the Files panel (folder icon, left sidebar).
3. `Runtime → Run all`.

**Locally:**
```bash
pip install pandas numpy matplotlib scikit-learn
jupyter notebook EV_Battery_Health_Predictor.ipynb
```

## Tools & Libraries

Python · Pandas · NumPy · Scikit-learn · Matplotlib · Google Colab

## Limitations & Next Steps

- Rated capacity was estimated per battery as its own observed maximum, since no factory nameplate capacity was provided in the file.
- A few ambient-temperature conditions (22°C, 43°C, 44°C) have limited sample counts.
- Future work: try gradient-boosted trees (XGBoost/LightGBM), add rolling-window resistance-trend features, or reframe as SOH% regression.
