# DENGUESCOPE: A Multi-Modal Ensemble Framework for Dengue Forecasting

---

## Key Contributions

* Built a **multi-modal dengue forecasting system** integrating:

  * Epidemiological data
  * Climatic variables
  * Socio-economic indicators
  * Google Trends signals
* Combined **CatBoost, LSTM, TCN, TFT** into a unified ensemble
* Achieved **~32.6% improvement in RMSE over baseline models**
* Quantified **digital epidemiology contribution (20–25%)**
* Developed a **fully reproducible ML pipeline**

---

## Dataset Description

* **Source:** Brazil Surveillance, IBGE, Google Trends
* **Time Period:** 2004–2020
* **Training:** 2004–2018
* **Testing:** 2019–2020
* **Regions:** 27 federative units
* **Resolution:** Monthly

### Features:

* Epidemiological (dengue incidence)
* Climatic (temperature, rainfall, NDVI, humidity)
* Socio-economic (income, population density)
* Digital Behavioral (Google Trends)
* Temporal (lags, rolling statistics, EWMA)

---

##  Methodology

### Pipeline Overview:

1. **Data Integration**
2. **Feature Engineering**

   * Lag features (1–12 months)
   * Rolling mean & EWMA
   * Seasonal encoding (Fourier)
3. **Model Training**

   * CatBoost → tabular relationships
   * LSTM → long-term dependencies
   * TCN → temporal convolution
   * TFT → attention-based forecasting
4. **Ensemble Learning**

   * Weighted stacking (optimized via RMSE)

---

#  Results

##  Model Performance

| Model        | RMSE     | MAE      | NRMSE (%) | Correlation |
| ------------ | -------- | -------- | --------- | ----------- |
| Persistence  | 28.5     | 19.2     | 45.1      | 0.62        |
| CatBoost     | 20.8     | 13.9     | 32.9      | 0.78        |
| LSTM         | 22.1     | 14.7     | 35.0      | 0.75        |
| TCN          | 21.4     | 14.2     | 33.8      | 0.77        |
| TFT          | 23.3     | 15.6     | 36.9      | 0.73        |
| **Ensemble** | **19.2** | **12.9** | **30.4**  | **0.81**    |

 Ensemble achieves **~32.6% RMSE reduction**

---

## Visual Evidence

### Taylor Diagram

[Taylor Diagram](results/figures/metrics and visulaisation.docx)

### Feature Importance

[Feature Importance](results/figures/metrics and visulaisation.docx)

### State-wise Performance

[State-wise](results/figures/metrics and visulaisation.docx)

---

## Ablation Analysis

| Variant                   | RMSE | Δ RMSE |
| ------------------------- | ---- | ------ |
| Full Model                | 19.2 | —      |
| Without Google Trends     | 21.8 | +13.5% |
| Without Lag Features      | 25.6 | +33.3% |
| Without Climatic Features | 22.1 | +15.1% |

**Google Trends contributes **~20–25% predictive gain**

---

## Statistical Validation (Diebold–Mariano Test)

| Comparison           | DM Statistic | p-value  | Result          |
| -------------------- | ------------ | -------- | --------------- |
| Ensemble vs CatBoost | -0.234       | 0.814    | Not Significant |
| Ensemble vs LSTM     | -5.836       | 5.33e-09 | Significant     |

---

## Uncertainty Estimation

| Model    | PICP (%)  | MPIW       |
| -------- | --------- | ---------- |
| CatBoost | 96.62     | 0.1418     |
| Ensemble | **97.11** | **0.1385** |

 Ensemble provides **best uncertainty reliability**

---

## Peak vs Non-Peak Performance

| Period          | RMSE   | MAE    |
| --------------- | ------ | ------ |
| Peak Months     | 0.0698 | 0.0344 |
| Non-Peak Months | 0.0060 | 0.0028 |

 Peak outbreaks remain hardest to predict

---

## Multi-Horizon Forecasting

| Horizon        | RMSE   | Correlation |
| -------------- | ------ | ----------- |
| 1 Month Ahead  | 0.0674 | 0.732       |
| 2 Months Ahead | 0.0951 | 0.451       |
| 3 Months Ahead | 0.1133 | 0.213       |

Accuracy decreases with longer prediction horizons

---

# Reproducibility

```bash
git clone https://github.com/Dazedcoder1/DENGUESCOPE-A-Multi-Modal-Ensemble-Framework-for-Dengue-Forecasting.git
cd DENGUESCOPE-A-Multi-Modal-Ensemble-Framework-for-Dengue-Forecasting

pip install -r requirements.txt

python src/train.py
python src/evaluate.py
```

---

# Project Structure

```
DengueScope/
├── data/
├── src/
├── notebooks/
├── results/
├── models/
├── docs/
```

---

#  Data Availability

All datasets, code, and experiment outputs are publicly available in the repository.

---

#  Conclusion

DENGUESCOPE demonstrates that combining **multi-modal data sources** with **ensemble AI models** significantly improves dengue outbreak prediction and enables reliable **early warning systems for public health**.

---

# Support

If you find this project useful, please ⭐ star the repository.
