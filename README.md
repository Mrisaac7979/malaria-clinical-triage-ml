# Malaria Clinical Triage - Binary Classification Pipeline

**Lightweight ML for Disease Prediction in Rural Health Settings (Togo)**

---

## Overview

This project implements the malaria probability triage module described in Phase 2 of the
research proposal. It demonstrates
that clinical and demographic features alone without laboratory data can support accurate
malaria triage using ensemble ML methods, consistent with the 2025 systematic review
published on medRxiv (DOI: 10.1101/2025.08.03.25332923).

**Author:** Komi Isaac Junior HOUNBO

---

## Clinical Context

In rural Togo, community health workers face presentations of fever, fatigue, and respiratory
symptoms that are ambiguous across malaria, typhoid, dengue, and other endemic conditions,
with no computational tool to support triage. This pipeline prototypes a solution deployable
on low-end Android devices (Tecno Spark, 2GB RAM) without internet connectivity.

**Key clinical constraint:** sensitivity >= 0.92 (minimum threshold for a screening tool
in low-resource settings, as specified in the research proposal Phase 2).

---

## Results

| Model | AUC-ROC (5-Fold CV) | Sensitivity (test) | AUC-ROC (test) |
|-------|--------------------:|-------------------:|---------------:|
| Logistic Regression | 0.9423 ± 0.016 | - | - |
| Random Forest | 0.9385 ± 0.017 | - | - |
| Gradient Boosting | 0.9300 ± 0.021 | - | - |
| **Ensemble (Soft Vote)** | **0.9402 ± 0.018** | **92.4%** | **0.9232** |

Decision threshold adjusted from 0.50 to 0.40 to meet the clinical sensitivity requirement.

---

## Dataset

Synthetic dataset (n=2,000) calibrated on published clinical feature distributions for
malaria and non-malaria fever presentations in sub-Saharan Africa.

**Features (13 clinical and demographic variables):**

| Feature | Type | Description |
|---------|------|-------------|
| `age_groupe` | Categorical | Age group: <5ans, 5-14ans, 15-49ans, 50+ans |
| `sexe` | Binary | Sex: 0=Female, 1=Male |
| `temperature_C` | Continuous | Body temperature (°C) |
| `duree_fievre_j` | Ordinal | Fever duration (days) |
| `cephalees` | Binary | Headache (0/1) |
| `vomissements` | Binary | Vomiting (0/1) |
| `frissons` | Binary | Chills (0/1) |
| `fatigue` | Binary | Fatigue (0/1) |
| `douleurs_articulaires` | Binary | Joint pain (0/1) |
| `saison_pluvieuse` | Binary | Rainy season (0/1) - malaria risk proxy |
| `frequence_resp` | Continuous | Respiratory rate (breaths/min) |
| `spo2_pct` | Continuous | Oxygen saturation (%) |
| `antecedent_palu` | Binary | Previous malaria episode (0/1) |

5% label noise added to simulate real-world annotation uncertainty.

---

## Project Structure

```
malaria_project/
│
├── data/
│   └── malaria_clinical_proxy.csv     # Generated dataset
│
├── figures/
│   ├── 01_eda_clinical.png            # EDA: feature distributions by class
│   └── 02_results_malaria.png         # Results dashboard
│
├── Malaria_Clinical_Triage_ML.ipynb   # Main notebook (run on Google Colab)
├── generate_data.py                   # Dataset generation script
├── train_evaluate.py                  # Training & evaluation script
└── README.md
```

---

## How to Run

### Option A - Google Colab (recommended)
1. Upload `Malaria_Clinical_Triage_ML.ipynb` to [colab.research.google.com](https://colab.research.google.com)
2. `Runtime → Run all`
3. No local installation required

### Option B - Local
```bash
pip install scikit-learn pandas numpy matplotlib seaborn imbalanced-learn
python generate_data.py
python train_evaluate.py
```

---

## Key Findings

- **Temperature** and **fever duration** are the most discriminative features, consistent
  with WHO clinical malaria diagnostic criteria.
- **Rainy season** and **malaria history** contribute substantially to model performance,
  reflecting the epidemiological context of Togo's Plateaux and Maritime regions.
- The **Ensemble (RF + Gradient Boosting + Logistic Regression)** approach aligns with
  the research proposal methodology and achieves the clinical sensitivity threshold.
- Threshold optimization (0.50 → 0.40) is a standard practice in clinical screening tools
  where false negatives are more costly than false positives.

---

## Research Perspective

This prototype will be extended in the Master's research to:
- Real anonymized clinical records from Togolese rural health facilities (2019-2024),
  collected via memorandum of understanding with the Togolese Ministry of Health
- SMOTE oversampling for class imbalance in the fever severity classification task
- Model compression: TensorFlow Lite INT8 quantization (target: <30MB, <300ms inference)
- Deployment: Flutter mobile application, French/Ewe interface, offline-first architecture
- Field validation: quasi-experimental design, n=120, RDT-confirmed ground truth

---

## References

- medRxiv (2025). Ensemble ML for Malaria Diagnosis in Resource-Limited Settings.
  DOI: 10.1101/2025.08.03.25332923
- BMC Infectious Diseases (2025). Malaria epidemic periods in Togo.
  DOI: 10.1186/s12879-025-10956-w
- WHO World Malaria Report 2024. Geneva: WHO.
- Scientific Reports (2024). Malaria risk mapping among children under five in Togo.
  DOI: 10.1038/s41598-024-58287-1
- Breiman, L. (2001). Random Forests. Machine Learning, 45(1), 5-32.
