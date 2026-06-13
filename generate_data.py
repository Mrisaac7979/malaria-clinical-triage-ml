"""
Génération d'un dataset clinique synthétique réaliste pour le triage du paludisme.

Features calibrées sur la littérature clinique (Togo, Afrique subsaharienne) :
- Koram & Molyneux (2007) : critères OMS de paludisme clinique
- Medrxiv (2025) : ensemble ML for malaria diagnosis, features cliniques/démographiques
- PMC11895289 (2025) : accès aux soins, fièvre, Togo

Classes :
  1 = Paludisme probable (triage positif → référer pour RDT/traitement)
  0 = Non-paludisme (autres causes : fièvre typhoïde, IRA, dengue...)
"""

import numpy as np
import pandas as pd

np.random.seed(42)
N = 2000

# ── Classe 1 : Paludisme probable ────────────────────────────────────────────
n_pos = N // 2

age          = np.random.choice(['<5ans', '5-14ans', '15-49ans', '50+ans'],
                                 n_pos, p=[0.37, 0.20, 0.33, 0.10])
sexe         = np.random.choice([0, 1], n_pos, p=[0.50, 0.50])          # 0=F, 1=M
temperature  = np.random.normal(38.9, 0.55, n_pos).clip(37.5, 41.5)
duree_fievre = np.random.choice([1, 2, 3, 4, 5, 6, 7],
                                 n_pos, p=[0.10, 0.20, 0.25, 0.20, 0.12, 0.08, 0.05])
cephalees    = np.random.binomial(1, 0.85, n_pos)
vomissements = np.random.binomial(1, 0.65, n_pos)
frissons     = np.random.binomial(1, 0.80, n_pos)
fatigue      = np.random.binomial(1, 0.88, n_pos)
douleurs_art = np.random.binomial(1, 0.60, n_pos)
saison_pluie = np.random.binomial(1, 0.72, n_pos)   # saison pluvieuse → risque ++
freq_resp    = np.random.normal(24, 4, n_pos).clip(16, 45)
spo2         = np.random.normal(96.5, 1.8, n_pos).clip(88, 100)
antecedent_palu = np.random.binomial(1, 0.55, n_pos)

# ── Classe 0 : Non-paludisme ──────────────────────────────────────────────────
n_neg = N // 2

age_n          = np.random.choice(['<5ans', '5-14ans', '15-49ans', '50+ans'],
                                   n_neg, p=[0.25, 0.20, 0.40, 0.15])
sexe_n         = np.random.choice([0, 1], n_neg, p=[0.50, 0.50])
temperature_n  = np.random.normal(37.8, 0.60, n_neg).clip(36.0, 40.5)
duree_fievre_n = np.random.choice([1, 2, 3, 4, 5, 6, 7],
                                   n_neg, p=[0.20, 0.25, 0.22, 0.15, 0.10, 0.05, 0.03])
cephalees_n    = np.random.binomial(1, 0.45, n_neg)
vomissements_n = np.random.binomial(1, 0.30, n_neg)
frissons_n     = np.random.binomial(1, 0.25, n_neg)
fatigue_n      = np.random.binomial(1, 0.55, n_neg)
douleurs_art_n = np.random.binomial(1, 0.25, n_neg)
saison_pluie_n = np.random.binomial(1, 0.50, n_neg)
freq_resp_n    = np.random.normal(20, 4, n_neg).clip(14, 40)
spo2_n         = np.random.normal(98.2, 1.2, n_neg).clip(92, 100)
antecedent_n   = np.random.binomial(1, 0.25, n_neg)

# ── Assemblage ────────────────────────────────────────────────────────────────
def make_df(age, sexe, temp, duree, ceph, vomi, friss, fat, doul,
            saison, fr, spo2, antec, label):
    return pd.DataFrame({
        'age_groupe':      age,
        'sexe':            sexe,
        'temperature_C':   temp.round(1),
        'duree_fievre_j':  duree,
        'cephalees':       ceph,
        'vomissements':    vomi,
        'frissons':        friss,
        'fatigue':         fat,
        'douleurs_articulaires': doul,
        'saison_pluvieuse': saison,
        'frequence_resp':  fr.round(1),
        'spo2_pct':        spo2.round(1),
        'antecedent_palu': antec,
        'label':           label
    })

df_pos = make_df(age, sexe, temperature, duree_fievre, cephalees, vomissements,
                 frissons, fatigue, douleurs_art, saison_pluie, freq_resp,
                 spo2, antecedent_palu, 1)

df_neg = make_df(age_n, sexe_n, temperature_n, duree_fievre_n, cephalees_n,
                 vomissements_n, frissons_n, fatigue_n, douleurs_art_n,
                 saison_pluie_n, freq_resp_n, spo2_n, antecedent_n, 0)

df = pd.concat([df_pos, df_neg], ignore_index=True)

# 5 % de bruit de label (incertitude diagnostic réaliste)
noise_idx = np.random.choice(df.index, size=int(0.05 * len(df)), replace=False)
df.loc[noise_idx, 'label'] = 1 - df.loc[noise_idx, 'label']

df = df.sample(frac=1, random_state=42).reset_index(drop=True)
df.to_csv('data/malaria_clinical_proxy.csv', index=False)

print(f"Dataset : {df.shape[0]} patients, {df.shape[1]-1} features cliniques")
print(df['label'].value_counts().rename({1: 'Paludisme probable', 0: 'Non-paludisme'}))
