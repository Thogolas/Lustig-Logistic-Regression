import os
import tarfile
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, roc_curve, confusion_matrix
from sklearn.model_selection import train_test_split

PISCES_TAR = ""  # set to .tar path to filter by PISCES, leave empty to use full dataset

df = pd.read_csv("/Users/patrickmincher/Desktop/LUSTIG SPSS/Generate_E6_E20_VKBAT/merged_complete.csv")

if PISCES_TAR:
    with tarfile.open(PISCES_TAR) as tar:
        members = [m for m in tar.getmembers() if not m.name.endswith(('.fasta', '.log', '.pc.gz')) and m.isfile() and '/' not in m.name]
        f = tar.extractfile(members[0])
        lines = f.read().decode().strip().split('\n')
    pisces_chains = []
    for line in lines[1:]:
        pdbchain = line.split()[0].strip()
        pisces_chains.append((pdbchain[:-1], pdbchain[-1]))
    pisces_df = pd.DataFrame(pisces_chains, columns=['Protein', 'Chain'])
    df = df.merge(pisces_df, on=['Protein', 'Chain'], how='inner')
    print(f"PISCES filter: {len(pisces_chains)} chains, {df[['Protein','Chain']].drop_duplicates().shape[0]} matched")

df['HAS_U'] = (df["NUM_U"] > 0).astype(int)

ss_mapping = {'Helix': 1, 'Sheet': 2, "Other": 3}

ss_cols = ['chou_fasman', 'sspro_5', 'gor4', 'dsc', 'jnet', 'psipred']
for col in ss_cols:
    df[col + '_num'] = df[col].map(ss_mapping)



feature_cols = ['E6', 'E20', 'isUnstruct', 'chou_fasman_num', 'sspro_5_num', 'gor4_num', 'dsc_num', 'jnet_num', 'psipred_num']

x = df[feature_cols]
y = df["HAS_U"]

print(y.value_counts())



df_clean = df[feature_cols + ['HAS_U']].dropna()
x = df_clean[feature_cols]
y = df_clean['HAS_U']

model = LogisticRegression(max_iter=1000, class_weight='balanced')
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

model = LogisticRegression(max_iter=1000, class_weight='balanced')
model.fit(x_train, y_train)

y_pred_prob = model.predict_proba(x_test)[:, 1]
auc = roc_auc_score(y_test, y_pred_prob)
print(f"AUC: {auc}")

for name, coef in zip(feature_cols, model.coef_[0]):
    print(f"    {name}: {coef:.4f}")

y_pred = model.predict(x_test)
tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
print(f"\nConfusion Matrix:")
print(f"    True Positives  (switch predicted switch):     {tp}")
print(f"    True Negatives  (non-switch predicted non-switch): {tn}")
print(f"    False Positives (non-switch predicted switch): {fp}")
print(f"    False Negatives (switch predicted non-switch): {fn}")