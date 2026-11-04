import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.model_selection import train_test_split

df = pd.read_csv("/Users/patrickmincher/Desktop/LUSTIG SPSS/Generate_E6_E20_VKBAT/merged.csv")

entropy = pd.read_csv("/Users/patrickmincher/Desktop/LUSTIG SPSS/Generate_E6_E20_VKBAT/entropy_rewrite/blast_results/checkpoint.csv")

df = df.merge(entropy,on=["Protein", "Chain", 'No.', 'Res'], how = 'inner')

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