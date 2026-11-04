import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split

# Load data
df = pd.read_csv("/Users/patrickmincher/Desktop/LUSTIG SPSS/Generate_E6_E20_VKBAT/merged.csv")
entropy = pd.read_csv("/Users/patrickmincher/Desktop/LUSTIG SPSS/Generate_E6_E20_VKBAT/entropy_rewrite/blast_results/checkpoint.csv")
df = df.merge(entropy, on=["Protein", "Chain", "No.", "Res"], how="inner")

# Trim disordered ends using isUnstruct
THRESHOLD = 0.9
WINDOW = 5

rows_to_keep = []
for (protein, chain), group in df.groupby(["Protein", "Chain"]):
    unstruct_vals = group["isUnstruct"].tolist()
    indices = group.index.tolist()
    n = len(unstruct_vals)

    # Find N-terminus cut point (trim while average is ABOVE threshold)
    n_cut = 0
    for i in range(n - WINDOW + 1):
        avg = sum(unstruct_vals[i:i + WINDOW]) / WINDOW
        if avg <= THRESHOLD:
            n_cut = i
            break

    # Find C-terminus cut point (trim while average is ABOVE threshold)
    c_cut = n
    for i in range(n - 1, WINDOW - 2, -1):
        avg = sum(unstruct_vals[i - WINDOW + 1:i + 1]) / WINDOW
        if avg <= THRESHOLD:
            c_cut = i + 1
            break

    # Keep only the residues between the cuts
    rows_to_keep.extend(indices[n_cut:c_cut])

df_trimmed = df.loc[rows_to_keep]

print(f"Before trimming: {len(df)} rows")
print(f"After trimming:  {len(df_trimmed)} rows")
print(f"Removed:         {len(df) - len(df_trimmed)} rows")

# Create target and features
df_trimmed['HAS_U'] = (df_trimmed["NUM_U"] > 0).astype(int)

ss_mapping = {'Helix': 1, 'Sheet': 2, 'Other': 3}
ss_cols = ['chou_fasman', 'sspro_5', 'gor4', 'dsc', 'jnet', 'psipred']
for col in ss_cols:
    df_trimmed[col + '_num'] = df_trimmed[col].map(ss_mapping)

feature_cols = ['E6', 'E20', 'isUnstruct', 'chou_fasman_num', 'sspro_5_num',
                'gor4_num', 'dsc_num', 'jnet_num', 'psipred_num']

df_clean = df_trimmed[feature_cols + ['HAS_U']].dropna()
X = df_clean[feature_cols]
y = df_clean['HAS_U']

print(f"\nClass distribution:")
print(y.value_counts())

# Train/test split and logistic regression
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = LogisticRegression(max_iter=1000, class_weight='balanced')
model.fit(X_train, y_train)

y_pred_prob = model.predict_proba(X_test)[:, 1]
auc = roc_auc_score(y_test, y_pred_prob)
print(f"\nAUC: {auc}")

for name, coef in zip(feature_cols, model.coef_[0]):
    print(f"    {name}: {coef:.4f}")
