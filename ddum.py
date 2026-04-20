
# PROIECT DDUM — Detectia Melanomului cu Machine Learning
# Dataset: SIIM-ISIC 2020 Melanoma Classification

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, ConfusionMatrixDisplay,
    roc_curve, roc_auc_score, classification_report
)

import warnings
warnings.filterwarnings('ignore')

plt.rcParams['figure.figsize'] = (10, 5)
plt.rcParams['font.size'] = 12
sns.set_theme(style='whitegrid')

print("=" * 60)
print("  PROIECT DDUM — Detectia Melanomului")
print("=" * 60)


# =============================================================
# PASUL 1: INCARCAREA DATELOR
# =============================================================
print("\n[1] Incarcarea datelor...")

df = pd.read_csv('train.csv')

print(f"    Shape: {df.shape[0]} randuri x {df.shape[1]} coloane")
print(f"\n    Primele 3 randuri:")
print(df.head(3).to_string())

print(f"\n    Tipuri de date:")
print(df.dtypes.to_string())

print(f"\n    Statistici descriptive:")
print(df.describe().round(2).to_string())


# =============================================================
# PASUL 2: EDA — EXPLORAREA DATELOR
# =============================================================
print("\n[2] Explorarea datelor (EDA)...")

# --- 2.1 Valori lipsa ---
print("\n    Valori lipsa per coloana:")
missing = df.isnull().sum()
missing_pct = (missing / len(df) * 100).round(2)
missing_df = pd.DataFrame({'Numar lipsa': missing, 'Procent (%)': missing_pct})
print(missing_df.to_string())

# --- 2.2 Distributia target ---
counts = df['target'].value_counts()
print(f"\n    Distributia claselor (TARGET):")
print(f"    Benign    (0): {counts[0]:5d}  ({counts[0]/len(df)*100:.1f}%)")
print(f"    Malignant (1): {counts[1]:5d}  ({counts[1]/len(df)*100:.1f}%)")
print(f"\n    ATENTIE: Dataset dezechilibrat!")
print(f"    O acuratete de {counts[0]/len(df)*100:.0f}% se obtine trivial")
print(f"    prezicand mereu 'benign' — de aceea folosim Recall si ROC-AUC!")

fig, axes = plt.subplots(1, 2, figsize=(12, 5))
labels = ['Benign (0)', 'Malignant (1)']
axes[0].bar(labels, counts.values, color=['steelblue', 'tomato'], edgecolor='black')
axes[0].set_title('Distributia claselor')
axes[0].set_ylabel('Numar leziuni')
for i, v in enumerate(counts.values):
    axes[0].text(i, v + 100, f'{v}\n({v/len(df)*100:.1f}%)',
                 ha='center', fontweight='bold')
axes[1].pie(counts.values, labels=labels, autopct='%1.1f%%',
            colors=['steelblue', 'tomato'], startangle=90,
            wedgeprops={'edgecolor': 'black'})
axes[1].set_title('Proportia claselor')
plt.suptitle('Dezechilibrul claselor — Dataset Melanom ISIC 2020',
             fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig('fig1_distributie_target.png', dpi=150, bbox_inches='tight')
plt.show()
print("    >> Salvat: fig1_distributie_target.png")

# --- 2.3 Distributia varstei ---
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
for cls, color, lbl in [(0, 'steelblue', 'Benign'), (1, 'tomato', 'Malignant')]:
    axes[0].hist(df[df['target'] == cls]['age_approx'].dropna(),
                 bins=20, alpha=0.6, color=color, label=lbl, edgecolor='white')
axes[0].set_title('Distributia varstei pe clase')
axes[0].set_xlabel('Varsta aproximativa (ani)')
axes[0].set_ylabel('Frecventa')
axes[0].legend()

df.boxplot(column='age_approx', by='target', ax=axes[1],
           patch_artist=True,
           boxprops=dict(facecolor='steelblue', alpha=0.6))
axes[1].set_title('Boxplot varsta pe clase')
axes[1].set_xlabel('Target (0=Benign, 1=Malignant)')
axes[1].set_ylabel('Varsta (ani)')
plt.suptitle('')
plt.suptitle('Analiza varstei in functie de diagnostic', fontweight='bold')
plt.tight_layout()
plt.savefig('fig2_varsta.png', dpi=150, bbox_inches='tight')
plt.show()
print("    >> Salvat: fig2_varsta.png")

print(f"\n    Varsta medie - Benign:    {df[df['target']==0]['age_approx'].mean():.1f} ani")
print(f"    Varsta medie - Malignant: {df[df['target']==1]['age_approx'].mean():.1f} ani")

# --- 2.4 Variabile categorice ---
fig, axes = plt.subplots(1, 2, figsize=(16, 5))

ct_sex = pd.crosstab(df['sex'], df['target'], normalize='index') * 100
ct_sex.columns = ['Benign %', 'Malignant %']
ct_sex['Malignant %'].sort_values().plot(kind='bar', ax=axes[0],
    color='tomato', edgecolor='black', alpha=0.8)
axes[0].set_title('Rata de malignitate dupa sex (%)')
axes[0].set_xlabel('Sex')
axes[0].set_ylabel('% Malignant')
axes[0].tick_params(axis='x', rotation=0)
for p in axes[0].patches:
    axes[0].annotate(f'{p.get_height():.2f}%',
                     (p.get_x() + p.get_width()/2, p.get_height()),
                     ha='center', va='bottom', fontsize=11)

ct_site = pd.crosstab(df['anatom_site_general_challenge'], df['target'], normalize='index') * 100
ct_site.columns = ['Benign %', 'Malignant %']
ct_site['Malignant %'].sort_values(ascending=True).plot(kind='barh', ax=axes[1],
    color='tomato', edgecolor='black', alpha=0.8)
axes[1].set_title('Rata de malignitate dupa localizare (%)')
axes[1].set_xlabel('% Malignant')

plt.suptitle('Variabile categorice vs Diagnostic', fontweight='bold')
plt.tight_layout()
plt.savefig('fig3_categorice.png', dpi=150, bbox_inches='tight')
plt.show()
print("    >> Salvat: fig3_categorice.png")


# =============================================================
# PASUL 3: PREPROCESAREA DATELOR
# =============================================================
print("\n[3] Preprocesarea datelor...")

# --- 3.1 Eliminare coloane inutile si data leakage ---
# diagnosis:        'melanoma' apare DOAR cand target=1 -> leakage!
# benign_malignant: acelasi lucru cu target, doar text -> leakage!
# image_name:       ID imagine, nu e feature util
# patient_id:       ID pacient, nu e feature util
df_model = df.drop(columns=[
    'image_name',
    'patient_id',
    'diagnosis',         # DATA LEAKAGE
    'benign_malignant'   # DATA LEAKAGE
])
print(f"    Coloane dupa eliminare: {df_model.columns.tolist()}")

# --- 3.2 Encoding variabile categorice ---
# sex este binar (2 valori) -> Label Encoding: male=1, female=0
df_model['sex'] = df_model['sex'].map({'male': 1, 'female': 0})

# anatom_site are 6 categorii fara ordine -> One-Hot Encoding
# dummy_na=True: valorile lipsa devin o coloana separata (nu le pierdem)
df_model = pd.get_dummies(
    df_model,
    columns=['anatom_site_general_challenge'],
    drop_first=False,
    dummy_na=True,
    dtype=int
)
print(f"    Coloane dupa encoding ({df_model.shape[1]} total):")
for col in df_model.columns:
    print(f"      {col}")

# --- 3.3 Tratarea valorilor lipsa ---
print(f"\n    Valori lipsa inainte de imputare:")
print(df_model.isnull().sum()[df_model.isnull().sum() > 0].to_string())

# age_approx: distributie asimetrica -> imputare cu MEDIANA
age_imputer = SimpleImputer(strategy='median')
df_model['age_approx'] = age_imputer.fit_transform(df_model[['age_approx']])

# sex: variabila binara -> imputare cu MODUL (valoarea cea mai frecventa)
sex_imputer = SimpleImputer(strategy='most_frequent')
df_model['sex'] = sex_imputer.fit_transform(df_model[['sex']]).ravel()

print(f"    Valori lipsa dupa imputare: {df_model.isnull().sum().sum()} — OK!")

# --- 3.4 Separare X si y ---
X = df_model.drop('target', axis=1)
y = df_model['target']

print(f"\n    Features X: {X.shape[1]} coloane, {X.shape[0]} randuri")
print(f"    Target  y: {y.sum()} pozitive ({y.sum()/len(y)*100:.1f}%)")

# --- 3.5 Split 70% / 15% / 15% stratificat ---
# stratify=y: pastreaza proportia 98/2% in fiecare subset
X_train, X_temp, y_train, y_temp = train_test_split(
    X, y, test_size=0.30, random_state=42, stratify=y
)
X_val, X_test, y_val, y_test = train_test_split(
    X_temp, y_temp, test_size=0.50, random_state=42, stratify=y_temp
)

print(f"\n    Dimensiuni seturi:")
print(f"      Train:    {X_train.shape[0]:5d} exemple — Melanom: {y_train.sum()}")
print(f"      Validare: {X_val.shape[0]:5d} exemple — Melanom: {y_val.sum()}")
print(f"      Test:     {X_test.shape[0]:5d} exemple — Melanom: {y_test.sum()}")

# Grafic distributie split
fig, axes = plt.subplots(1, 3, figsize=(14, 4))
for ax, (title, y_split) in zip(axes, [
    ('Train (70%)', y_train),
    ('Validare (15%)', y_val),
    ('Test (15%)', y_test)
]):
    c = y_split.value_counts().sort_index()
    ax.bar(['Benign', 'Malignant'], c.values, color=['steelblue', 'tomato'], edgecolor='black')
    ax.set_title(title, fontweight='bold')
    ax.set_ylabel('Nr. exemple')
    for i, v in enumerate(c.values):
        ax.text(i, v+5, f'{v}\n({v/len(y_split)*100:.1f}%)', ha='center', fontsize=9)
plt.suptitle('Distributia claselor dupa split stratificat', fontweight='bold')
plt.tight_layout()
plt.savefig('fig4_split.png', dpi=150, bbox_inches='tight')
plt.show()
print("    >> Salvat: fig4_split.png")

# --- 3.6 Scalare (FARA data leakage) ---
# REGULA: fit() DOAR pe train, transform() pe val si test
# Altfel "scurgem" informatii din test in antrenare!
scaler = StandardScaler()
X_train_sc = scaler.fit_transform(X_train)   # fit + transform
X_val_sc   = scaler.transform(X_val)          # doar transform
X_test_sc  = scaler.transform(X_test)         # doar transform

print(f"\n    Scalare StandardScaler aplicata (fit doar pe train).")


# =============================================================
# PASUL 4: ANTRENAREA MODELELOR
# =============================================================
print("\n[4] Antrenarea modelelor...")

# --- Functie de evaluare ---
def evaluate(model, X_tr, X_v, X_te, y_tr, y_v, y_te, name):
    """
    Calculeaza Accuracy, Precision, Recall, F1 si ROC-AUC
    pe seturile de train, validare si test.
    Returneaza un dictionar cu toate rezultatele.
    """
    results = {}
    for label, Xs, ys in [
        ('Train',    X_tr, y_tr),
        ('Validare', X_v,  y_v),
        ('Test',     X_te, y_te)
    ]:
        yp    = model.predict(Xs)
        yprob = model.predict_proba(Xs)[:, 1]
        results[label] = {
            'Accuracy':  round(accuracy_score(ys, yp), 4),
            'Precision': round(precision_score(ys, yp, zero_division=0), 4),
            'Recall':    round(recall_score(ys, yp, zero_division=0), 4),
            'F1':        round(f1_score(ys, yp, zero_division=0), 4),
            'ROC-AUC':   round(roc_auc_score(ys, yprob), 4),
        }
    df_res = pd.DataFrame(results).T
    print(f"\n    === {name} ===")
    print(df_res.to_string())
    return results

# =============================================================
# ALGORITMUL 1: LOGISTIC REGRESSION
# =============================================================
print("\n    --- Algoritmul 1: Logistic Regression ---")
print("    Justificare: model liniar, simplu, interpretabil,")
print("    potrivit ca baseline pentru clasificare binara.")
print("    class_weight='balanced' compenseaza dezechilibrul 98/2%.")

lr = LogisticRegression(
    class_weight='balanced',  # important pentru date dezechilibrate
    max_iter=1000,
    random_state=42
)
lr.fit(X_train_sc, y_train)  # LR necesita date scalate
print("    Antrenat!")

res_lr = evaluate(
    lr,
    X_train_sc, X_val_sc, X_test_sc,
    y_train, y_val, y_test,
    'Logistic Regression (default, class_weight=balanced)'
)

# Coeficientii LR care feature are impact mai mare
coef_df = pd.DataFrame({
    'Feature':    X.columns,
    'Coeficient': lr.coef_[0]
}).sort_values('Coeficient', key=abs, ascending=True)

colors_lr = ['tomato' if c > 0 else 'steelblue' for c in coef_df['Coeficient']]
plt.figure(figsize=(9, 5))
plt.barh(coef_df['Feature'], coef_df['Coeficient'],
         color=colors_lr, edgecolor='black', alpha=0.8)
plt.axvline(0, color='black', linewidth=1)
plt.xlabel('Coeficient (pozitiv = creste riscul de melanom)')
plt.title('Coeficientii Logistic Regression\n(rosu = creste riscul, albastru = scade riscul)',
          fontweight='bold')
plt.tight_layout()
plt.savefig('fig5_coeficienti_lr.png', dpi=150, bbox_inches='tight')
plt.show()
print("    >> Salvat: fig5_coeficienti_lr.png")

# =============================================================
# ALGORITMUL 2: RANDOM FOREST
# =============================================================
print("\n    --- Algoritmul 2: Random Forest ---")
print("    Justificare: ansamblu de arbori de decizie (bagging),")
print("    captureaza relatii neliniare, robust la outlieri,")
print("    nu necesita scalare, ofera importanta caracteristicilor.")
print("    Antrenare... (poate dura 30-60 secunde)")

rf = RandomForestClassifier(
    n_estimators=200,         # 200 arbori de decizie
    class_weight='balanced',  # compenseaza dezechilibrul
    random_state=42,
    n_jobs=-1                 # foloseste toate CPU cores
)
rf.fit(X_train, y_train) 
print("    Antrenat!")

res_rf = evaluate(
    rf,
    X_train, X_val, X_test,
    y_train, y_val, y_test,
    'Random Forest (n_estimators=200, class_weight=balanced)'
)

# Importanta caracteristicilor
importances = pd.Series(rf.feature_importances_, index=X.columns)
importances = importances.sort_values(ascending=True)

plt.figure(figsize=(9, 5))
colors_imp = plt.cm.RdYlGn(np.linspace(0.2, 0.9, len(importances)))
bars = plt.barh(importances.index, importances.values,
                color=colors_imp, edgecolor='black', alpha=0.85)
plt.xlabel('Importanta (reducerea impuritatii Gini)')
plt.title('Importanta caracteristicilor — Random Forest', fontweight='bold')
for bar, val in zip(bars, importances.values):
    if val > 0.005:
        plt.text(val + 0.001, bar.get_y() + bar.get_height()/2,
                 f'{val:.3f}', va='center', fontsize=9)
plt.tight_layout()
plt.savefig('fig6_importanta_rf.png', dpi=150, bbox_inches='tight')
plt.show()
print("    >> Salvat: fig6_importanta_rf.png")

print("\n    Top 5 cele mai importante caracteristici:")
print(importances.sort_values(ascending=False).head(5).round(4).to_string())


# =============================================================
# PASUL 5: EVALUAREA MODELELOR
# =============================================================
print("\n[5] Evaluarea detaliata a modelelor...")

CLASS_NAMES = ['Benign', 'Malignant']

# --- Matrici de confuzie ---
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
for ax, model, Xte, title in [
    (axes[0], lr, X_test_sc, 'Logistic Regression'),
    (axes[1], rf, X_test,    'Random Forest')
]:
    cm = confusion_matrix(y_test, model.predict(Xte))
    tn, fp, fn, tp = cm.ravel()
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=CLASS_NAMES)
    disp.plot(ax=ax, colorbar=False, cmap='Blues')
    ax.set_title(f'{title}\nTN={tn}  FP={fp}  FN={fn}  TP={tp}',
                 fontweight='bold')

plt.suptitle('Matrici de confuzie — Set TEST\n'
             'FN = melanom diagnosticat gresit ca benign (periculos!)',
             fontsize=12, fontweight='bold')
plt.tight_layout()
plt.savefig('fig7_matrici_confuzie.png', dpi=150, bbox_inches='tight')
plt.show()
print("    >> Salvat: fig7_matrici_confuzie.png")

# --- Curbe ROC ---
plt.figure(figsize=(8, 6))
for model, Xte, name, color in [
    (lr, X_test_sc, 'Logistic Regression', 'steelblue'),
    (rf, X_test,    'Random Forest',        'tomato')
]:
    yprob = model.predict_proba(Xte)[:, 1]
    fpr, tpr, _ = roc_curve(y_test, yprob)
    auc = roc_auc_score(y_test, yprob)
    plt.plot(fpr, tpr, color=color, linewidth=2, label=f'{name} (AUC={auc:.3f})')

plt.plot([0,1],[0,1],'k--', linewidth=1, label='Clasificator aleator (AUC=0.5)')
plt.xlabel('False Positive Rate (1 - Specificity)')
plt.ylabel('True Positive Rate (Sensitivity / Recall)')
plt.title('Curbe ROC — Set TEST', fontweight='bold')
plt.legend(loc='lower right')
plt.tight_layout()
plt.savefig('fig8_roc.png', dpi=150, bbox_inches='tight')
plt.show()
print("    >> Salvat: fig8_roc.png")

# --- Raport complet ---
print("\n    === Raport clasificare — Logistic Regression ===")
print(classification_report(y_test, lr.predict(X_test_sc), target_names=CLASS_NAMES))

print("\n    === Raport clasificare — Random Forest ===")
print(classification_report(y_test, rf.predict(X_test), target_names=CLASS_NAMES))


# =============================================================
# PASUL 6: OPTIMIZAREA HIPERPARAMETRILOR
# =============================================================
print("\n[6] Optimizarea hiperparametrilor cu GridSearchCV...")
print("    Metrica de optimizare: ROC-AUC")
print("    (mai robusta decat Accuracy pe date dezechilibrate)")

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# --- GridSearch pentru Random Forest ---
print("\n    GridSearchCV Random Forest (poate dura 3-7 min)...")
param_grid_rf = {
    'n_estimators':     [100, 200, 300],
    'max_depth':        [5, 10, 20, None],
    'min_samples_leaf': [1, 5, 10],
}

grid_rf = GridSearchCV(
    RandomForestClassifier(class_weight='balanced', random_state=42, n_jobs=-1),
    param_grid_rf,
    cv=cv,
    scoring='roc_auc',
    n_jobs=-1,
    verbose=1
)
grid_rf.fit(X_train, y_train)

print(f"\n    Cei mai buni hiperparametri RF: {grid_rf.best_params_}")
print(f"    Cel mai bun ROC-AUC (CV 5-fold): {grid_rf.best_score_:.4f}")

rf_opt = grid_rf.best_estimator_
res_rf_opt = evaluate(
    rf_opt,
    X_train, X_val, X_test,
    y_train, y_val, y_test,
    f'Random Forest OPTIMIZAT — {grid_rf.best_params_}'
)

# --- GridSearch pentru Logistic Regression ---
print("\n    GridSearchCV Logistic Regression...")
param_grid_lr = {
    'C':       [0.001, 0.01, 0.1, 1, 10, 100],
    'penalty': ['l1', 'l2'],
    'solver':  ['liblinear']
}

grid_lr = GridSearchCV(
    LogisticRegression(class_weight='balanced', max_iter=1000, random_state=42),
    param_grid_lr,
    cv=cv,
    scoring='roc_auc',
    n_jobs=-1,
    verbose=1
)
grid_lr.fit(X_train_sc, y_train)

print(f"\n    Cei mai buni hiperparametri LR: {grid_lr.best_params_}")
print(f"    Cel mai bun ROC-AUC (CV 5-fold): {grid_lr.best_score_:.4f}")

lr_opt = grid_lr.best_estimator_
res_lr_opt = evaluate(
    lr_opt,
    X_train_sc, X_val_sc, X_test_sc,
    y_train, y_val, y_test,
    f'Logistic Regression OPTIMIZATA — {grid_lr.best_params_}'
)


# =============================================================
# PASUL 7: COMPARATIA FINALA A MODELELOR
# =============================================================
print("\n[7] Comparatia finala a tuturor modelelor...")

models_comp = {
    'Logistic Regression (default)':   (lr,     X_test_sc),
    'Logistic Regression (optimizat)': (lr_opt, X_test_sc),
    'Random Forest (default)':         (rf,     X_test),
    'Random Forest (optimizat)':       (rf_opt, X_test),
}

print(f"\n    {'Model':<40} {'Acc':>8} {'Prec':>8} {'Recall':>8} {'F1':>8} {'AUC':>8}")
print("    " + "-" * 84)

comp_results = []
for name, (model, Xte) in models_comp.items():
    yp    = model.predict(Xte)
    yprob = model.predict_proba(Xte)[:, 1]
    row = {
        'Model':     name,
        'Accuracy':  accuracy_score(y_test, yp),
        'Precision': precision_score(y_test, yp, zero_division=0),
        'Recall':    recall_score(y_test, yp, zero_division=0),
        'F1':        f1_score(y_test, yp, zero_division=0),
        'ROC-AUC':   roc_auc_score(y_test, yprob),
    }
    comp_results.append(row)
    print(f"    {name:<40}"
          f"{row['Accuracy']:>8.4f}"
          f"{row['Precision']:>8.4f}"
          f"{row['Recall']:>8.4f}"
          f"{row['F1']:>8.4f}"
          f"{row['ROC-AUC']:>8.4f}")

comp_df = pd.DataFrame(comp_results).set_index('Model')

# Grafic comparativ metrici
fig, axes = plt.subplots(1, 2, figsize=(16, 5))
metrics = ['Recall', 'ROC-AUC', 'F1', 'Precision']
x = np.arange(len(metrics))
width = 0.2
colors_bar = ['steelblue', 'cornflowerblue', 'tomato', 'salmon']

for i, (model_name, row) in enumerate(comp_df.iterrows()):
    vals = [row[m] for m in metrics]
    axes[0].bar(x + i*width, vals, width, label=model_name[:22],
                color=colors_bar[i], edgecolor='black', alpha=0.85)

axes[0].set_xticks(x + width*1.5)
axes[0].set_xticklabels(metrics)
axes[0].set_ylabel('Scor')
axes[0].set_ylim(0, 1.15)
axes[0].set_title('Comparatie metrici — toti algoritmii', fontweight='bold')
axes[0].legend(fontsize=8)

# Curbe ROC comparative
colors_roc = ['steelblue', 'cornflowerblue', 'tomato', 'salmon']
for (name, (model, Xte)), color in zip(models_comp.items(), colors_roc):
    yprob = model.predict_proba(Xte)[:, 1]
    fpr, tpr, _ = roc_curve(y_test, yprob)
    auc = roc_auc_score(y_test, yprob)
    axes[1].plot(fpr, tpr, color=color, linewidth=2,
                 label=f'{name[:22]} (AUC={auc:.3f})')

axes[1].plot([0,1],[0,1],'k--', linewidth=1, label='Aleator (AUC=0.5)')
axes[1].set_xlabel('False Positive Rate')
axes[1].set_ylabel('True Positive Rate (Recall)')
axes[1].set_title('Curbe ROC — toti algoritmii', fontweight='bold')
axes[1].legend(loc='lower right', fontsize=8)

plt.tight_layout()
plt.savefig('fig9_comparatie_finala.png', dpi=150, bbox_inches='tight')
plt.show()
print("    >> Salvat: fig9_comparatie_finala.png")


# =============================================================
# REZUMAT FINAL
# =============================================================
print("\n" + "=" * 60)
print("  REZUMAT FINAL")
print("=" * 60)

best_model_name = comp_df['ROC-AUC'].idxmax()
best_auc = comp_df['ROC-AUC'].max()
best_recall = comp_df.loc[best_model_name, 'Recall']

print(f"\n  Cel mai bun model (dupa ROC-AUC): {best_model_name}")
print(f"  ROC-AUC:  {best_auc:.4f}")
print(f"  Recall:   {best_recall:.4f}")
print(f"\n  Figuri generate:")
figuri = [
    'fig1_distributie_target.png',
    'fig2_varsta.png',
    'fig3_categorice.png',
    'fig4_split.png',
    'fig5_coeficienti_lr.png',
    'fig6_importanta_rf.png',
    'fig7_matrici_confuzie.png',
    'fig8_roc.png',
    'fig9_comparatie_finala.png',
]
for fig_name in figuri:
    print(f"    - {fig_name}")

print("\n  PROIECT COMPLET! Toate figurile au fost salvate.")
print("  Urmatorul pas: scrie raportul PDF folosind aceste rezultate.")
print("=" * 60)