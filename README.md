# Melanoma Detection — SIIM-ISIC 2020

Binary classification of skin lesions (benign vs malignant) using clinical metadata from the ISIC 2020 dataset (33,126 records, 1.8% malignant).

## Models
- **Logistic Regression** — best Recall: **0.701** (detects 61/87 melanomas)
- **Random Forest** — best ROC-AUC: **0.705**, F1: **0.075**

Both models use `class_weight="balanced"` and were tuned with GridSearchCV + StratifiedKFold. Key predictor: patient age (Gini importance = 0.744).

## Usage
```bash
pip install numpy pandas scikit-learn matplotlib seaborn
python ddum.py
```
Place `train.csv` from [Kaggle](https://www.kaggle.com/c/siim-isic-melanoma-classification/data) in the project root.

## Reference
Rotemberg et al., *Scientific Data* 8, 34 (2021). https://doi.org/10.1038/s41597-021-00815-z
