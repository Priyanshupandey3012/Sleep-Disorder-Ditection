# ============================================================
#   SLEEP DISORDER DETECTION - MODEL BUILDING
#   Task    : Multiclass (5 disorder types)
#   Models  : Random Forest + XGBoost
#   Goal    : Highest Accuracy
# ============================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
import sys
import joblib

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (accuracy_score, classification_report,
                             confusion_matrix, roc_auc_score)
from sklearn.model_selection import cross_val_score, GridSearchCV
from xgboost import XGBClassifier

warnings.filterwarnings('ignore')
sys.stdout.reconfigure(encoding='utf-8')
sns.set_theme(style="whitegrid")
plt.rcParams['figure.dpi'] = 120

DISORDER_NAMES = {
    0: 'None',
    1: 'Insomnia',
    2: 'Sleep Apnea',
    3: 'Narcolepsy',
    4: 'Restless Leg Syndrome'
}

# ============================================================
# STEP 1: LOAD PROCESSED DATA
# ============================================================

print("=" * 60)
print("STEP 1: LOAD PROCESSED DATA")
print("=" * 60)

train_df = pd.read_csv("sleep_train.csv")
test_df  = pd.read_csv("sleep_test.csv")

X_train = train_df.drop(columns=['Disorder_Label'])
y_train = train_df['Disorder_Label'].astype(int)

X_test  = test_df.drop(columns=['Disorder_Label'])
y_test  = test_df['Disorder_Label'].astype(int)

print(f"Training set : {X_train.shape}")
print(f"Test set     : {X_test.shape}")
print(f"Classes      : {sorted(y_train.unique())} -> {[DISORDER_NAMES[i] for i in sorted(y_train.unique())]}")


# ============================================================
# STEP 2: TRAIN RANDOM FOREST
# ============================================================

print("\n" + "=" * 60)
print("STEP 2: TRAINING RANDOM FOREST")
print("=" * 60)

rf_model = RandomForestClassifier(
    n_estimators=200,
    max_depth=None,
    min_samples_split=2,
    min_samples_leaf=1,
    max_features='sqrt',
    random_state=42,
    n_jobs=-1,
    class_weight='balanced'
)

rf_model.fit(X_train, y_train)
rf_preds = rf_model.predict(X_test)
rf_acc   = accuracy_score(y_test, rf_preds)

print(f"Random Forest Accuracy : {rf_acc * 100:.2f}%")

# Cross-validation
rf_cv = cross_val_score(rf_model, X_train, y_train, cv=5, scoring='accuracy', n_jobs=-1)
print(f"Cross-Val Accuracy     : {rf_cv.mean()*100:.2f}% (+/- {rf_cv.std()*100:.2f}%)")


# ============================================================
# STEP 3: TRAIN XGBOOST
# ============================================================

print("\n" + "=" * 60)
print("STEP 3: TRAINING XGBOOST")
print("=" * 60)

xgb_model = XGBClassifier(
    n_estimators=300,
    max_depth=6,
    learning_rate=0.1,
    subsample=0.8,
    colsample_bytree=0.8,
    use_label_encoder=False,
    eval_metric='mlogloss',
    random_state=42,
    n_jobs=-1
)

xgb_model.fit(X_train, y_train,
              eval_set=[(X_test, y_test)],
              verbose=False)

xgb_preds = xgb_model.predict(X_test)
xgb_acc   = accuracy_score(y_test, xgb_preds)

print(f"XGBoost Accuracy   : {xgb_acc * 100:.2f}%")

xgb_cv = cross_val_score(xgb_model, X_train, y_train, cv=5, scoring='accuracy', n_jobs=-1)
print(f"Cross-Val Accuracy : {xgb_cv.mean()*100:.2f}% (+/- {xgb_cv.std()*100:.2f}%)")


# ============================================================
# STEP 4: COMPARE MODELS
# ============================================================

print("\n" + "=" * 60)
print("STEP 4: MODEL COMPARISON")
print("=" * 60)

results = pd.DataFrame({
    'Model'          : ['Random Forest', 'XGBoost'],
    'Test Accuracy'  : [rf_acc * 100, xgb_acc * 100],
    'CV Mean'        : [rf_cv.mean() * 100, xgb_cv.mean() * 100],
    'CV Std'         : [rf_cv.std() * 100, xgb_cv.std() * 100],
})
print(results.to_string(index=False))

best_model_name = 'XGBoost' if xgb_acc >= rf_acc else 'Random Forest'
best_model      = xgb_model if xgb_acc >= rf_acc else rf_model
best_preds      = xgb_preds if xgb_acc >= rf_acc else rf_preds
best_acc        = max(xgb_acc, rf_acc)

print(f"\n[WINNER] Best Model: {best_model_name} with {best_acc*100:.2f}% accuracy")


# ============================================================
# STEP 5: DETAILED CLASSIFICATION REPORT
# ============================================================

print("\n" + "=" * 60)
print("STEP 5: CLASSIFICATION REPORT (Best Model)")
print("=" * 60)

target_names = [DISORDER_NAMES[i] for i in sorted(y_test.unique())]
print(classification_report(y_test, best_preds, target_names=target_names))


# ============================================================
# STEP 6: HYPERPARAMETER TUNING (Best Model)
# ============================================================

print("\n" + "=" * 60)
print("STEP 6: HYPERPARAMETER TUNING")
print("=" * 60)
print("Tuning best model... (this may take 1-2 minutes)")

if best_model_name == 'XGBoost':
    param_grid = {
        'n_estimators' : [200, 300, 400],
        'max_depth'    : [4, 6, 8],
        'learning_rate': [0.05, 0.1, 0.2],
    }
    tuned = XGBClassifier(
        subsample=0.8, colsample_bytree=0.8,
        eval_metric='mlogloss', random_state=42, n_jobs=-1
    )
else:
    param_grid = {
        'n_estimators': [100, 200, 300],
        'max_depth'   : [None, 10, 20],
        'max_features': ['sqrt', 'log2'],
    }
    tuned = RandomForestClassifier(
        random_state=42, n_jobs=-1, class_weight='balanced'
    )

grid_search = GridSearchCV(
    tuned, param_grid, cv=3,
    scoring='accuracy', n_jobs=-1, verbose=0
)
grid_search.fit(X_train, y_train)

tuned_model  = grid_search.best_estimator_
tuned_preds  = tuned_model.predict(X_test)
tuned_acc    = accuracy_score(y_test, tuned_preds)

print(f"Best Parameters : {grid_search.best_params_}")
print(f"Before Tuning   : {best_acc*100:.2f}%")
print(f"After Tuning    : {tuned_acc*100:.2f}%")
print(f"Improvement     : +{(tuned_acc - best_acc)*100:.2f}%")


# ============================================================
# PLOT 1: MODEL ACCURACY COMPARISON
# ============================================================

fig, axes = plt.subplots(1, 2, figsize=(13, 5))
fig.suptitle('Model Performance Comparison', fontsize=14, fontweight='bold')

# Accuracy bar chart
models     = ['Random Forest', 'XGBoost', f'{best_model_name}\n(Tuned)']
accuracies = [rf_acc*100, xgb_acc*100, tuned_acc*100]
bar_colors = ['#3498db', '#e74c3c', '#2ecc71']
bars = axes[0].bar(models, accuracies, color=bar_colors, edgecolor='white', width=0.5)
axes[0].set_ylim(0, 110)
axes[0].set_ylabel('Accuracy (%)')
axes[0].set_title('Test Accuracy Comparison')
for bar, acc in zip(bars, accuracies):
    axes[0].text(bar.get_x() + bar.get_width()/2,
                 bar.get_height() + 1,
                 f'{acc:.2f}%', ha='center', fontweight='bold')

# CV score comparison
cv_means = [rf_cv.mean()*100, xgb_cv.mean()*100]
cv_stds  = [rf_cv.std()*100,  xgb_cv.std()*100]
axes[1].bar(['Random Forest', 'XGBoost'], cv_means,
            yerr=cv_stds, color=['#3498db', '#e74c3c'],
            capsize=8, edgecolor='white', width=0.4)
axes[1].set_ylim(0, 110)
axes[1].set_ylabel('CV Accuracy (%)')
axes[1].set_title('5-Fold Cross Validation Accuracy')
for i, (m, s) in enumerate(zip(cv_means, cv_stds)):
    axes[1].text(i, m + s + 1, f'{m:.2f}%', ha='center', fontweight='bold')

plt.tight_layout()
plt.savefig('model_01_accuracy_comparison.png', bbox_inches='tight')
plt.show()
print("Saved: model_01_accuracy_comparison.png")


# ============================================================
# PLOT 2: CONFUSION MATRIX (Tuned Model)
# ============================================================

fig, axes = plt.subplots(1, 2, figsize=(16, 6))
fig.suptitle('Confusion Matrices', fontsize=14, fontweight='bold')

for ax, preds, title in zip(axes,
                             [best_preds, tuned_preds],
                             [f'{best_model_name} (Original)',
                              f'{best_model_name} (Tuned)']):
    cm = confusion_matrix(y_test, preds)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax,
                xticklabels=target_names, yticklabels=target_names,
                linewidths=0.5)
    ax.set_title(title, fontsize=12)
    ax.set_xlabel('Predicted Label')
    ax.set_ylabel('True Label')
    ax.tick_params(axis='x', rotation=25)

plt.tight_layout()
plt.savefig('model_02_confusion_matrix.png', bbox_inches='tight')
plt.show()
print("Saved: model_02_confusion_matrix.png")


# ============================================================
# PLOT 3: FEATURE IMPORTANCE
# ============================================================

fig, axes = plt.subplots(1, 2, figsize=(16, 7))
fig.suptitle('Top 15 Feature Importances', fontsize=14, fontweight='bold')

for ax, model, title, color in zip(
        axes,
        [rf_model, tuned_model],
        ['Random Forest', f'{best_model_name} (Tuned)'],
        ['#3498db', '#2ecc71']):

    importances = pd.Series(model.feature_importances_, index=X_train.columns)
    top15 = importances.sort_values(ascending=True).tail(15)
    top15.plot(kind='barh', ax=ax, color=color, edgecolor='white')
    ax.set_title(f'{title} - Feature Importance')
    ax.set_xlabel('Importance Score')

plt.tight_layout()
plt.savefig('model_03_feature_importance.png', bbox_inches='tight')
plt.show()
print("Saved: model_03_feature_importance.png")


# ============================================================
# STEP 7: SAVE BEST MODEL
# ============================================================

print("\n" + "=" * 60)
print("STEP 7: SAVING BEST MODEL")
print("=" * 60)

joblib.dump(tuned_model, 'sleep_disorder_best_model.pkl')
joblib.dump(X_train.columns.tolist(), 'model_feature_names.pkl')
print("Saved: sleep_disorder_best_model.pkl")
print("Saved: model_feature_names.pkl")


# ============================================================
# FINAL SUMMARY
# ============================================================

print("\n" + "=" * 60)
print("MODEL BUILDING COMPLETE — FINAL SUMMARY")
print("=" * 60)
print(f"  Random Forest Accuracy : {rf_acc*100:.2f}%")
print(f"  XGBoost Accuracy       : {xgb_acc*100:.2f}%")
print(f"  Best Model             : {best_model_name}")
print(f"  After Tuning           : {tuned_acc*100:.2f}%")
print(f"  Features Used          : {X_train.shape[1]}")
print(f"  Classes Predicted      : 5 disorder types")
print("=" * 60)
print("  Saved Files:")
print("    -> sleep_disorder_best_model.pkl")
print("    -> model_feature_names.pkl")
print("    -> model_01_accuracy_comparison.png")
print("    -> model_02_confusion_matrix.png")
print("    -> model_03_feature_importance.png")
print("=" * 60)
print("  Next Step: Build Prediction App!")
print("=" * 60)
