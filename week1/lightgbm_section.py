# ═══════════════════════════════════════════════════════════════
# Section 8B: LightGBM — Every Hyperparameter Explained
# ═══════════════════════════════════════════════════════════════
#
# LightGBM (Light Gradient Boosting Machine) is a gradient boosting
# framework that builds trees SEQUENTIALLY (unlike Random Forest's parallel trees).
#
# Key Differences from Random Forest:
# ┌────────────────────┬────────────────────────┬────────────────────────┐
# │ Aspect             │ Random Forest          │ LightGBM               │
# ├────────────────────┼────────────────────────┼────────────────────────┤
# │ Tree Building      │ Parallel (independent) │ Sequential (boosted)   │
# │ Split Strategy     │ Best split per node    │ Leaf-wise (best first) │
# │ Error Correction   │ Averages predictions   │ Each tree fixes prior  │
# │ Overfitting Risk   │ Lower (bagging)        │ Higher (must tune LR)  │
# │ Speed              │ Moderate               │ Very fast              │
# │ Missing Values     │ Needs imputation       │ Handles natively       │
# │ Categorical        │ Needs encoding         │ Handles natively       │
# └────────────────────┴────────────────────────┴────────────────────────┘
#
# LightGBM's LEAF-WISE growth:
#   Unlike level-wise (XGBoost), LightGBM picks the leaf with the
#   MAXIMUM delta loss to split → faster convergence, but can overfit
#   on small data (control with num_leaves & min_child_samples).

import lightgbm as lgb
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, roc_auc_score, log_loss, brier_score_loss)

# ═══════════════════════════════════════════════════════════════
# LIGHTGBM WITH EVERY HYPERPARAMETER COMMENTED
# ═══════════════════════════════════════════════════════════════

lgbm_model = lgb.LGBMClassifier(

    # ── BOOSTING_TYPE (Boosting algorithm) ──
    # 'gbdt'   = Gradient Boosted Decision Trees (DEFAULT, most common)
    # 'dart'   = Dropouts meet Multiple Additive Regression Trees
    #            (randomly drops trees to prevent overfitting, like dropout in NNs)
    # 'goss'   = Gradient-based One-Side Sampling
    #            (keeps all large gradient instances, randomly samples small gradients → faster)
    # 'rf'     = Random Forest mode (uses bagging instead of boosting)
    boosting_type='gbdt',

    # ── N_ESTIMATORS (Number of boosting rounds / trees) ──
    # Each round builds one tree that corrects errors of previous trees
    # MORE rounds = better fit but risk of overfitting (unlike RF where more is always better)
    # Use EARLY STOPPING rather than setting a fixed high number
    # Start with 500-1000, let early stopping find the optimal point
    n_estimators=500,

    # ── LEARNING_RATE (Shrinkage / Step size) ──
    # Controls how much each tree contributes to the final prediction
    # SMALLER = more conservative, needs MORE trees, but better generalisation
    # LARGER  = faster training, needs FEWER trees, but risk of overfitting
    # Rule of thumb: 0.01-0.1 for production, 0.3 for quick experiments
    # Lower LR + more trees = generally better than high LR + fewer trees
    learning_rate=0.05,

    # ── NUM_LEAVES (Maximum number of leaves per tree) ──
    # LightGBM's PRIMARY complexity control (more important than max_depth)
    # Should be LESS than 2^max_depth to avoid overfitting
    # DEFAULT = 31
    # Small datasets: 15-31
    # Large datasets: 63-255
    # Rule: num_leaves ≤ 2^max_depth (e.g., max_depth=7 → num_leaves ≤ 128)
    num_leaves=31,

    # ── MAX_DEPTH (Maximum depth of each tree) ──
    # -1 = No limit (DEFAULT) — depth controlled by num_leaves instead
    # Set to 6-8 to prevent overfitting alongside num_leaves
    # LightGBM is LEAF-WISE, so max_depth is a secondary control
    max_depth=7,

    # ── MIN_CHILD_SAMPLES (Minimum data points in a leaf) ──
    # Equivalent to min_samples_leaf in Random Forest
    # Higher = more conservative, reduces overfitting
    # DEFAULT = 20
    # Small datasets: 5-10
    # Large datasets: 50-100
    min_child_samples=20,

    # ── MIN_CHILD_WEIGHT (Minimum sum of hessian in a leaf) ──
    # Controls leaf creation based on the sum of second-order gradients
    # Higher = more conservative (fewer leaves created)
    # DEFAULT = 1e-3
    # For binary classification, hessian ≈ p*(1-p) per sample
    min_child_weight=1e-3,

    # ── SUBSAMPLE (Row subsampling ratio) ──
    # Fraction of training data sampled for each tree (like max_samples in RF)
    # 1.0 = Use all data (DEFAULT)
    # 0.7-0.9 = Typical range for reducing overfitting
    # Applied per iteration (not per leaf)
    subsample=0.8,

    # ── SUBSAMPLE_FREQ (How often to perform subsampling) ──
    # 0 = No subsampling (DEFAULT)
    # 1 = Subsample every iteration (set this when subsample < 1.0!)
    # k = Subsample every k iterations
    subsample_freq=1,

    # ── COLSAMPLE_BYTREE (Feature subsampling per tree) ──
    # Fraction of features randomly selected for each tree
    # Similar to max_features in Random Forest
    # 1.0 = Use all features (DEFAULT)
    # 0.6-0.9 = Typical range for reducing overfitting and correlation between trees
    colsample_bytree=0.8,

    # ── REG_ALPHA (L1 regularisation on leaf weights) ──
    # Adds penalty proportional to |weight| (like Lasso)
    # Drives small leaf weights to exactly 0 → feature selection effect
    # 0.0 = No L1 regularisation (DEFAULT)
    # 0.01-0.1 = Typical range
    reg_alpha=0.1,

    # ── REG_LAMBDA (L2 regularisation on leaf weights) ──
    # Adds penalty proportional to weight² (like Ridge)
    # Shrinks all leaf weights towards 0 (but doesn't zero them out)
    # 0.0 = No L2 regularisation (DEFAULT)
    # 0.1-10.0 = Typical range
    reg_lambda=1.0,

    # ── MIN_SPLIT_GAIN (Minimum loss reduction to make a split) ──
    # Only split if the gain exceeds this threshold (like min_impurity_decrease in RF)
    # 0.0 = Split as long as there's any improvement (DEFAULT)
    # Higher = more conservative (pre-pruning)
    min_split_gain=0.01,

    # ── SCALE_POS_WEIGHT (Handle class imbalance) ──
    # Weight applied to positive class
    # 1.0 = Equal weight (DEFAULT)
    # For imbalanced data: set to (num_negative / num_positive)
    # Alternative: use is_unbalance=True for automatic calculation
    scale_pos_weight=1.0,

    # ── IS_UNBALANCE (Automatic class imbalance handling) ──
    # True = Automatically set scale_pos_weight = neg_count / pos_count
    # False = No automatic balancing (DEFAULT)
    # Use EITHER is_unbalance=True OR set scale_pos_weight manually, not both
    is_unbalance=True,

    # ── OBJECTIVE (Loss function) ──
    # 'binary'       = Binary log loss (DEFAULT for classification)
    # 'cross_entropy' = Cross-entropy (different parameterisation)
    # 'regression'   = L2 loss (for regression tasks)
    # 'lambdarank'   = For ranking tasks
    objective='binary',

    # ── METRIC (Evaluation metric during training) ──
    # 'binary_logloss' = Log loss (DEFAULT for binary)
    # 'auc'            = AUC-ROC
    # 'binary_error'   = Error rate
    # Can specify multiple: ['binary_logloss', 'auc']
    metric='binary_logloss',

    # ── MAX_BIN (Max number of bins for feature discretisation) ──
    # LightGBM bins continuous features into discrete bins for speed
    # 255 = DEFAULT (good balance of speed and accuracy)
    # Lower (63-127) = Faster but less precise splits
    # Higher (512-1024) = More precise but slower and more memory
    max_bin=255,

    # ── CAT_SMOOTH (Smoothing for categorical features) ──
    # Only relevant when using categorical features natively
    # Reduces the effect of noisy categories with few data points
    # Higher = more smoothing (less overfitting to rare categories)
    # DEFAULT = 10
    cat_smooth=10,

    # ── N_JOBS (Parallel processing) ──
    # -1 = Use all CPU cores
    # LightGBM parallelises feature histogram computation, not trees
    n_jobs=-1,

    # ── RANDOM_STATE (Reproducibility) ──
    random_state=42,

    # ── VERBOSE (Training output) ──
    # -1 = Silent
    #  0 = Warnings only
    #  1 = Info level (shows training progress)
    verbose=-1,

    # ── IMPORTANCE_TYPE (Feature importance calculation method) ──
    # 'split' = Number of times a feature is used in splits (DEFAULT)
    # 'gain'  = Total gain of splits using this feature
    # 'gain' is generally more informative than 'split'
    importance_type='gain',
)

# ── Train with Early Stopping ──
# Early stopping monitors a validation metric and stops training
# when it hasn't improved for 'callbacks' rounds
# This AUTOMATICALLY finds the optimal number of trees

lgbm_model.fit(
    X_train, y_train,
    eval_set=[(X_test, y_test)],          # Monitor performance on test set
    callbacks=[
        lgb.early_stopping(
            stopping_rounds=50,            # Stop if no improvement for 50 rounds
            verbose=False                  # Don't print early stopping messages
        ),
        lgb.log_evaluation(period=0),      # Silent training
    ],
)

# ── Predictions ──
y_pred_lgbm = lgbm_model.predict(X_test)
y_proba_lgbm = lgbm_model.predict_proba(X_test)[:, 1]

print("LIGHTGBM RESULTS")
print("=" * 60)
print(f"Accuracy:  {accuracy_score(y_test, y_pred_lgbm):.4f}")
print(f"Precision: {precision_score(y_test, y_pred_lgbm):.4f}")
print(f"Recall:    {recall_score(y_test, y_pred_lgbm):.4f}")
print(f"F1 Score:  {f1_score(y_test, y_pred_lgbm):.4f}")
print(f"ROC-AUC:   {roc_auc_score(y_test, y_proba_lgbm):.4f}")
print(f"Log Loss:  {log_loss(y_test, y_proba_lgbm):.4f}")
print(f"Brier Score: {brier_score_loss(y_test, y_proba_lgbm):.4f}")
print(f"\nBest iteration: {lgbm_model.best_iteration_}")
print(f"Trees used: {lgbm_model.best_iteration_} out of {lgbm_model.n_estimators} max")


# ═══════════════════════════════════════════════════════════════
# LIGHTGBM — Feature Importance (Gain-based)
# ═══════════════════════════════════════════════════════════════
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

fi_lgbm = pd.DataFrame({
    'Feature': available_features,
    'Importance': lgbm_model.feature_importances_
}).sort_values('Importance', ascending=False)

print("\nLIGHTGBM FEATURE IMPORTANCE (by total gain)")
print("=" * 60)
for _, row in fi_lgbm.head(15).iterrows():
    bar = '█' * int(row['Importance'] / fi_lgbm['Importance'].max() * 40)
    print(f"  {row['Feature']:30s}  {row['Importance']:10.1f}  {bar}")

fig, ax = plt.subplots(figsize=(10, 8))
top15_lgbm = fi_lgbm.head(15)
ax.barh(top15_lgbm['Feature'][::-1], top15_lgbm['Importance'].values[::-1],
        color='#FF6F61')
ax.set_xlabel('Feature Importance (Total Gain)')
ax.set_title('LightGBM — Top 15 Feature Importances')
plt.tight_layout()
plt.show()


# ═══════════════════════════════════════════════════════════════
# MODEL COMPARISON — LR vs RF vs LightGBM
# ═══════════════════════════════════════════════════════════════
from sklearn.metrics import (roc_curve, precision_recall_curve,
                             average_precision_score, confusion_matrix,
                             ConfusionMatrixDisplay, classification_report)

models_comparison = {
    'Logistic Regression': (y_pred_lr, y_proba_lr),
    'Random Forest':       (y_pred_rf, y_proba_rf),
    'LightGBM':            (y_pred_lgbm, y_proba_lgbm),
}

# ── Comparison Table ──
print("\n" + "=" * 80)
print("MODEL COMPARISON — All Three Models")
print("=" * 80)
print(f"{'Metric':<15} {'Log. Reg.':>12} {'Random Forest':>14} {'LightGBM':>12}")
print(f"{'─'*15} {'─'*12} {'─'*14} {'─'*12}")

for metric_name, metric_fn in [
    ('Accuracy',  lambda yt, yp, ypr: accuracy_score(yt, yp)),
    ('Precision', lambda yt, yp, ypr: precision_score(yt, yp)),
    ('Recall',    lambda yt, yp, ypr: recall_score(yt, yp)),
    ('F1 Score',  lambda yt, yp, ypr: f1_score(yt, yp)),
    ('ROC-AUC',   lambda yt, yp, ypr: roc_auc_score(yt, ypr)),
    ('Log Loss',  lambda yt, yp, ypr: log_loss(yt, ypr)),
    ('Brier',     lambda yt, yp, ypr: brier_score_loss(yt, ypr)),
]:
    vals = []
    for name, (yp, ypr) in models_comparison.items():
        vals.append(metric_fn(y_test, yp, ypr))
    best_idx = np.argmin(vals) if metric_name in ['Log Loss', 'Brier'] else np.argmax(vals)
    markers = ['  ', '  ', '  ']
    markers[best_idx] = ' ★'
    print(f"{metric_name:<15} {vals[0]:>10.4f}{markers[0]} {vals[1]:>12.4f}{markers[1]} {vals[2]:>10.4f}{markers[2]}")


# ── ROC & PR Curves — All 3 Models ──
fig, axes = plt.subplots(1, 3, figsize=(20, 6))

# ROC Curves
colors = {'Logistic Regression': '#3498db', 'Random Forest': '#2ecc71', 'LightGBM': '#FF6F61'}
for name, (y_pred, y_proba) in models_comparison.items():
    fpr, tpr, _ = roc_curve(y_test, y_proba)
    auc = roc_auc_score(y_test, y_proba)
    axes[0].plot(fpr, tpr, label=f'{name} (AUC={auc:.3f})', linewidth=2, color=colors[name])
axes[0].plot([0,1], [0,1], 'k--', alpha=0.3)
axes[0].set_xlabel('False Positive Rate')
axes[0].set_ylabel('True Positive Rate')
axes[0].set_title('ROC Curves — All Models')
axes[0].legend()
axes[0].grid(True, alpha=0.3)

# Precision-Recall Curves
for name, (y_pred, y_proba) in models_comparison.items():
    prec, rec, _ = precision_recall_curve(y_test, y_proba)
    ap = average_precision_score(y_test, y_proba)
    axes[1].plot(rec, prec, label=f'{name} (AP={ap:.3f})', linewidth=2, color=colors[name])
axes[1].set_xlabel('Recall')
axes[1].set_ylabel('Precision')
axes[1].set_title('Precision-Recall Curves — All Models')
axes[1].legend()
axes[1].grid(True, alpha=0.3)

# Confusion Matrix — LightGBM
cm_lgbm = confusion_matrix(y_test, y_pred_lgbm)
ConfusionMatrixDisplay(cm_lgbm, display_labels=['Stay', 'Churn']).plot(ax=axes[2], cmap='Oranges')
axes[2].set_title('LightGBM Confusion Matrix')

plt.tight_layout()
plt.show()

# ── Classification Report ──
print("\n" + "=" * 50)
print("  LightGBM — Classification Report")
print("=" * 50)
print(classification_report(y_test, y_pred_lgbm, target_names=['Stay', 'Churn']))


# ═══════════════════════════════════════════════════════════════
# 10-FOLD CV — Including LightGBM
# ═══════════════════════════════════════════════════════════════
from sklearn.model_selection import StratifiedKFold, cross_validate

skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
scoring_metrics = ['accuracy', 'precision', 'recall', 'f1', 'roc_auc']

print("\n10-FOLD STRATIFIED CROSS-VALIDATION — LightGBM")
print("=" * 80)

lgbm_cv = lgb.LGBMClassifier(
    boosting_type='gbdt', n_estimators=500, learning_rate=0.05,
    num_leaves=31, max_depth=7, min_child_samples=20,
    subsample=0.8, subsample_freq=1, colsample_bytree=0.8,
    reg_alpha=0.1, reg_lambda=1.0, is_unbalance=True,
    random_state=42, verbose=-1, n_jobs=-1
)

cv_results_lgbm = cross_validate(lgbm_cv, X_scaled, y, cv=skf,
                                  scoring=scoring_metrics, n_jobs=-1)

print(f"\n{'─'*40}")
print(f"  LightGBM (10-Fold CV)")
print(f"{'─'*40}")
for metric in scoring_metrics:
    scores = cv_results_lgbm[f'test_{metric}']
    print(f"  {metric:12s}: {scores.mean():.4f} ± {scores.std():.4f}  "
          f"[min={scores.min():.4f}, max={scores.max():.4f}]")
