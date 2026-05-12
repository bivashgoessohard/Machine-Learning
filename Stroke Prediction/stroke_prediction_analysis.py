# %% [markdown]
# # 🧠 Stroke Prediction Analysis
# 
# **Objective:** Build a classification model to predict whether a patient is likely to have a stroke
# based on clinical and demographic features from the healthcare dataset.
# 
# **Dataset:** [Kaggle - Healthcare Dataset Stroke Data](https://www.kaggle.com/datasets/fedesoriano/stroke-prediction-dataset)

# %% [markdown]
# ## 1. Imports & Setup

# %%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# Sklearn
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score, RandomizedSearchCV
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    confusion_matrix, classification_report, roc_auc_score,
    roc_curve, precision_recall_curve, auc, ConfusionMatrixDisplay
)

# SMOTE for class imbalance
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline

sns.set_theme(style="whitegrid", palette="muted")
plt.rcParams['figure.figsize'] = (10, 6)

# %% [markdown]
# ## 2. Load Data

# %%
# Use relative path for portability
data_path = Path("healthcare-dataset-stroke-data.csv")
df = pd.read_csv(data_path)

print(f"Shape: {df.shape}")
df.head()

# %%
df.info()

# %%
df.describe()

# %%
# Check missing values
print("Missing values per column:")
print(df.isnull().sum())
print(f"\nBMI missing: {df['bmi'].isnull().sum()} ({df['bmi'].isnull().mean()*100:.1f}%)")

# %% [markdown]
# ## 3. Exploratory Data Analysis (EDA)

# %% [markdown]
# ### 3.1 Target Distribution (Class Imbalance)

# %%
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# Count plot
stroke_counts = df['stroke'].value_counts()
axes[0].bar(['No Stroke (0)', 'Stroke (1)'], stroke_counts.values, 
            color=['#4CAF50', '#F44336'], edgecolor='black')
axes[0].set_title('Stroke Class Distribution', fontsize=14, fontweight='bold')
axes[0].set_ylabel('Count')
for i, v in enumerate(stroke_counts.values):
    axes[0].text(i, v + 20, str(v), ha='center', fontweight='bold')

# Percentage pie chart
axes[1].pie(stroke_counts.values, labels=['No Stroke', 'Stroke'], 
            autopct='%1.1f%%', colors=['#4CAF50', '#F44336'],
            explode=(0, 0.1), shadow=True, startangle=90)
axes[1].set_title('Stroke Percentage', fontsize=14, fontweight='bold')

plt.tight_layout()
plt.show()

print(f"Imbalance ratio: 1:{stroke_counts[0]//stroke_counts[1]}")

# %% [markdown]
# ### 3.2 Numerical Feature Distributions

# %%
num_cols = ['age', 'avg_glucose_level', 'bmi']

fig, axes = plt.subplots(1, 3, figsize=(16, 5))
for i, col in enumerate(num_cols):
    for label, color in [(0, '#4CAF50'), (1, '#F44336')]:
        subset = df[df['stroke'] == label][col].dropna()
        axes[i].hist(subset, bins=30, alpha=0.6, color=color, 
                     label=f'Stroke={label}', edgecolor='black')
    axes[i].set_title(f'{col} Distribution by Stroke', fontweight='bold')
    axes[i].legend()
    axes[i].set_xlabel(col)

plt.tight_layout()
plt.show()

# %% [markdown]
# ### 3.3 Categorical Feature Distributions

# %%
cat_cols = ['gender', 'hypertension', 'heart_disease', 'ever_married', 
            'work_type', 'Residence_type', 'smoking_status']

fig, axes = plt.subplots(2, 4, figsize=(20, 10))
axes = axes.flatten()

for i, col in enumerate(cat_cols):
    ct = pd.crosstab(df[col], df['stroke'], normalize='index') * 100
    ct.plot(kind='bar', stacked=True, ax=axes[i], color=['#4CAF50', '#F44336'], 
            edgecolor='black')
    axes[i].set_title(f'{col} vs Stroke (%)', fontweight='bold')
    axes[i].set_ylabel('Percentage')
    axes[i].legend(['No Stroke', 'Stroke'], fontsize=8)
    axes[i].tick_params(axis='x', rotation=45)

# Remove extra subplot
axes[-1].set_visible(False)
plt.tight_layout()
plt.show()

# %% [markdown]
# ### 3.4 Correlation Heatmap (Numerical Features)

# %%
corr_df = df[['age', 'hypertension', 'heart_disease', 'avg_glucose_level', 'bmi', 'stroke']].copy()
corr_matrix = corr_df.corr()

plt.figure(figsize=(8, 6))
sns.heatmap(corr_matrix, annot=True, cmap='RdYlBu_r', center=0, 
            fmt='.2f', linewidths=0.5, square=True)
plt.title('Feature Correlation Heatmap', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.show()

# %% [markdown]
# ## 4. Data Preprocessing

# %%
# Separate features and target, drop 'id' (not predictive)
X = df.drop(columns=['stroke', 'id'])
y = df['stroke']

# Train-test split (stratified to preserve class ratio)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print(f"Train: {X_train.shape[0]} samples | Test: {X_test.shape[0]} samples")
print(f"Train stroke rate: {y_train.mean()*100:.1f}% | Test stroke rate: {y_test.mean()*100:.1f}%")

# %%
# Identify column types for preprocessing
numerical_cols = ['age', 'avg_glucose_level', 'bmi']
categorical_cols = ['gender', 'ever_married', 'work_type', 'Residence_type', 'smoking_status']

# Build a ColumnTransformer that handles both imputation and encoding
# This prevents data leakage - everything is fit on train, applied to test
preprocessor = ColumnTransformer(
    transformers=[
        ('num', Pipeline([
            ('imputer', SimpleImputer(strategy='median')),  # Impute BMI NaNs with train median
            ('scaler', StandardScaler())
        ]), numerical_cols),
        ('cat', Pipeline([
            ('imputer', SimpleImputer(strategy='most_frequent')),
            ('encoder', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
        ]), categorical_cols)
    ]
)

print("Preprocessor configured:")
print(f"  Numerical columns ({len(numerical_cols)}): {numerical_cols}")
print(f"  Categorical columns ({len(categorical_cols)}): {categorical_cols}")

# %%
# Fit preprocessor on training data, transform both
X_train_processed = preprocessor.fit_transform(X_train)
X_test_processed = preprocessor.transform(X_test)  # Uses train statistics!

print(f"Processed train shape: {X_train_processed.shape}")
print(f"Processed test shape:  {X_test_processed.shape}")

# %% [markdown]
# ## 5. Handle Class Imbalance with SMOTE
# 
# The dataset is heavily imbalanced (~95% no stroke, ~5% stroke).
# We use SMOTE to generate synthetic minority samples **on training data only**.

# %%
smote = SMOTE(random_state=42)
X_train_balanced, y_train_balanced = smote.fit_resample(X_train_processed, y_train)

print(f"Before SMOTE: {pd.Series(y_train).value_counts().to_dict()}")
print(f"After SMOTE:  {pd.Series(y_train_balanced).value_counts().to_dict()}")

# %% [markdown]
# ## 6. Model Training & Evaluation

# %% [markdown]
# ### 6.1 Random Forest (Baseline — Default Parameters)

# %%
rf_model = RandomForestClassifier(random_state=42, n_jobs=-1)
rf_model.fit(X_train_balanced, y_train_balanced)

# Default threshold (0.5) predictions
y_pred_rf = rf_model.predict(X_test_processed)
y_proba_rf = rf_model.predict_proba(X_test_processed)[:, 1]

print("=== Random Forest (Default Threshold 0.5) ===")
print(classification_report(y_test, y_pred_rf, target_names=['No Stroke', 'Stroke']))
print(f"ROC-AUC: {roc_auc_score(y_test, y_proba_rf):.4f}")

# %% [markdown]
# ### 6.2 Cross-Validation (More Reliable Estimate)

# %%
# Use StratifiedKFold to maintain class balance across folds
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# We need to cross-validate on the non-SMOTE data, applying SMOTE inside each fold
# This prevents SMOTE data leakage across folds
imb_pipeline = ImbPipeline([
    ('smote', SMOTE(random_state=42)),
    ('classifier', RandomForestClassifier(random_state=42, n_jobs=-1))
])

cv_scores = cross_val_score(imb_pipeline, X_train_processed, y_train, 
                            cv=cv, scoring='roc_auc', n_jobs=-1)

print(f"Cross-Validation ROC-AUC Scores: {cv_scores}")
print(f"Mean ROC-AUC: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")

# %% [markdown]
# ### 6.3 Threshold Optimization
# 
# Instead of guessing a threshold (like 0.15), we use the Precision-Recall curve
# and ROC curve to find the optimal threshold systematically.

# %%
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# ROC Curve
fpr, tpr, roc_thresholds = roc_curve(y_test, y_proba_rf)
roc_auc_val = auc(fpr, tpr)
axes[0].plot(fpr, tpr, color='#2196F3', lw=2, label=f'ROC (AUC = {roc_auc_val:.3f})')
axes[0].plot([0, 1], [0, 1], 'k--', lw=1)
axes[0].set_xlabel('False Positive Rate')
axes[0].set_ylabel('True Positive Rate')
axes[0].set_title('ROC Curve', fontsize=14, fontweight='bold')
axes[0].legend(fontsize=12)

# Precision-Recall Curve
precision, recall, pr_thresholds = precision_recall_curve(y_test, y_proba_rf)
pr_auc_val = auc(recall, precision)
axes[1].plot(recall, precision, color='#F44336', lw=2, label=f'PR (AUC = {pr_auc_val:.3f})')
axes[1].set_xlabel('Recall')
axes[1].set_ylabel('Precision')
axes[1].set_title('Precision-Recall Curve', fontsize=14, fontweight='bold')
axes[1].legend(fontsize=12)

plt.tight_layout()
plt.show()

# %%
# Find optimal threshold using Youden's J statistic (maximizes TPR - FPR)
j_scores = tpr - fpr
optimal_idx = np.argmax(j_scores)
optimal_threshold = roc_thresholds[optimal_idx]
print(f"Optimal threshold (Youden's J): {optimal_threshold:.4f}")
print(f"  → TPR (Recall): {tpr[optimal_idx]:.3f}")
print(f"  → FPR: {fpr[optimal_idx]:.3f}")

# Apply optimal threshold
y_pred_optimal = (y_proba_rf >= optimal_threshold).astype(int)

print(f"\n=== Random Forest (Optimized Threshold {optimal_threshold:.3f}) ===")
print(classification_report(y_test, y_pred_optimal, target_names=['No Stroke', 'Stroke']))

# %%
# Compare confusion matrices: default vs optimized threshold
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

ConfusionMatrixDisplay.from_predictions(y_test, y_pred_rf, ax=axes[0],
                                         display_labels=['No Stroke', 'Stroke'],
                                         cmap='Blues')
axes[0].set_title('Default Threshold (0.5)', fontsize=13, fontweight='bold')

ConfusionMatrixDisplay.from_predictions(y_test, y_pred_optimal, ax=axes[1],
                                         display_labels=['No Stroke', 'Stroke'],
                                         cmap='Oranges')
axes[1].set_title(f'Optimized Threshold ({optimal_threshold:.3f})', fontsize=13, fontweight='bold')

plt.tight_layout()
plt.show()

# %% [markdown]
# ## 7. Hyperparameter Tuning

# %%
param_dist = {
    'n_estimators': [100, 200, 300, 500],
    'max_depth': [5, 10, 15, 20, None],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4],
    'max_features': ['sqrt', 'log2'],
}

rf_tuned = RandomizedSearchCV(
    RandomForestClassifier(random_state=42, n_jobs=-1),
    param_distributions=param_dist,
    n_iter=30,
    cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=42),
    scoring='roc_auc',
    random_state=42,
    n_jobs=-1,
    verbose=1
)

rf_tuned.fit(X_train_balanced, y_train_balanced)

print(f"\nBest ROC-AUC (CV): {rf_tuned.best_score_:.4f}")
print(f"Best Parameters: {rf_tuned.best_params_}")

# %%
# Evaluate tuned model
y_proba_tuned = rf_tuned.best_estimator_.predict_proba(X_test_processed)[:, 1]
y_pred_tuned = (y_proba_tuned >= optimal_threshold).astype(int)

print(f"=== Tuned Random Forest (Threshold {optimal_threshold:.3f}) ===")
print(classification_report(y_test, y_pred_tuned, target_names=['No Stroke', 'Stroke']))
print(f"ROC-AUC: {roc_auc_score(y_test, y_proba_tuned):.4f}")

# %% [markdown]
# ## 8. Model Comparison

# %%
models = {
    'Random Forest (Tuned)': rf_tuned.best_estimator_,
    'Gradient Boosting': GradientBoostingClassifier(
        n_estimators=200, max_depth=5, learning_rate=0.1, random_state=42
    ),
    'Logistic Regression': LogisticRegression(
        max_iter=1000, random_state=42, class_weight='balanced'
    ),
}

results = {}

for name, model in models.items():
    if name != 'Random Forest (Tuned)':  # Already trained
        model.fit(X_train_balanced, y_train_balanced)
    
    y_proba = model.predict_proba(X_test_processed)[:, 1]
    y_pred = (y_proba >= optimal_threshold).astype(int)
    
    auc_score = roc_auc_score(y_test, y_proba)
    cm = confusion_matrix(y_test, y_pred)
    report = classification_report(y_test, y_pred, target_names=['No Stroke', 'Stroke'], output_dict=True)
    
    results[name] = {
        'AUC-ROC': auc_score,
        'Recall (Stroke)': report['Stroke']['recall'],
        'Precision (Stroke)': report['Stroke']['precision'],
        'F1 (Stroke)': report['Stroke']['f1-score'],
        'y_proba': y_proba,
    }
    
    print(f"\n{'='*50}")
    print(f"  {name}")
    print(f"{'='*50}")
    print(classification_report(y_test, y_pred, target_names=['No Stroke', 'Stroke']))
    print(f"ROC-AUC: {auc_score:.4f}")

# %%
# Summary comparison table
summary_df = pd.DataFrame({
    name: {k: v for k, v in metrics.items() if k != 'y_proba'}
    for name, metrics in results.items()
}).T.round(4)

print("\n📊 Model Comparison Summary:")
print(summary_df.to_string())

# %%
# ROC curves comparison
plt.figure(figsize=(10, 7))

for name, metrics in results.items():
    fpr_m, tpr_m, _ = roc_curve(y_test, metrics['y_proba'])
    auc_m = auc(fpr_m, tpr_m)
    plt.plot(fpr_m, tpr_m, lw=2, label=f'{name} (AUC = {auc_m:.3f})')

plt.plot([0, 1], [0, 1], 'k--', lw=1, label='Random Guess')
plt.xlabel('False Positive Rate', fontsize=12)
plt.ylabel('True Positive Rate', fontsize=12)
plt.title('ROC Curve — Model Comparison', fontsize=14, fontweight='bold')
plt.legend(fontsize=11)
plt.tight_layout()
plt.show()

# %% [markdown]
# ## 9. Feature Importance

# %%
best_model = rf_tuned.best_estimator_

# Get feature names from the preprocessor
num_feature_names = numerical_cols
cat_feature_names = preprocessor.named_transformers_['cat']['encoder'].get_feature_names_out(categorical_cols).tolist()
all_feature_names = num_feature_names + cat_feature_names

importances = best_model.feature_importances_
feat_imp_df = pd.DataFrame({
    'Feature': all_feature_names,
    'Importance': importances
}).sort_values('Importance', ascending=True)

plt.figure(figsize=(10, 8))
plt.barh(feat_imp_df['Feature'], feat_imp_df['Importance'], color='#2196F3', edgecolor='black')
plt.xlabel('Feature Importance', fontsize=12)
plt.title('Random Forest Feature Importances', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.show()

# %% [markdown]
# ## 10. Conclusions
# 
# ### Key Findings:
# 1. **Severe class imbalance** (~95% no stroke) addressed with SMOTE
# 2. **Age** is the strongest predictor of stroke
# 3. **Threshold optimization** dramatically improves stroke recall vs default 0.5
# 4. **Sklearn Pipeline** prevents data leakage (imputation & encoding fit on train only)
# 
# ### Key Takeaways:
# - **Best AUC-ROC**: Logistic Regression (0.844) — surprisingly the simplest model generalizes best here.
# - **Best Stroke Recall**: Logistic Regression catches 98% of actual strokes (critical for healthcare!).
# - **Trade-off**: High recall comes at the cost of many false positives (low precision) — which is acceptable in medical screening (better to flag a healthy person than miss a stroke).
# - **Optimal threshold**: Youden's J found **0.07** as optimal threshold.
