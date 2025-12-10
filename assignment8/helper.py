# ==============================================================================
# Assignment 8: Polynomial Regression - Complete Code
# ==============================================================================

# --- 1. SETUP: IMPORT LIBRARIES ---
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, KFold, cross_val_score, GridSearchCV
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.pipeline import Pipeline
from sklearn.metrics import r2_score, mean_squared_error
import warnings

# Suppress warnings for a cleaner output
warnings.filterwarnings('ignore')

# Set plot style for better visuals
plt.style.use('seaborn-v0_8-whitegrid')


# --- 2. LOAD DATA AND PERFORM INITIAL SPLIT ---

# 🚨 IMPORTANT: REPLACE THIS SECTION WITH YOUR ACTUAL DATA FILE
# For example: df = pd.read_csv('your_data_file.csv')
# ----------------------------------------------------------------
# For demonstration purposes, synthetic data is generated here.
np.random.seed(42)

def load_data(path="C:\MACHINE LEARNING LAB\assignment8\polynomial_regression.csv"):
    df = pd.read_csv(path)
    if {'x', 'y'}.issubset(df.columns):
        return df[['x', 'y']].rename(columns={'x':'x','y':'y'})
    # fallback: assume first two columns are x and y
    cols = df.columns.tolist()
    return df[[cols[0], cols[1]]].rename(columns={cols[0]:'x', cols[1]:'y'})
np.random.seed(42)
data = load_data("C:\MACHINE LEARNING LAB\assignment8\polynomial_regression.csv")
X_data = data[['x']].values
y_data = data['y'].values
X_data = X_data.flatten()
y_data = y_data.flatten()

df = pd.DataFrame({'x': X_data, 'y': y_data})
# --- END OF REPLACEMENT SECTION ---

# Display the first few rows of the dataframe
# print("Data Head:")
# print(df.head())


# Define features (X) and target (y)
X = df[['x']]
y = df['y']

# Perform the 80:20 split for training and testing data
X_train_full, X_test, y_train_full, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42
)

print(f"Full training set size: {len(X_train_full)} points")
print(f"Test set size: {len(X_test)} points")
print("\n" + "="*50 + "\n")


# --- PART 1: OBSERVING ERROR FLUCTUATION ---
print("--- Starting Part 1: Observing Error Fluctuation ---")
results = []
n_samples = 30
sample_size = 20
max_degree = 10

for i in range(n_samples):
    X_sample, y_sample = X_train_full.sample(n=sample_size, random_state=i).align(y_train_full, join='inner', axis=0)
    X_sample = X_sample.sort_index()
    y_sample = y_sample.sort_index()

    for degree in range(1, max_degree + 1):
        pipeline = Pipeline([
            ('poly', PolynomialFeatures(degree=degree, include_bias=False)),
            ('regressor', LinearRegression())
        ])
        pipeline.fit(X_sample, y_sample)

        y_train_pred = pipeline.predict(X_sample)
        train_error = mean_squared_error(y_sample, y_train_pred)

        y_test_pred = pipeline.predict(X_test)
        test_error = mean_squared_error(y_test, y_test_pred)

        results.append({
            'sample_id': i, 'degree': degree,
            'train_error': train_error, 'test_error': test_error,
            'error_diff': train_error - test_error
        })

results_df = pd.DataFrame(results)

# Create violin plots for Part 1
fig, axes = plt.subplots(1, 2, figsize=(18, 7))
fig.suptitle('Part 1: Error Fluctuation Analysis', fontsize=16)

# Plot 1: Test Error vs. Degree
sns.violinplot(x='degree', y='test_error', data=results_df, ax=axes[0], palette='viridis', cut=0)
axes[0].set_title('Distribution of Test Error vs. Polynomial Degree', fontsize=14)
axes[0].set_xlabel('Polynomial Degree', fontsize=12)
axes[0].set_ylabel('Test Error (MSE)', fontsize=12)
axes[0].set_ylim(bottom=0, top=results_df['test_error'].quantile(0.95))

# Plot 2: (Train Error - Test Error) vs. Degree
sns.violinplot(x='degree', y='error_diff', data=results_df, ax=axes[1], palette='plasma', cut=0)
axes[1].set_title('Distribution of (Train Error - Test Error) vs. Degree', fontsize=14)
axes[1].set_xlabel('Polynomial Degree', fontsize=12)
axes[1].set_ylabel('Train Error - Test Error', fontsize=12)

plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.show()

print("Part 1 complete. Plots are displayed.")
print("\n" + "="*50 + "\n")


# --- PART 2: K-FOLD CROSS-VALIDATION ON A SAMPLE ---
print("--- Starting Part 2: K-Fold CV on a Sample ---")
X_sample_p2, y_sample_p2 = X_train_full.sample(n=20, random_state=100).align(y_train_full, join='inner', axis=0)

k = 5
cv_scores = []
degrees_to_try = range(1, 16)

for degree in degrees_to_try:
    pipeline = Pipeline([
        ('poly', PolynomialFeatures(degree=degree, include_bias=False)),
        ('regressor', LinearRegression())
    ])
    scores = cross_val_score(pipeline, X_sample_p2, y_sample_p2, cv=k, scoring='neg_mean_squared_error')
    cv_scores.append(-scores.mean())

best_degree_p2 = degrees_to_try[np.argmin(cv_scores)]
print(f"Optimal degree found using 5-fold CV: {best_degree_p2}")

# Plot the CV scores vs degree for Part 2
plt.figure(figsize=(12, 6))
plt.plot(degrees_to_try, cv_scores, marker='o', linestyle='-', color='royalblue')
plt.title('Part 2: 5-Fold Cross-Validation Error vs. Polynomial Degree', fontsize=14)
plt.xlabel('Polynomial Degree', fontsize=12)
plt.ylabel('Average Cross-Validation MSE', fontsize=12)
plt.axvline(x=best_degree_p2, color='red', linestyle='--', label=f'Best Degree: {best_degree_p2}')
plt.xticks(degrees_to_try)
plt.legend()
plt.grid(True)
plt.show()

# Train and evaluate the final model for Part 2
final_pipeline_p2 = Pipeline([
    ('poly', PolynomialFeatures(degree=best_degree_p2, include_bias=False)),
    ('regressor', LinearRegression())
])
final_pipeline_p2.fit(X_sample_p2, y_sample_p2)
y_test_pred_p2 = final_pipeline_p2.predict(X_test)
final_r2_p2 = r2_score(y_test, y_test_pred_p2)

print(f"\n--- Part 2 Final Model (Degree = {best_degree_p2}) ---")
print(f"Test Set R-squared (R²): {final_r2_p2:.4f}")
print("Part 2 complete. Plot is displayed.")
print("\n" + "="*50 + "\n")


# --- PART 3: REGULARIZATION ON THE FULL TRAINING DATA ---
print("--- Starting Part 3: Regularization on Full Training Data ---")

pipeline_p3 = Pipeline([
    ('poly', PolynomialFeatures(include_bias=False)),
    ('scaler', StandardScaler()),
    ('regressor', 'passthrough')
])

param_grid = {
    'poly__degree': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12],
    'regressor__alpha': np.logspace(-3, 3, 7)
}

# L2 Regularization (Ridge)
print("Running GridSearchCV for Ridge (L2)... (This may take a few minutes)\n")
param_grid_ridge = {**param_grid, 'regressor': [Ridge()]}
# Set n_jobs=1 to disable parallel processing and avoid Windows-specific errors
grid_search_ridge = GridSearchCV(pipeline_p3, param_grid_ridge, cv=10, scoring='r2', n_jobs=1)
grid_search_ridge.fit(X_train_full, y_train_full)
print("--- Part 3: L2 Regularization (Ridge) Results ---")
print(f"Best parameters found: {grid_search_ridge.best_params_}")
print(f"Best 10-fold CV R² score: {grid_search_ridge.best_score_:.4f}\n")

# L1 Regularization (Lasso)
print("Running GridSearchCV for Lasso (L1)... (This may take a few minutes)\n")
param_grid_lasso = {**param_grid, 'regressor': [Lasso(max_iter=20000)]}
# Set n_jobs=1 to disable parallel processing
grid_search_lasso = GridSearchCV(pipeline_p3, param_grid_lasso, cv=10, scoring='r2', n_jobs=1)
grid_search_lasso.fit(X_train_full, y_train_full)
print("--- Part 3: L1 Regularization (Lasso) Results ---")
print(f"Best parameters found: {grid_search_lasso.best_params_}")
print(f"Best 10-fold CV R² score: {grid_search_lasso.best_score_:.4f}\n")

# Final Model Selection and Evaluation
if grid_search_ridge.best_score_ > grid_search_lasso.best_score_:
    print("🏆 Ridge (L2) model performed better in cross-validation.")
    best_model = grid_search_ridge.best_estimator_
    model_type = "Ridge"
else:
    print("🏆 Lasso (L1) model performed better in cross-validation.")
    best_model = grid_search_lasso.best_estimator_
    model_type = "Lasso"

y_test_pred_final = best_model.predict(X_test)
final_r2 = r2_score(y_test, y_test_pred_final)

print(f"\n--- ✅ FINAL MODEL PERFORMANCE ON TEST SET ({model_type}) ---")
print(f"The final R-squared (R²) score is: {final_r2:.4f}")
print("Part 3 complete. Assignment finished.")
print("\n" + "="*50 + "\n")