import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score, KFold
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor


# =======================
# Utility Functions
# =======================

def split_and_scale(data, target_col, test_size=0.2, seed=42):
    """
    Splits the dataset into training and testing sets.
    Then applies StandardScaler to normalize features.
    Returns scaled train/test features and labels.
    """
    X = data.drop(columns=[target_col]).copy()
    y = data[target_col].copy()

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=seed
    )

    # Feature scaling
    scaler = StandardScaler()
    X_train_scaled = pd.DataFrame(scaler.fit_transform(X_train), columns=X_train.columns, index=X_train.index)
    X_test_scaled = pd.DataFrame(scaler.transform(X_test), columns=X_test.columns, index=X_test.index)

    return X_train_scaled, X_test_scaled, y_train, y_test, scaler


def fit_ols_model(X, y):
    """
    Fits an Ordinary Least Squares (OLS) regression using statsmodels.
    Adds a constant (intercept) term to X.
    Returns the fitted model object.
    """
    X_const = sm.add_constant(X)
    model = sm.OLS(y, X_const).fit()
    return model


def calc_vif(X):
    """
    Computes Variance Inflation Factor (VIF) for each feature.
    Returns a DataFrame with feature names and VIF values.
    """
    X_const = sm.add_constant(X)
    vif_list = []
    for i, col in enumerate(X.columns):
        vif = variance_inflation_factor(X_const.values, i + 1)
        vif_list.append((col, vif))
    return pd.DataFrame(vif_list, columns=['feature', 'VIF']).sort_values('VIF', ascending=False).reset_index(drop=True)


def cross_val_rmse(X, y, folds=5, seed=42):
    """
    Performs k-fold cross validation using LinearRegression.
    Evaluates the model with Root Mean Squared Error (RMSE).
    Returns the mean RMSE across folds.
    """
    lr = LinearRegression()
    scores = cross_val_score(
        lr, X, y,
        scoring='neg_mean_squared_error',
        cv=KFold(n_splits=folds, shuffle=True, random_state=seed)
    )
    rmse = np.sqrt(-scores).mean()
    return rmse


# =======================
# Feature Selection Loop
# =======================

def feature_selection(X, y, pval_limit=0.05, vif_limit=10, vif_warn=4, max_adj_r2_drop=0.01, folds=5, seed=42):
    """
    Iteratively selects features using:
      - p-values of coefficients (remove if > pval_limit)
      - VIF values (remove if > vif_limit)
      - Cross-validation RMSE and adjusted R² (drop feature only if model doesn’t worsen)

    Prints R², adjusted R², and VIFs before/after dropping each feature.
    """
    X_working = X.copy()
    selection_log = []

    round_counter = 1

    while True:
        model = fit_ols_model(X_working, y)

        # Print model summary at this step
        print(f"\n========== Iteration {round_counter} ==========")
        print(f"Features in model: {list(X_working.columns)}")
        print(f"R²: {model.rsquared:.4f}, Adjusted R²: {model.rsquared_adj:.4f}")

        # Get VIFs
        vif_table = calc_vif(X_working)
        print("\nVIF values before dropping:")
        print(vif_table)

        # Get p-values
        pvalues = model.pvalues.drop('const') if 'const' in model.pvalues.index else model.pvalues
        pvalues = pvalues.sort_values(ascending=False)

        # Step 1: Drop high VIF feature
        high_vif_feats = vif_table[vif_table['VIF'] > vif_limit]['feature'].tolist()
        if high_vif_feats:
            candidate = high_vif_feats[0]
            new_model = fit_ols_model(X_working.drop(columns=[candidate]), y)

            print(f"\nTrying to drop high VIF feature: {candidate}")
            print(f"Adjusted R² before: {model.rsquared_adj:.4f}, after: {new_model.rsquared_adj:.4f}")

            if new_model.rsquared_adj >= model.rsquared_adj - max_adj_r2_drop:
                print(f"Dropped feature: {candidate}")
                new_vif = calc_vif(X_working.drop(columns=[candidate]))
                print("\nVIF values after dropping:")
                print(new_vif)

                selection_log.append({
                    'action': 'drop_high_vif',
                    'feature': candidate,
                    'adj_r2_before': model.rsquared_adj,
                    'adj_r2_after': new_model.rsquared_adj
                })
                X_working = X_working.drop(columns=[candidate])
                round_counter += 1
                continue
            else:
                print(f"Retained feature {candidate} (Adjusted R² dropped too much).")

        # Step 2: Drop based on high p-value or moderate VIF
        high_p_feats = pvalues[pvalues > pval_limit].index.tolist()
        moderate_vif_feats = vif_table[(vif_table['VIF'] > vif_warn) & (vif_table['VIF'] <= vif_limit)]['feature'].tolist()

        candidates = list(set(high_p_feats + moderate_vif_feats))
        if candidates:
            worst = candidates[0]
            new_model = fit_ols_model(X_working.drop(columns=[worst]), y)

            old_rmse = cross_val_rmse(X_working, y, folds=folds, seed=seed)
            new_rmse = cross_val_rmse(X_working.drop(columns=[worst]), y, folds=folds, seed=seed)

            print(f"\nTrying to drop candidate feature: {worst}")
            print(f"Adjusted R² before: {model.rsquared_adj:.4f}, after: {new_model.rsquared_adj:.4f}")
            print(f"CV RMSE before: {old_rmse:.4f}, after: {new_rmse:.4f}")

            if (new_model.rsquared_adj >= model.rsquared_adj - max_adj_r2_drop) and (new_rmse <= old_rmse):
                print(f"Dropped feature: {worst}")
                new_vif = calc_vif(X_working.drop(columns=[worst]))
                print("\nVIF values after dropping:")
                print(new_vif)

                selection_log.append({
                    'action': 'drop_candidate',
                    'feature': worst,
                    'adj_r2_before': model.rsquared_adj,
                    'adj_r2_after': new_model.rsquared_adj,
                    'rmse_before': old_rmse,
                    'rmse_after': new_rmse
                })
                X_working = X_working.drop(columns=[worst])
                round_counter += 1
                continue
            else:
                print(f"Retained feature {worst} (Adjusted R² or RMSE worsened).")

        # Stop if no more features can be dropped safely
        print("\nNo more features dropped. Final feature set:")
        print(list(X_working.columns))
        break

    return X_working, selection_log


# =======================
# Example Run
# =======================

if __name__ == "__main__":
    # Load your CSV
    data = pd.read_csv("your_file.csv")  # change to your CSV path

    # Split & scale
    X_train, X_test, y_train, y_test, scaler = split_and_scale(data, target_col="y")

    # Run feature selection
    final_X, log = feature_selection(X_train, y_train)

    print("\n==== Final Selected Features ====")
    print(final_X.columns.tolist())

    # Final model
    final_model = fit_ols_model(final_X, y_train)
    print("\nFinal Model Summary:")
    print(final_model.summary())
