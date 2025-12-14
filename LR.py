import pandas as pd
import numpy as np

from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import scipy.stats as stats  # Shapiro-Wilk test
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

# configs both dataset with and without outliners
CFG_LIST = [
    {
        "label": "PC_5_noOutliers",
        "train_path": "train_reduced_GA13_5PCA_log_noOutliers.csv",
        "val_path":   "val_reduced_GA13_5PCA_log_noOutliers.csv",
        "test_path":  "test_reduced_GA13_5PCA_log_noOutliers.csv",
    },
    {
        "label": "PC_5",
        "train_path": "train_reduced_GA13_5PCA_log.csv",
        "val_path":   "val_reduced_GA13_5PCA_log.csv",
        "test_path":  "test_reduced_GA13_5PCA_log.csv",
    }
]

# y value
TARGET_COL = "taxi_time"
# polynomial features can be changed between {1-3}
DEG = 3

# creates candidate alphas
ALPHA_LIST = np.logspace(-3, 3, 13)

# get value for RMSE MAE R^2
def eval_metrics(y_true, y_pred):
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae  = mean_absolute_error(y_true, y_pred)
    r2   = r2_score(y_true, y_pred)
    return rmse, mae, r2

# get MRE value(ignor the 0 value for y hat)
def eval_mre(y_true, y_pred):
    """Mean Relative Error (ignore zero targets)."""
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    mask = y_true != 0
    if mask.sum() == 0:
        return np.nan
    mre = np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask]))
    return mre

# Shapiro-Wilk test(P_value >0.5 mean the residuals are approximately normal)
def print_shapiro(residuals, name):
    """Shapiro-Wilk normality test for residuals."""
    w_stat, p_val = stats.shapiro(residuals)
    print(f"Shapiro-Wilk test for {name} residuals:")
    print(f"W-statistic = {w_stat:.4f}, p-value = {p_val:.4e}")
    if p_val > 0.05:
        print("Fail to reject H0: residuals may be normal.")
    else:
        print("Reject H0: residuals are not normal.")
    print("-" * 30)

# print out RMSE MAE MRE R^2
def print_metrics(y_true, y_pred, name):
    rmse, mae, r2 = eval_metrics(y_true, y_pred)
    mre = eval_mre(y_true, y_pred)
    print(f"--- {name} ---")
    print(f"RMSE: {rmse:.4f}")
    print(f"MAE : {mae:.4f}")
    print(f"MRE : {mre:.4f}")
    print(f"R^2 : {r2:.4f}")
    print("-" * 30)
    return rmse, mae, r2, mre

# load train val test from CFG_LIST
def load_data_from_cfg(cfg):
    df_train = pd.read_csv(cfg["train_path"])
    df_val   = pd.read_csv(cfg["val_path"])
    df_test  = pd.read_csv(cfg["test_path"])
    return df_train, df_val, df_test

def prepare_features(df_train, df_val, df_test, degree):
    """Standardise features and create polynomial features."""
    # select feature columns (exclude target)
    feature_cols = [c for c in df_train.columns if c != TARGET_COL]

    # X is features, y is target
    X_train = df_train[feature_cols].values
    y_train = df_train[TARGET_COL].values

    X_val = df_val[feature_cols].values
    y_val = df_val[TARGET_COL].values

    X_test = df_test[feature_cols].values
    y_test = df_test[TARGET_COL].values

    # standardize: fit only on train
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_val_s   = scaler.transform(X_val)
    X_test_s  = scaler.transform(X_test)

    # polynomial expansion
    poly = PolynomialFeatures(degree=degree, include_bias=False)
    X_train_p = poly.fit_transform(X_train_s)
    X_val_p   = poly.transform(X_val_s)
    X_test_p  = poly.transform(X_test_s)

    return X_train_p, y_train, X_val_p, y_val, X_test_p, y_test

# plot two set data
def plot_diagnostics_combined(diag_list, degree):
    """
    Combined 3x6 diagnostics plots for two datasets.
    Left 3 columns: first dataset; right 3 columns: second dataset.
    """
    assert len(diag_list) == 2, "This function assumes exactly two datasets."

    fig, axes = plt.subplots(3, 6, figsize=(18, 10))

    for j, item in enumerate(diag_list):
        label = item["label"]
        y_train = item["y_train"]
        y_pred_train = item["y_pred_train"]
        y_val   = item["y_val"]
        y_pred_val   = item["y_pred_val"]
        y_test  = item["y_test"]
        y_pred_test  = item["y_pred_test"]

        res_train = y_train - y_pred_train
        res_val   = y_val   - y_pred_val
        res_test  = y_test  - y_pred_test

        col_offset = j * 3  # 0 for first dataset, 3 for second

        sets = [
            ("Train", y_train, y_pred_train, res_train),
            ("Validation", y_val, y_pred_val, res_val),
            ("Test", y_test, y_pred_test, res_test),
        ]

        for i, (name, y_true, y_pred, res) in enumerate(sets):
            # metrics for title
            rmse, mae, r2 = eval_metrics(y_true, y_pred)

            # Actual vs Predicted
            ax = axes[i, col_offset + 0]
            ax.scatter(y_true, y_pred, alpha=0.6, edgecolors="none")
            min_v = min(y_true.min(), y_pred.min())
            max_v = max(y_true.max(), y_pred.max())
            ax.plot([min_v, max_v], [min_v, max_v], "r--")
            ax.set_xlabel("Actual value")
            ax.set_ylabel("Predicted value")
            ax.set_title(
                f"{name} ({label})\nRMSE={rmse:.3f}, R^2={r2:.3f}",
                fontsize=9
            )

            # Residual vs Predicted
            ax = axes[i, col_offset + 1]
            ax.scatter(y_pred, res, alpha=0.6, edgecolors="none")
            ax.axhline(0, color="r", linestyle="--")
            ax.set_xlabel("Predicted value")
            ax.set_ylabel("Residual")
            ax.set_title(f"{name} ({label}) residual vs pred", fontsize=9)

            # Residual distribution + normal curve
            ax = axes[i, col_offset + 2]
            ax.hist(res, bins=30, density=True, alpha=0.7)
            mu = np.mean(res)
            sigma = np.std(res, ddof=0)
            if sigma > 0:
                x_lin = np.linspace(res.min(), res.max(), 200)
                pdf = stats.norm.pdf(x_lin, loc=mu, scale=sigma)
                ax.plot(x_lin, pdf, "r--")
            ax.set_xlabel("Residual")
            ax.set_ylabel("Density")
            ax.set_title(f"{name} ({label}) residual dist.", fontsize=9)

    # vertical separator line between the two datasets
    line = Line2D([0.5, 0.5], [0.05, 0.95],
                  transform=fig.transFigure,
                  color="black", linewidth=2)
    fig.add_artist(line)

    fig.suptitle(f"Diagnostics plots comparison (DEG={degree})", fontsize=16)
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.show()


if __name__ == "__main__":
    diag_list = []

    for cfg in CFG_LIST:
        label = cfg["label"]
        print("\n" + "-" * 30)
        print(f"Dataset = {label}, DEG = {DEG}")
        print("-" * 30)

        df_train, df_val, df_test = load_data_from_cfg(cfg)
        X_train_p, y_train, X_val_p, y_val, X_test_p, y_test = prepare_features(
            df_train, df_val, df_test, degree=DEG
        )

        best_alpha = None
        best_rmse_val = np.inf

        for alpha in ALPHA_LIST:
            model = Ridge(alpha=alpha, random_state=42)
            model.fit(X_train_p, y_train)

            y_pred_val_tmp = model.predict(X_val_p)
            rmse_v, _, _ = eval_metrics(y_val, y_pred_val_tmp)

            if rmse_v < best_rmse_val:
                best_rmse_val = rmse_v
                best_alpha = alpha

        print(f"Best alpha for {label}: {best_alpha:.4g}, val RMSE = {best_rmse_val:.4f}")

        best_model = Ridge(alpha=best_alpha, random_state=42)
        best_model.fit(X_train_p, y_train)

        y_pred_train = best_model.predict(X_train_p)
        y_pred_val   = best_model.predict(X_val_p)
        y_pred_test  = best_model.predict(X_test_p)

        # metrics
        _ = print_metrics(y_train, y_pred_train, "Train set")
        _ = print_metrics(y_val,   y_pred_val,   "Validation set")
        _ = print_metrics(y_test,  y_pred_test,  "Test set")

        # Shapiro-Wilk tests
        res_train = y_train - y_pred_train
        res_val   = y_val   - y_pred_val
        res_test  = y_test  - y_pred_test

        print_shapiro(res_train, "Train set")
        print_shapiro(res_val,   "Validation set")
        print_shapiro(res_test,  "Test set")

        # store for combined plots
        diag_list.append({
            "label": label,
            "y_train": y_train,
            "y_pred_train": y_pred_train,
            "y_val": y_val,
            "y_pred_val": y_pred_val,
            "y_test": y_test,
            "y_pred_test": y_pred_test,
        })

    # draw one big figure for both datasets
    plot_diagnostics_combined(diag_list, DEG)
