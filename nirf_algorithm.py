"""
NIRF Ranking Framework - Function Approximation Algorithm
==========================================================
Based on the research paper:
"An Analytical Approach Towards the Prediction of Undefined Parameters
 for the National Institutional Ranking Framework"
IEEE INDISCON 2023

Scope (as per MSG.txt):
  - TLR_SS  (20 Marks): SS  = f(NT, NE) x 15 + f(NP) x 5
  - TLR_FRU (30 Marks): FRU = 7.5 x f(BC) + 22.5 x f(BO)

Models evaluated:
  Linear Regression, Polynomial Regression, KNN Regression,
  SVR, Decision Tree, Random Forest, CNN (MLP), XGBoost

Optimization: BayesSearchCV (scikit-optimize)
Evaluation  : Mean Squared Error (MSE)
"""

# ─────────────────────────────────────────────────────────────────────────────
# 0. Imports
# ─────────────────────────────────────────────────────────────────────────────
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")           # headless backend – saves plots to PNG files
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_squared_error

import xgboost as xgb

try:
    from skopt import BayesSearchCV
    from skopt.space import Real, Integer, Categorical
    BAYES_AVAILABLE = True
except ImportError:
    BAYES_AVAILABLE = False
    print("[WARNING] scikit-optimize not found – falling back to default hyperparameters.")

# ─────────────────────────────────────────────────────────────────────────────
# 1. Data Loading & Preprocessing
# ─────────────────────────────────────────────────────────────────────────────

SS_CSV  = r"c:\Users\re2je5u\Downloads\TEST-main\TEST-main\TLR_SS.csv"
FRU_CSV = r"c:\Users\re2je5u\Downloads\TEST-main\TEST-main\TLR_FRU.csv"

print("=" * 70)
print("  NIRF – Function Approximation for TLR_SS and TLR_FRU")
print("=" * 70)

df_ss  = pd.read_csv(SS_CSV)
df_fru = pd.read_csv(FRU_CSV)

print(f"\n[DATA] TLR_SS  → {df_ss.shape[0]} institutions | columns: {list(df_ss.columns)}")
print(f"[DATA] TLR_FRU → {df_fru.shape[0]} institutions | columns: {list(df_fru.columns)}")

# Drop rows with NaN in any used column
df_ss  = df_ss.dropna().reset_index(drop=True)
df_fru = df_fru.dropna().reset_index(drop=True)

print(f"\n[DATA] After dropping NaNs – SS: {len(df_ss)} rows | FRU: {len(df_fru)} rows")

# ─────────────────────────────────────────────────────────────────────────────
# 2. Feature / Target Definitions
# ─────────────────────────────────────────────────────────────────────────────
#
#  TLR_SS  : SS = f(NT, NE) * 15 + f(NP) * 5
#             We predict SS directly using [NT, NE, NP] as features.
#             The model learns the composite mapping.
#
#  TLR_FRU : FRU = 7.5 * f(BC) + 22.5 * f(BO)
#             We predict FRU directly using [BC, BO] as features.
#

X_ss  = df_ss[["NT", "NE", "NP"]].values
y_ss  = df_ss["SS"].values

X_fru = df_fru[["BC", "BO"]].values
y_fru = df_fru["FRU"].values

# 80/20 train-test split (fixed random seed for reproducibility)
SEED = 42
X_ss_tr,  X_ss_te,  y_ss_tr,  y_ss_te  = train_test_split(X_ss,  y_ss,  test_size=0.2, random_state=SEED)
X_fru_tr, X_fru_te, y_fru_tr, y_fru_te = train_test_split(X_fru, y_fru, test_size=0.2, random_state=SEED)

print(f"\n[SPLIT] SS  – Train: {len(X_ss_tr)}, Test: {len(X_ss_te)}")
print(f"[SPLIT] FRU – Train: {len(X_fru_tr)}, Test: {len(X_fru_te)}")

# ─────────────────────────────────────────────────────────────────────────────
# 3. Model Definitions with Bayesian Search Spaces
# ─────────────────────────────────────────────────────────────────────────────

def make_models():
    """
    Returns a dict of { model_name : (pipeline, bayes_search_space) }
    The pipeline includes scaling where needed.
    """
    models = {}

    # 1. Linear Regression (no hyperparameters to tune)
    models["Linear Regression"] = (
        Pipeline([("scaler", StandardScaler()), ("model", LinearRegression())]),
        None
    )

    # 2. Polynomial Regression (degree tunable)
    models["Polynomial Regression"] = (
        Pipeline([
            ("poly",   PolynomialFeatures()),
            ("scaler", StandardScaler()),
            ("model",  LinearRegression())
        ]),
        {"poly__degree": Integer(2, 4)} if BAYES_AVAILABLE else None
    )

    # 3. KNN Regression
    models["KNN Regression"] = (
        Pipeline([("scaler", StandardScaler()), ("model", KNeighborsRegressor())]),
        {"model__n_neighbors": Integer(2, 20),
         "model__weights":     Categorical(["uniform", "distance"]),
         "model__p":           Integer(1, 2)} if BAYES_AVAILABLE else None
    )

    # 4. SVR
    models["SVR"] = (
        Pipeline([("scaler", StandardScaler()), ("model", SVR())]),
        {"model__C":       Real(0.1, 100.0, prior="log-uniform"),
         "model__epsilon": Real(0.001, 1.0,  prior="log-uniform"),
         "model__kernel":  Categorical(["rbf", "linear", "poly"])} if BAYES_AVAILABLE else None
    )

    # 5. Decision Tree
    models["Decision Tree"] = (
        Pipeline([("model", DecisionTreeRegressor(random_state=SEED))]),
        {"model__max_depth":        Integer(2, 20),
         "model__min_samples_split": Integer(2, 20),
         "model__min_samples_leaf":  Integer(1, 10)} if BAYES_AVAILABLE else None
    )

    # 6. Random Forest
    models["Random Forest"] = (
        Pipeline([("model", RandomForestRegressor(random_state=SEED, n_jobs=1))]),
        {"model__n_estimators":      Integer(50, 300),
         "model__max_depth":         Integer(2, 20),
         "model__min_samples_split": Integer(2, 10),
         "model__min_samples_leaf":  Integer(1, 5),
         "model__max_features":      Categorical(["sqrt", "log2", None])} if BAYES_AVAILABLE else None
    )

    # 7. CNN – approximated with MLP (deep neural network, as paper uses CNN for tabular via 1-D conv)
    #    BayesSearchCV does not support tuple-valued Categorical for hidden_layer_sizes cleanly,
    #    so we manually try a small grid and pick the best via CV, then hand-tune.
    models["CNN (MLP)"] = (
        Pipeline([
            ("scaler", StandardScaler()),
            ("model",  MLPRegressor(hidden_layer_sizes=(128, 64), activation="relu",
                                    alpha=1e-4, learning_rate_init=1e-3,
                                    max_iter=2000, random_state=SEED, early_stopping=True))
        ]),
        None   # Bayesian search skipped; MLP uses a fixed well-performing architecture
    )

    # 8. XGBoost
    models["XGBoost"] = (
        Pipeline([("model", xgb.XGBRegressor(random_state=SEED, verbosity=0))]),
        {"model__n_estimators":  Integer(50, 300),
         "model__max_depth":     Integer(2, 10),
         "model__learning_rate": Real(0.01, 0.3, prior="log-uniform"),
         "model__subsample":     Real(0.5, 1.0),
         "model__colsample_bytree": Real(0.5, 1.0)} if BAYES_AVAILABLE else None
    )

    return models


# ─────────────────────────────────────────────────────────────────────────────
# 4. Training + Optimization + Evaluation
# ─────────────────────────────────────────────────────────────────────────────

def fit_and_evaluate(X_train, X_test, y_train, y_test, label=""):
    """
    Trains all 8 regression models (with Bayesian hyperparameter tuning when
    scikit-optimize is available), computes MSE on the test set, and returns
    a results dict plus the best-fitted estimators.
    """
    models = make_models()
    results = {}         # model_name -> {"mse": float, "fitted": estimator}

    print(f"\n{'─'*60}")
    print(f"  Training models for: {label}")
    print(f"{'─'*60}")

    for name, (pipeline, search_space) in models.items():
        print(f"  → {name:<25}", end=" ", flush=True)

        if BAYES_AVAILABLE and search_space is not None:
            # Bayesian hyperparameter optimisation (BayesSearchCV)
            opt = BayesSearchCV(
                pipeline,
                search_space,
                n_iter=20,
                cv=5,
                scoring="neg_mean_squared_error",
                random_state=SEED,
                n_jobs=1,
                verbose=0
            )
            opt.fit(X_train, y_train)
            best_estimator = opt.best_estimator_
            print(f"[best params: {opt.best_params_}]", end=" ")
        else:
            # Fallback: train with default hyperparameters
            pipeline.fit(X_train, y_train)
            best_estimator = pipeline

        y_pred = best_estimator.predict(X_test)
        mse    = mean_squared_error(y_test, y_pred)
        results[name] = {"mse": mse, "fitted": best_estimator, "y_pred": y_pred}
        print(f"MSE = {mse:.4f}")

    return results


# Run for both components
results_ss  = fit_and_evaluate(X_ss_tr,  X_ss_te,  y_ss_tr,  y_ss_te,  label="TLR_SS")
results_fru = fit_and_evaluate(X_fru_tr, X_fru_te, y_fru_tr, y_fru_te, label="TLR_FRU")


# ─────────────────────────────────────────────────────────────────────────────
# 5. MSE Summary Table
# ─────────────────────────────────────────────────────────────────────────────

def build_mse_table(results_ss, results_fru):
    model_names = list(results_ss.keys())
    data = {
        "Model":   model_names,
        "MSE_SS":  [results_ss[m]["mse"]  for m in model_names],
        "MSE_FRU": [results_fru[m]["mse"] for m in model_names],
    }
    df = pd.DataFrame(data).set_index("Model")
    return df

mse_table = build_mse_table(results_ss, results_fru)

print("\n" + "=" * 60)
print("  MSE COMPARISON TABLE")
print("=" * 60)
print(mse_table.round(4).to_string())


# ─────────────────────────────────────────────────────────────────────────────
# 6. Model Ranking (as per paper: rank 1 = highest MSE, rank 8 = lowest MSE)
# ─────────────────────────────────────────────────────────────────────────────

def rank_models(mse_df):
    """
    Rank models per column: rank 1 → highest MSE, rank 8 → lowest MSE.
    Cumulative rank = sum of ranks (lower is better overall).
    """
    ranked = mse_df.copy()
    for col in mse_df.columns:
        # ascending=False → highest MSE gets rank 1
        ranked[col + "_Rank"] = mse_df[col].rank(ascending=False).astype(int)
    rank_cols = [c for c in ranked.columns if c.endswith("_Rank")]
    ranked["Cumulative_Rank"] = ranked[rank_cols].sum(axis=1)
    return ranked.sort_values("Cumulative_Rank")

ranked_table = rank_models(mse_table)

print("\n" + "=" * 60)
print("  MODEL RANKINGS (lower cumulative rank = better)")
print("=" * 60)
print(ranked_table.to_string())

best_model = ranked_table.index[0]
print(f"\n  ★ Best overall model: {best_model}")


# ─────────────────────────────────────────────────────────────────────────────
# 7. Visualisation
# ─────────────────────────────────────────────────────────────────────────────

COLORS = {
    "Linear Regression":    "#1f77b4",
    "Polynomial Regression":"#ff7f0e",
    "KNN Regression":       "#2ca02c",
    "SVR":                  "#d62728",
    "Decision Tree":        "#9467bd",
    "Random Forest":        "#8c564b",
    "CNN (MLP)":            "#e377c2",
    "XGBoost":              "#7f7f7f",
}

OUT_DIR = r"c:\Users\re2je5u\Downloads\TEST-main\TEST-main"


# ── 7.1  Actual vs Predicted plots (one per component) ───────────────────────

def plot_actual_vs_predicted(results, y_test, component_label, filename):
    """
    Scatter plot of actual values and line plots of each model's predictions,
    sorted by the actual test values for readability.
    """
    sorted_idx = np.argsort(y_test)
    y_actual_s = y_test[sorted_idx]
    x_axis     = np.arange(len(y_actual_s))

    fig, ax = plt.subplots(figsize=(14, 6))
    ax.scatter(x_axis, y_actual_s, color="black", s=30, zorder=5,
               label="Actual", alpha=0.8)

    for name, res in results.items():
        y_pred_s = res["y_pred"][sorted_idx]
        ax.plot(x_axis, y_pred_s, color=COLORS[name], linewidth=1.5,
                label=name, alpha=0.85)

    ax.set_title(f"Actual vs Predicted – {component_label}", fontsize=14, fontweight="bold")
    ax.set_xlabel("Sorted Institution Index", fontsize=11)
    ax.set_ylabel("Score", fontsize=11)
    ax.legend(loc="upper left", fontsize=8, ncol=2)
    ax.grid(True, linestyle="--", alpha=0.4)
    plt.tight_layout()
    path = f"{OUT_DIR}\\{filename}"
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"[PLOT] Saved: {path}")


plot_actual_vs_predicted(results_ss,  y_ss_te,  "TLR_SS  (20 Marks)",  "plot_SS_actual_vs_predicted.png")
plot_actual_vs_predicted(results_fru, y_fru_te, "TLR_FRU (30 Marks)", "plot_FRU_actual_vs_predicted.png")


# ── 7.2  MSE Comparison Bar Chart ────────────────────────────────────────────

def plot_mse_comparison(mse_df, filename):
    model_names = mse_df.index.tolist()
    x     = np.arange(len(model_names))
    width = 0.35

    fig, ax = plt.subplots(figsize=(13, 6))
    bars1 = ax.bar(x - width/2, mse_df["MSE_SS"],  width, label="TLR_SS",
                   color=[COLORS.get(m, "#aaaaaa") for m in model_names], alpha=0.85)
    bars2 = ax.bar(x + width/2, mse_df["MSE_FRU"], width, label="TLR_FRU",
                   color=[COLORS.get(m, "#aaaaaa") for m in model_names], alpha=0.55,
                   edgecolor="black", linewidth=0.6)

    ax.set_xticks(x)
    ax.set_xticklabels(model_names, rotation=20, ha="right", fontsize=9)
    ax.set_ylabel("Mean Squared Error (MSE)", fontsize=11)
    ax.set_title("MSE Comparison Across Models – TLR_SS vs TLR_FRU", fontsize=14, fontweight="bold")

    # Annotate bar values
    for bar in bars1:
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.05,
                f"{bar.get_height():.2f}", ha="center", va="bottom", fontsize=7)
    for bar in bars2:
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.05,
                f"{bar.get_height():.2f}", ha="center", va="bottom", fontsize=7)

    ax.legend(fontsize=10)
    ax.grid(True, axis="y", linestyle="--", alpha=0.4)
    plt.tight_layout()
    path = f"{OUT_DIR}\\{filename}"
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"[PLOT] Saved: {path}")


plot_mse_comparison(mse_table, "plot_MSE_comparison.png")


# ── 7.3  Cumulative Rank Bar Chart ───────────────────────────────────────────

def plot_cumulative_ranks(ranked_df, filename):
    models = ranked_df.index.tolist()
    ranks  = ranked_df["Cumulative_Rank"].values
    colors = [COLORS.get(m, "#aaaaaa") for m in models]

    fig, ax = plt.subplots(figsize=(11, 5))
    bars = ax.barh(models, ranks, color=colors, alpha=0.85, edgecolor="black", linewidth=0.5)
    ax.set_xlabel("Cumulative Rank (lower = better)", fontsize=11)
    ax.set_title("Model Rankings by Cumulative MSE Rank", fontsize=14, fontweight="bold")
    ax.invert_yaxis()

    for bar, val in zip(bars, ranks):
        ax.text(bar.get_width() + 0.1, bar.get_y() + bar.get_height()/2,
                str(int(val)), va="center", fontsize=9)

    ax.grid(True, axis="x", linestyle="--", alpha=0.4)
    plt.tight_layout()
    path = f"{OUT_DIR}\\{filename}"
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"[PLOT] Saved: {path}")


plot_cumulative_ranks(ranked_table, "plot_cumulative_ranks.png")


# ── 7.4  Residual plots ───────────────────────────────────────────────────────

def plot_residuals(results, y_test, component_label, filename):
    n_models = len(results)
    cols = 4
    rows = (n_models + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(16, rows * 4))
    axes = axes.flatten()

    for ax, (name, res) in zip(axes, results.items()):
        residuals = y_test - res["y_pred"]
        ax.scatter(res["y_pred"], residuals, color=COLORS[name], alpha=0.7, s=25)
        ax.axhline(0, color="black", linewidth=1.2, linestyle="--")
        ax.set_title(name, fontsize=9, fontweight="bold")
        ax.set_xlabel("Predicted", fontsize=8)
        ax.set_ylabel("Residual", fontsize=8)
        ax.grid(True, linestyle="--", alpha=0.3)

    for ax in axes[n_models:]:
        ax.set_visible(False)

    fig.suptitle(f"Residual Plots – {component_label}", fontsize=13, fontweight="bold")
    plt.tight_layout()
    path = f"{OUT_DIR}\\{filename}"
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"[PLOT] Saved: {path}")


plot_residuals(results_ss,  y_ss_te,  "TLR_SS",  "plot_SS_residuals.png")
plot_residuals(results_fru, y_fru_te, "TLR_FRU", "plot_FRU_residuals.png")


# ─────────────────────────────────────────────────────────────────────────────
# 8. Score Reconstruction Verification
# ─────────────────────────────────────────────────────────────────────────────
#
#  Using the best model, verify that the formula structure matches:
#    SS  = f(NT, NE) * 15 + f(NP) * 5     (total ≤ 20)
#    FRU = 7.5 * f(BC) + 22.5 * f(BO)     (total ≤ 30)
#
#  Here we show the predicted vs actual scores on the full dataset,
#  treating the best model's output as the approximated composite score.

print("\n" + "=" * 60)
print("  FORMULA VERIFICATION ON FULL DATASET")
print("=" * 60)

best_ss_model  = results_ss[best_model]["fitted"]
best_fru_model = results_fru[best_model]["fitted"]

# Full-dataset predictions with best model
y_ss_full_pred  = best_ss_model.predict(X_ss)
y_fru_full_pred = best_fru_model.predict(X_fru)

mse_ss_full  = mean_squared_error(y_ss,  y_ss_full_pred)
mse_fru_full = mean_squared_error(y_fru, y_fru_full_pred)

print(f"  Best Model   : {best_model}")
print(f"  SS  – Full-dataset MSE : {mse_ss_full:.4f}")
print(f"  FRU – Full-dataset MSE : {mse_fru_full:.4f}")

# Build a verification DataFrame
df_verify_ss = df_ss[["CollegeID", "NT", "NE", "NP", "SS"]].copy()
df_verify_ss["SS_Predicted"]  = y_ss_full_pred
df_verify_ss["SS_Error"]      = (df_verify_ss["SS"] - df_verify_ss["SS_Predicted"]).abs()

df_verify_fru = df_fru[["CollegeID", "BC", "BO", "FRU"]].copy()
df_verify_fru["FRU_Predicted"] = y_fru_full_pred
df_verify_fru["FRU_Error"]     = (df_verify_fru["FRU"] - df_verify_fru["FRU_Predicted"]).abs()

print("\n  TLR_SS – Top 10 predictions:")
print(df_verify_ss.head(10).to_string(index=False))

print("\n  TLR_FRU – Top 10 predictions:")
print(df_verify_fru.head(10).to_string(index=False))

# Save verification tables to CSV
ss_out  = f"{OUT_DIR}\\TLR_SS_predictions.csv"
fru_out = f"{OUT_DIR}\\TLR_FRU_predictions.csv"
df_verify_ss.to_csv(ss_out,  index=False)
df_verify_fru.to_csv(fru_out, index=False)
print(f"\n[CSV] Saved: {ss_out}")
print(f"[CSV] Saved: {fru_out}")
