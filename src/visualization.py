import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import pandas as pd
from matplotlib.ticker import EngFormatter
from sklearn.metrics import PredictionErrorDisplay
from sklearn.utils.validation import check_is_fitted, check_consistent_length

RANDOM_STATE = 42

sns.set_theme(palette="bright")

PALETTE = "coolwarm"
SCATTER_ALPHA = 0.2

# Plots a horizontal bar chart of model coefficients for visual analysis
def plot_coefficients(df_coefs, title="Coefficients"):
    fig, ax = plt.subplots(figsize=(8, 6))
    df_coefs.plot.barh()
    plt.title(title)
    plt.axvline(x=0, color=".5")
    plt.xlabel("Coefficients")
    plt.gca().get_legend().remove()
    return fig

# Plots residual distribution, residuals vs predicted values, and actual vs predicted values to evaluate regression performance
def plot_residual(y_true, y_pred):
    residual = y_true - y_pred

    fig, axs = plt.subplots(1, 3, figsize=(12, 6))

    sns.histplot(residual, kde=True, ax=axs[0])

    error_display_01 = PredictionErrorDisplay.from_predictions(
        y_true=y_true, y_pred=y_pred, kind="residual_vs_predicted", ax=axs[1]
    )

    error_display_02 = PredictionErrorDisplay.from_predictions(
        y_true=y_true, y_pred=y_pred, kind="actual_vs_predicted", ax=axs[2]
    )

    return fig

def plot_residual_estimator(estimator, X, y, eng_formatter=False, fracao_amostra=0.25):
    # 0) garante que o modelo está treinado
    check_is_fitted(estimator)

    # 1) normaliza tipos e limpa NaN simultaneamente
    X_df = pd.DataFrame(X).reset_index(drop=True)

    if isinstance(y, pd.DataFrame):
        y_sr = y.iloc[:, 0].reset_index(drop=True)
    else:
        y_sr = pd.Series(np.ravel(y)).reset_index(drop=True)

    mask = y_sr.notna() & ~X_df.isna().any(axis=1)
    X_clean = X_df.loc[mask].reset_index(drop=True)
    y_true  = y_sr.loc[mask].reset_index(drop=True)

    # 2) predição + garantir 1D
    y_pred = estimator.predict(X_clean)
    y_pred = np.asarray(y_pred)
    if y_pred.ndim == 2:
        if y_pred.shape[1] == 1:
            y_pred = y_pred.reshape(-1)
        else:
            raise ValueError(
                f"y_pred é multioutput com shape {y_pred.shape}. "
                f"Selecione uma saída (ex.: y_pred[:,0]) antes de plotar."
            )
    else:
        y_pred = y_pred.reshape(-1)

    y_true = np.asarray(y_true).reshape(-1)

    # 3) figura e plots
    fig, axs = plt.subplots(1, 3, figsize=(12, 6))

    PredictionErrorDisplay.from_predictions(
        y_true=y_true,
        y_pred=y_pred,
        kind="residual_vs_predicted",
        ax=axs[1],
        subsample=fracao_amostra,
        random_state=42,
        scatter_kwargs={"alpha": 0.2},
    )

    PredictionErrorDisplay.from_predictions(
        y_true=y_true,
        y_pred=y_pred,
        kind="actual_vs_predicted",
        ax=axs[2],
        subsample=fracao_amostra,
        random_state=42,
        scatter_kwargs={"alpha": 0.2},
    )

    residual = y_true - y_pred
    try:
        import seaborn as sns
        sns.histplot(residual, kde=True, ax=axs[0])
    except Exception:
        axs[0].hist(residual, bins=30)
        axs[0].set_title("Residuals distribution")

    if eng_formatter:
        for ax in axs:
            ax.yaxis.set_major_formatter(EngFormatter())
            ax.xaxis.set_major_formatter(EngFormatter())

    return fig


# Plots a boxplot comparison of metrics (time, R², MAE, RMSE) across different regression models
def plot_model_metrics_comparison(df_results):
    fig, axs = plt.subplots(2, 2, figsize=(8, 8), sharex=True)

    metric_comparison = [
        "time_seconds",
        "test_r2",
        "test_neg_mean_absolute_error",
        "test_neg_root_mean_squared_error",
    ]

    metric_names = [
        "Tempo (s)",
        "R²",
        "MAE",
        "RMSE",
    ]

    for ax, metric, name in zip(axs.flatten(), metric_comparison, metric_names):
        sns.boxplot(
            x="model",
            y=metric,
            data=df_results,
            ax=ax,
            showmeans=True,
        )
        ax.set_title(name)
        ax.set_ylabel(name)
        ax.tick_params(axis="x", rotation=90)

    return fig
