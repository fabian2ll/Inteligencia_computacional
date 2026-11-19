# Evaluación fuera de muestra de 2025 con Poisson y LSTM.
# Entrena con 2008-2024 y compara 2025 real contra 2025 predicho.
import json
import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.linear_model import PoissonRegressor
from sklearn.model_selection import ParameterGrid
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")

SEED = 42
np.random.seed(SEED)
torch.manual_seed(SEED)

BASE = Path(__file__).resolve().parent
PROC = BASE / "data" / "processed"
OUT = BASE / "resultados" / "2025_holdout"
OUT.mkdir(parents=True, exist_ok=True)

TRAIN_MIN = 2008
TRAIN_MAX = 2024
TEST_YEAR = 2025
LAGS = (1, 2, 3)
FOLDS = [2018, 2019, 2020, 2021, 2022, 2023, 2024]


def mae(y_true, y_pred):
    return float(np.mean(np.abs(np.asarray(y_true) - np.asarray(y_pred))))


def rmse(y_true, y_pred):
    return float(np.sqrt(np.mean((np.asarray(y_true) - np.asarray(y_pred)) ** 2)))


def precision_global(y_true, y_pred):
    return float(100 * (1 - np.sum(np.abs(np.asarray(y_true) - np.asarray(y_pred)))
                        / max(np.sum(np.asarray(y_true)), 1)))


MORTALIDAD = pd.read_csv(PROC / "mortalidad_tipo_sexo_anio.csv")
years = list(range(2005, 2026))
series = (
    MORTALIDAD[["cod", "sexo"]]
    .drop_duplicates()
    .sort_values(["cod", "sexo"])
    .reset_index(drop=True)
)
series["serie_id"] = range(len(series))

panel = (
    series.merge(pd.DataFrame({"anio": years}), how="cross")
    .merge(MORTALIDAD[["cod", "sexo", "anio", "muertes"]],
           on=["cod", "sexo", "anio"], how="left")
)
panel["muertes"] = panel["muertes"].fillna(0).astype(int)
panel = panel.sort_values(["serie_id", "anio"]).reset_index(drop=True)

grouped = panel.groupby("serie_id")["muertes"]
for lag in LAGS:
    panel[f"lag{lag}"] = grouped.shift(lag)
panel["mm3"] = panel[[f"lag{lag}" for lag in LAGS]].mean(axis=1)
panel["anio_idx"] = panel["anio"] - 2005
panel["atipico_covid"] = panel["anio"].between(2020, 2022).astype(int)
panel["sexo_masc"] = (panel["sexo"] == "Masculino").astype(int)
tipo_nombre = {
    201: "Estómago",
    202: "Colon y recto",
    203: "Otros digestivos",
    204: "Tráquea, bronquios y pulmón",
    205: "Otros respiratorios",
    206: "Mama (mujer)",
    207: "Cuello uterino",
    208: "Cuerpo del útero",
    209: "Útero no especificado",
    210: "Próstata",
    211: "Otros genitourinarios",
    212: "Leucemia",
    213: "Linfático y hematopoyético",
    214: "Otras localizaciones",
    215: "In situ / benignos / incierto",
}
panel["tipo"] = panel["cod"].map(tipo_nombre)

tabla = panel[
    (panel["anio"].between(TRAIN_MIN, TEST_YEAR))
    & panel["lag3"].notna()
].copy()

train = tabla[tabla["anio"] <= TRAIN_MAX].copy()
test = tabla[tabla["anio"] == TEST_YEAR].copy()
assert len(test) == len(series)
assert test["anio"].unique().tolist() == [TEST_YEAR]

series = tabla[["serie_id", "cod", "tipo", "sexo"]].drop_duplicates().sort_values("serie_id")
n_series = len(series)
cods = sorted(tabla["cod"].unique())
y_train = train["muertes"].to_numpy(dtype=float)


def features_glm(df):
    x = pd.DataFrame({
        "l1": np.log1p(df["lag1"]),
        "l2": np.log1p(df["lag2"]),
        "l3": np.log1p(df["lag3"]),
        "anio_idx": df["anio_idx"],
        "atipico_covid": df["atipico_covid"],
        "sexo_masc": df["sexo_masc"],
    }, index=df.index)
    for cod in cods[1:]:
        x[f"cod_{cod}"] = (df["cod"] == cod).astype(float)
    return x.astype(float)


continuous = ["l1", "l2", "l3", "anio_idx"]


def fit_poisson(df, alpha):
    x = features_glm(df)
    scaler = StandardScaler().fit(x[continuous])
    xs = x.copy()
    xs[continuous] = scaler.transform(x[continuous])
    model = PoissonRegressor(alpha=alpha, max_iter=3000).fit(xs, df["muertes"].to_numpy(float))
    return model, scaler, list(xs.columns)


def predict_poisson(model, scaler, columns, df):
    x = features_glm(df).reindex(columns=columns, fill_value=0)
    xs = x.copy()
    xs[continuous] = scaler.transform(x[continuous])
    return np.clip(model.predict(xs), 0, None)


cv_poisson = []
for alpha in [1e-4, 1e-3, 1e-2, 1e-1, 1.0]:
    fold_maes = []
    for year in FOLDS:
        fit_fold = train[train["anio"] < year]
        val_fold = train[train["anio"] == year]
        model, scaler, columns = fit_poisson(fit_fold, alpha)
        pred = predict_poisson(model, scaler, columns, val_fold)
        fold_maes.append(mae(val_fold["muertes"], pred))
    cv_poisson.append({"alpha": alpha, "mae_cv": float(np.mean(fold_maes)),
                       "mae_folds": fold_maes})

poisson_alpha = min(cv_poisson, key=lambda r: r["mae_cv"])["alpha"]
poisson_model, poisson_scaler, poisson_columns = fit_poisson(train, poisson_alpha)
poisson_pred = predict_poisson(poisson_model, poisson_scaler, poisson_columns, test)


# LSTM
historic = tabla.pivot(index="serie_id", columns="anio", values="muertes")

train_values = np.log1p(train["muertes"].to_numpy(dtype=float))
lstm_mean = float(train_values.mean())
lstm_std = float(train_values.std())


def scale_lstm(value):
    return (np.log1p(value) - lstm_mean) / lstm_std


model_rows = tabla.copy().sort_values(["serie_id", "anio"]).reset_index(drop=True)
X_seq = np.zeros((len(model_rows), 3, 2), dtype=np.float32)
for i, row in model_rows.iterrows():
    serie_id = int(row["serie_id"])
    year = int(row["anio"])
    for j, k in enumerate((3, 2, 1)):
        X_seq[i, j, 0] = scale_lstm(row[f"lag{k}"])
        prev_year = year - k
        X_seq[i, j, 1] = int(2020 <= prev_year <= 2022)

static = np.eye(n_series, dtype=np.float32)[model_rows["serie_id"].to_numpy()]
y_lstm = model_rows["muertes"].to_numpy(dtype=np.float32)
train_mask = model_rows["anio"] <= TRAIN_MAX
test_mask = model_rows["anio"] == TEST_YEAR

X_seq_tensor = torch.tensor(X_seq)
X_static_tensor = torch.tensor(static)
y_lstm_tensor = torch.tensor(y_lstm)


class LSTMRed(nn.Module):
    def __init__(self, hidden):
        super().__init__()
        self.lstm = nn.LSTM(2, hidden, batch_first=True)
        self.drop = nn.Dropout(0.1)
        self.out = nn.Linear(hidden + n_series, 1)

    def forward(self, seq, static_features):
        _, (hidden, _) = self.lstm(seq)
        combined = torch.cat([self.drop(hidden[-1]), static_features], dim=1)
        return self.out(combined).squeeze(1)


def train_lstm(indices, hidden, wd, epochs, seed):
    torch.manual_seed(seed)
    model = LSTMRed(hidden)
    with torch.no_grad():
        model.out.bias.fill_(float(np.log(y_lstm[indices].mean() + 1e-3)))
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=wd)
    loss_fn = nn.PoissonNLLLoss(log_input=True)
    model.train()
    for _ in range(epochs):
        optimizer.zero_grad()
        loss = loss_fn(model(X_seq_tensor[indices], X_static_tensor[indices]),
                       y_lstm_tensor[indices])
        loss.backward()
        optimizer.step()
    model.eval()
    return model


def predict_lstm(models, indices):
    with torch.no_grad():
        pred = [
            torch.exp(model(X_seq_tensor[indices], X_static_tensor[indices]).clamp(max=6)).numpy()
            for model in models
        ]
    return np.clip(np.mean(pred, axis=0), 0, None)


lstm_cv = []
grid = ParameterGrid({"hidden": [8, 16], "wd": [1e-4, 1e-2], "epocas": [100, 300]})
for params in grid:
    fold_maes = []
    for year in FOLDS:
        fit_idx = np.flatnonzero((model_rows["anio"] >= TRAIN_MIN) & (model_rows["anio"] < year))
        val_idx = np.flatnonzero(model_rows["anio"] == year)
        models = [
            train_lstm(fit_idx, params["hidden"], params["wd"], params["epocas"], SEED + seed)
            for seed in range(2)
        ]
        pred = predict_lstm(models, val_idx)
        fold_maes.append(mae(y_lstm[val_idx], pred))
    lstm_cv.append({
        "hidden": params["hidden"], "wd": params["wd"], "epocas": params["epocas"],
        "mae_cv": float(np.mean(fold_maes)), "mae_folds": fold_maes
    })

lstm_best = min(lstm_cv, key=lambda r: r["mae_cv"])
lstm_models = [
    train_lstm(
        np.flatnonzero(train_mask),
        lstm_best["hidden"], lstm_best["wd"], lstm_best["epocas"], SEED + seed
    )
    for seed in range(5)
]
lstm_test_idx = np.flatnonzero(test_mask)
lstm_pred = predict_lstm(lstm_models, lstm_test_idx)


predictions = pd.DataFrame({
    "serie_id": test["serie_id"].to_numpy(),
    "cod": test["cod"].to_numpy(),
    "tipo_cancer": test["tipo"].to_numpy(),
    "sexo": test["sexo"].to_numpy(),
    "anio": TEST_YEAR,
    "real": test["muertes"].to_numpy(dtype=int),
})
predictions["Poisson"] = poisson_pred
predictions["LSTM"] = lstm_pred

predictions.to_csv(OUT / "predicciones_2025_poisson_lstm.csv", index=False)

metrics_rows = []
for model_name in ("Poisson", "LSTM"):
    pred = predictions[model_name].to_numpy()
    real = predictions["real"].to_numpy()
    metrics_rows.append({
        "modelo": model_name,
        "anio_real": TEST_YEAR,
        "anio_entrenamiento_desde": TRAIN_MIN,
        "anio_entrenamiento_hasta": TRAIN_MAX,
        "observaciones_prueba": len(real),
        "total_real": int(real.sum()),
        "total_predicho": round(float(pred.sum()), 3),
        "MAE": mae(real, pred),
        "RMSE": rmse(real, pred),
        "precision_global_pct": precision_global(real, pred),
    })

metrics = pd.DataFrame(metrics_rows)
metrics.to_csv(OUT / "metricas_2025_poisson_lstm.csv", index=False)

pd.DataFrame(cv_poisson).to_csv(OUT / "cv_poisson.csv", index=False)
pd.DataFrame(lstm_cv).to_csv(OUT / "cv_lstm.csv", index=False)

summary = {
    "semilla": SEED,
    "periodo_entrenamiento": [TRAIN_MIN, TRAIN_MAX],
    "anio_prueba": TEST_YEAR,
    "numero_series": n_series,
    "observaciones_entrenamiento": len(train),
    "observaciones_prueba": len(test),
    "poisson_alpha": poisson_alpha,
    "lstm": lstm_best,
    "nota_precision": (
        "precision_global_pct = 100 * (1 - suma(|real - predicho|) / suma(real)). "
        "Puede ser negativa si el error supera el total real."
    ),
}
with open(OUT / "configuracion_2025.json", "w", encoding="utf-8") as file:
    json.dump(summary, file, ensure_ascii=False, indent=2)


# Gráficas individuales: real vs predicho por cáncer y sexo.
palette = {"Poisson": "#1976D2", "LSTM": "#E67E22"}
for model_name in ("Poisson", "LSTM"):
    data = predictions.copy()
    data["etiqueta"] = data["tipo_cancer"] + " - " + data["sexo"]
    data = data.sort_values(["tipo_cancer", "sexo"])

    fig, ax = plt.subplots(figsize=(14, 8))
    x = np.arange(len(data))
    ax.plot(x, data["real"], color="#263238", marker="o", linewidth=2.2,
            markersize=5.5, label="Real 2025")
    ax.plot(x, data[model_name], color=palette[model_name], marker="o",
            linewidth=2.2, markersize=5.5, label=f"Predicho {model_name}")
    ax.set_xticks(x)
    ax.set_xticklabels(data["etiqueta"], rotation=55, ha="right", fontsize=8)
    ax.set_ylabel("Muertes")
    ax.set_title(f"2025 real vs predicho - Modelo {model_name}")
    ax.grid(axis="y", alpha=0.25)
    ax.legend()

    row_metric = metrics[metrics["modelo"] == model_name].iloc[0]
    text = (
        f"Precisión global: {row_metric['precision_global_pct']:.2f}%\n"
        f"MAE: {row_metric['MAE']:.3f} | RMSE: {row_metric['RMSE']:.3f}\n"
        f"Real: {row_metric['total_real']} | Predicho: {row_metric['total_predicho']:.1f}"
    )
    ax.text(
        0.995, 0.98, text, transform=ax.transAxes,
        ha="right", va="top", fontsize=9,
        bbox=dict(boxstyle="round", facecolor="white", alpha=0.92)
    )
    fig.tight_layout()
    fig.savefig(OUT / f"real_vs_predicho_2025_{model_name.lower().replace(' ', '_')}.png", dpi=180)
    plt.close(fig)

# Comparación directa de los totales por sexo.
sex_comparison = predictions.groupby("sexo").agg(
    real=("real", "sum"), poisson=("Poisson", "sum"), lstm=("LSTM", "sum")
).reset_index()

fig, ax = plt.subplots(figsize=(9, 6))
x = np.arange(len(sex_comparison))
ax.plot(x, sex_comparison["real"], color="#263238", marker="o",
        linewidth=2.5, markersize=7, label="Real 2025")
ax.plot(x, sex_comparison["poisson"], color=palette["Poisson"], marker="o",
        linewidth=2.5, markersize=7, label="Poisson")
ax.plot(x, sex_comparison["lstm"], color=palette["LSTM"], marker="o",
        linewidth=2.5, markersize=7, label="LSTM")
ax.set_xticks(x, sex_comparison["sexo"])
ax.set_ylabel("Muertes totales")
ax.set_title("2025: total real y predicho por sexo")
ax.grid(axis="y", alpha=0.25)
ax.legend()
fig.tight_layout()
fig.savefig(OUT / "real_vs_predicho_2025_totales_sexo.png", dpi=180)
plt.close(fig)

print(metrics.to_string(index=False))
print("Guardado en:", OUT)
