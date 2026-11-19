# %% [markdown]
# # 04b -Benchmark con microdatos: tipo × sexo × grupo de edad × año (2008-2025)
# Misma metodología que 04_modelado.py, pero con la dimensión de edad.
# Escribe en resultados/microdatos para no sobrescribir los resultados originales.

# %%
import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm
import torch
import torch.nn as nn
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import PoissonRegressor
from sklearn.model_selection import ParameterGrid, ParameterSampler
from sklearn.preprocessing import StandardScaler
from xgboost import XGBRegressor

warnings.filterwarnings("ignore")
SEED = 42
np.random.seed(SEED)
torch.manual_seed(SEED)
BASE = Path(__file__).resolve().parent
PROC = BASE / "data" / "processed"
RES = BASE / "resultados" / "microdatos"
RES.mkdir(parents=True, exist_ok=True)

YEARS = list(range(2008, 2026))
LAGS = (1, 2, 3)
TRAIN_FIN = 2021
ATIPICOS = (2020, 2022)
EDADES = ["0-14", "15-44", "45-64", "65+"]
FOLDS = [2018, 2019, 2020, 2021]
SHORT = {201: "Estómago", 202: "Colon y recto", 203: "Otros digestivos", 204: "Tráquea, bronquios y pulmón",
         205: "Otros respiratorios", 206: "Mama (mujer)", 207: "Cuello uterino", 208: "Cuerpo del útero",
         209: "Útero no especificado", 210: "Próstata", 211: "Otros genitourinarios", 212: "Leucemia",
         213: "Linfático y hematopoyético", 214: "Otras localizaciones",
         215: "In situ / benignos / incierto"}

# %% [markdown]
# ## 1. Panel y variables predictoras (solo información pasada)

# %%
m = pd.read_csv(PROC / "mortalidad_microdatos_edad.csv")
series = (m[["cod", "sexo", "grupo_edad"]].drop_duplicates()
            .sort_values(["cod", "sexo", "grupo_edad"]).reset_index(drop=True))
series["serie_id"] = range(len(series))
malla = series.merge(pd.DataFrame({"anio": YEARS}), how="cross")
panel = malla.merge(m[["cod", "sexo", "grupo_edad", "anio", "muertes"]],
                    on=["cod", "sexo", "grupo_edad", "anio"], how="left")
panel["muertes"] = panel["muertes"].fillna(0).astype(int)
panel = panel.sort_values(["serie_id", "anio"]).reset_index(drop=True)

g = panel.groupby("serie_id")["muertes"]
for k in LAGS:
    panel[f"lag{k}"] = g.shift(k)
panel["mm3"] = panel[[f"lag{k}" for k in LAGS]].mean(axis=1, skipna=False)
panel["anio_idx"] = panel["anio"] - YEARS[0]
panel["atipico_covid"] = panel["anio"].between(*ATIPICOS).astype(int)
panel["sexo_masc"] = (panel["sexo"] == "Masculino").astype(int)
panel["edad_idx"] = panel["grupo_edad"].map({e: i for i, e in enumerate(EDADES)})
panel["tipo"] = panel["cod"].map(SHORT)
panel["conjunto"] = np.select([panel["lag3"].isna(), panel["anio"] <= TRAIN_FIN],
                              ["sin_rezagos", "train"], default="test")

piv = panel.pivot(index="serie_id", columns="anio", values="muertes")
chk = panel[panel["conjunto"] != "sin_rezagos"]
for k in LAGS:
    esperado = np.array([piv.loc[s, a - k] for s, a in zip(chk["serie_id"], chk["anio"])])
    assert np.array_equal(chk[f"lag{k}"].values, esperado)
print("Series:", len(series), "| filas:", len(panel), "|", panel["conjunto"].value_counts().to_dict())
print("Sin fuga: OK")
print("Base en prueba: MAE", round(float((chk[chk.conjunto == "test"].muertes -
                                          chk[chk.conjunto == "test"].mm3).abs().mean()), 3))

cols = ["serie_id", "cod", "tipo", "sexo", "grupo_edad", "anio", "muertes", "lag1", "lag2", "lag3",
        "mm3", "anio_idx", "atipico_covid", "sexo_masc", "edad_idx", "conjunto"]
panel[cols].to_csv(PROC / "tabla_modelo_microdatos.csv", index=False)

d = panel[panel["conjunto"] != "sin_rezagos"].reset_index(drop=True)
idx_train = d.index[d["conjunto"] == "train"]
idx_test = d.index[d["conjunto"] == "test"]
y_all = d["muertes"].values.astype(float)
COD = sorted(d["cod"].unique())
FT = ["lag1", "lag2", "lag3", "mm3", "anio_idx", "atipico_covid", "sexo_masc", "cod", "edad_idx"]
CONT = ["l1", "l2", "l3", "anio_idx"]


def mae(y, p):
    return float(np.mean(np.abs(np.asarray(y) - np.asarray(p))))


def rmse(y, p):
    return float(np.sqrt(np.mean((np.asarray(y) - np.asarray(p)) ** 2)))


def pliegues():
    for y in FOLDS:
        yield y, idx_train[d.loc[idx_train, "anio"] < y], idx_train[d.loc[idx_train, "anio"] == y]


cv_filas = []


def busca(nombre, rejilla, ajusta_predice):
    mejor, mejor_mae = None, np.inf
    for p in rejilla:
        maes = [mae(y_all[va], ajusta_predice(tr, va, p)) for _, tr, va in pliegues()]
        m = float(np.mean(maes))
        cv_filas.append({"modelo": nombre, "params": json.dumps(p, default=str), "mae_cv": m,
                         **{f"mae_{y}": v for y, v in zip(FOLDS, maes)}})
        if m < mejor_mae:
            mejor, mejor_mae = p, m
    print(f"  {nombre:18s} mejor MAE CV: {mejor_mae:.3f} | {mejor}")
    return mejor


# %% [markdown]
# ## 2. Modelos

# %%
def x_glm(df):
    X = pd.DataFrame({"l1": np.log1p(df["lag1"]), "l2": np.log1p(df["lag2"]), "l3": np.log1p(df["lag3"]),
                      "anio_idx": df["anio_idx"], "atipico_covid": df["atipico_covid"],
                      "sexo_masc": df["sexo_masc"]}, index=df.index)
    for c in COD[1:]:
        X[f"cod_{c}"] = (df["cod"] == c).astype(float)
    for e in EDADES[1:]:
        X[f"edad_{e}"] = (df["grupo_edad"] == e).astype(float)
    return X.astype(float)


def glm_ajusta(familia, df, p):
    X = x_glm(df)
    sc = StandardScaler().fit(X[CONT])
    Xs = X.copy()
    Xs[CONT] = sc.transform(X[CONT])
    y = df["muertes"].values.astype(float)
    if familia == "poisson":
        mod = PoissonRegressor(alpha=p["alpha"], max_iter=3000).fit(Xs, y)
    else:
        mod = sm.GLM(y, sm.add_constant(Xs, has_constant="add"),
                     family=sm.families.NegativeBinomial(alpha=p["disp"])).fit()
    return {"m": mod, "sc": sc, "fam": familia, "cols": list(Xs.columns)}


def glm_predice(mod, df):
    X = x_glm(df)
    Xs = X.copy()
    Xs[CONT] = mod["sc"].transform(X[CONT])
    if mod["fam"] == "poisson":
        return mod["m"].predict(Xs)
    return np.asarray(mod["m"].predict(sm.add_constant(Xs, has_constant="add")))


fp_glm = lambda fam: lambda tr, va, p: glm_predice(glm_ajusta(fam, d.loc[tr], p), d.loc[va])


def fp_arbol(clase, fijos):
    def f(tr, va, p):
        return clase(**fijos, **p).fit(d.loc[tr, FT], y_all[tr]).predict(d.loc[va, FT])
    return f


RF_FIJOS = {"n_estimators": 300, "random_state": SEED, "n_jobs": -1}
XGB_FIJOS = {"objective": "count:poisson", "tree_method": "hist", "random_state": SEED, "n_jobs": 1}

print("Ajuste de hiperparámetros:")
mejores = {}
mejores["Poisson"] = busca("Poisson", ParameterGrid({"alpha": [1e-4, 1e-3, 1e-2, 1e-1, 1.0]}), fp_glm("poisson"))
mejores["Binomial negativa"] = busca("Binomial negativa",
                                     ParameterGrid({"disp": [0.01, 0.05, 0.1, 0.25, 0.5, 1.0]}), fp_glm("nb"))
mejores["Random Forest"] = busca("Random Forest",
                                 ParameterGrid({"max_depth": [3, 5, None], "min_samples_leaf": [1, 3, 5],
                                                "max_features": [0.5, 1.0]}),
                                 fp_arbol(RandomForestRegressor, RF_FIJOS))
mejores["XGBoost"] = busca("XGBoost", list(ParameterSampler(
    {"n_estimators": [100, 200, 400], "max_depth": [2, 3, 4], "learning_rate": [0.03, 0.1],
     "subsample": [0.7, 1.0], "min_child_weight": [1, 3, 5]}, n_iter=25, random_state=SEED)),
    fp_arbol(XGBRegressor, XGB_FIJOS))

# %%
W = len(LAGS)
previo = panel[panel["anio"] <= TRAIN_FIN]
mu = float(np.log1p(previo["muertes"]).mean())
sd = float(np.log1p(previo["muertes"]).std())


def escala(v):
    return (np.log1p(v) - mu) / sd


filas = panel[panel["conjunto"] != "sin_rezagos"].reset_index(drop=True)
X_seq = np.zeros((len(filas), W, 2), dtype="float32")
for i, (s, a) in enumerate(zip(filas["serie_id"], filas["anio"])):
    for j, k in enumerate(range(W, 0, -1)):
        X_seq[i, j, 0] = escala(piv.loc[s, a - k])
        X_seq[i, j, 1] = int(ATIPICOS[0] <= a - k <= ATIPICOS[1])
X_static = np.eye(len(series), dtype="float32")[filas["serie_id"].values]
np.savez_compressed(PROC / "lstm_secuencias_microdatos.npz", X_seq=X_seq, X_static=X_static,
                    y=filas["muertes"].values.astype("float32"), anio=filas["anio"].values,
                    serie_id=filas["serie_id"].values, conjunto=filas["conjunto"].values.astype(str), mu=mu, sd=sd)

XS = torch.tensor(X_seq)
XST = torch.tensor(X_static)
N_SERIES = XST.shape[1]


class Red(nn.Module):
    def __init__(self, hidden, dropout=0.1):
        super().__init__()
        self.lstm = nn.LSTM(2, hidden, batch_first=True)
        self.drop = nn.Dropout(dropout)
        self.out = nn.Linear(hidden + N_SERIES, 1)

    def forward(self, xs, xst):
        _, (h, _) = self.lstm(xs)
        return self.out(torch.cat([self.drop(h[-1]), xst], dim=1)).squeeze(1)


def lstm_entrena(tr, p, semilla):
    torch.manual_seed(semilla)
    red = Red(p["hidden"])
    with torch.no_grad():
        red.out.bias.fill_(float(np.log(y_all[tr].mean() + 1e-3)))
    opt = torch.optim.Adam(red.parameters(), lr=0.01, weight_decay=p["wd"])
    perdida = nn.PoissonNLLLoss(log_input=True)
    ytr = torch.tensor(y_all[tr], dtype=torch.float32)
    red.train()
    for _ in range(p["epocas"]):
        opt.zero_grad()
        perdida(red(XS[tr], XST[tr]), ytr).backward()
        opt.step()
    return red.eval()


def lstm_predice(redes, idx):
    with torch.no_grad():
        return np.mean([torch.exp(r(XS[idx], XST[idx]).clamp(max=6)).numpy() for r in redes], axis=0)


fp_lstm = lambda n: lambda tr, va, p: lstm_predice([lstm_entrena(tr, p, SEED + s) for s in range(n)], va)
mejores["LSTM"] = busca("LSTM", ParameterGrid(
    {"hidden": [8, 16], "wd": [1e-4, 1e-2], "epocas": [100, 300]}), fp_lstm(2))

# %% [markdown]
# ## 3. Evaluación final y comparación con el proyecto original

# %%
y_te = y_all[idx_test]
preds = {"Línea base (prom. 3 años)": d.loc[idx_test, "mm3"].values}
finales = {}
finales["Poisson"] = glm_ajusta("poisson", d.loc[idx_train], mejores["Poisson"])
preds["Poisson"] = glm_predice(finales["Poisson"], d.loc[idx_test])
finales["Binomial negativa"] = glm_ajusta("nb", d.loc[idx_train], mejores["Binomial negativa"])
preds["Binomial negativa"] = glm_predice(finales["Binomial negativa"], d.loc[idx_test])
finales["Random Forest"] = RandomForestRegressor(**RF_FIJOS, **mejores["Random Forest"]).fit(
    d.loc[idx_train, FT], y_all[idx_train])
preds["Random Forest"] = finales["Random Forest"].predict(d.loc[idx_test, FT])
finales["XGBoost"] = XGBRegressor(**XGB_FIJOS, **mejores["XGBoost"]).fit(d.loc[idx_train, FT], y_all[idx_train])
preds["XGBoost"] = finales["XGBoost"].predict(d.loc[idx_test, FT])
redes = [lstm_entrena(idx_train, mejores["LSTM"], SEED + s) for s in range(5)]
preds["LSTM"] = lstm_predice(redes, idx_test)

base_mae = mae(y_te, preds["Línea base (prom. 3 años)"])
metricas = pd.DataFrame([{"modelo": n, "MAE": mae(y_te, p), "RMSE": rmse(y_te, p),
                          "mejora_MAE_vs_base_%": (1 - mae(y_te, p) / base_mae) * 100}
                         for n, p in preds.items()]).sort_values("MAE").reset_index(drop=True)
original = pd.read_csv(BASE / "resultados" / "metricas_modelos.csv").set_index("modelo")["MAE"]
metricas["MAE_original_sin_edad"] = metricas["modelo"].map(original)
metricas["delta_MAE"] = metricas["MAE"] - metricas["MAE_original_sin_edad"]
print("\nPrueba 2022-2025, panel con edad:")
print(metricas.round(3).to_string(index=False))

largo = []
for nombre, p in preds.items():
    sub = d.loc[idx_test, ["serie_id", "cod", "tipo", "sexo", "grupo_edad", "anio", "muertes"]].copy()
    sub["modelo"], sub["prediccion"] = nombre, p
    largo.append(sub)
pd.concat(largo).to_csv(RES / "predicciones_test.csv", index=False)
metricas.to_csv(RES / "metricas_modelos.csv", index=False)
pd.DataFrame(cv_filas).to_csv(RES / "cv_resultados.csv", index=False)
with open(RES / "mejores_hiperparametros.json", "w", encoding="utf-8") as f:
    json.dump(mejores, f, indent=1, default=str)
print("\nGuardado en:", RES)
