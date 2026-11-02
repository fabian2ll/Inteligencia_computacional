# %% [markdown]
# # 04 - Modelado y evaluación (fases 4 y 5 de CRISP-DM)
# Entrena y compara: línea base (promedio de 3 años), Poisson, binomial negativa, Random Forest,
# XGBoost y LSTM. Todos usan la misma regla: predecir un año con los valores REALES de los años previos.
# - Ajuste de hiperparámetros: validación cruzada por años (ventana expansiva) SOLO dentro de 2008-2021.
# - Prueba: 2022-2025, se evalúa una única vez con los mejores hiperparámetros.
# Requiere: pip install scikit-learn xgboost statsmodels torch joblib
# Ejecuta cada celda (# %%) con Shift+Enter, o todo el archivo con el botón Run.

# %%
import json
import warnings
from pathlib import Path

import joblib
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
RES = BASE / "resultados"
RES.mkdir(exist_ok=True)

FOLDS = [2018, 2019, 2020, 2021]   # año de validación de cada pliegue; se entrena con los años anteriores

# %% [markdown]
# ## 1. Datos y funciones de apoyo

# %%
t = pd.read_csv(PROC / "tabla_modelo.csv")
d = t[t["conjunto"] != "sin_rezagos"].reset_index(drop=True)
idx_train = d.index[d["conjunto"] == "train"]
idx_test = d.index[d["conjunto"] == "test"]
y_all = d["muertes"].values.astype(float)
COD = sorted(d["cod"].unique())
FT = ["lag1", "lag2", "lag3", "mm3", "anio_idx", "atipico_covid", "sexo_masc", "cod"]   # modelos de árboles
CONT = ["l1", "l2", "l3", "anio_idx"]                                                    # continuas de los GLM
print(f"Filas del modelo: {len(d)} | entrenamiento {len(idx_train)} | prueba {len(idx_test)}")


def mae(y, p):
    return float(np.mean(np.abs(np.asarray(y) - np.asarray(p))))


def rmse(y, p):
    return float(np.sqrt(np.mean((np.asarray(y) - np.asarray(p)) ** 2)))


def pliegues():
    for y in FOLDS:
        tr = idx_train[d.loc[idx_train, "anio"] < y]
        va = idx_train[d.loc[idx_train, "anio"] == y]
        yield y, tr, va


cv_filas = []


def busca(nombre, rejilla, ajusta_predice):
    """Evalúa cada combinación de hiperparámetros con la validación por años y devuelve la de menor MAE."""
    mejor, mejor_mae = None, np.inf
    for p in rejilla:
        maes = [mae(y_all[va], ajusta_predice(tr, va, p)) for _, tr, va in pliegues()]
        m = float(np.mean(maes))
        cv_filas.append({"modelo": nombre, "params": json.dumps(p, default=str), "mae_cv": m,
                         **{f"mae_{y}": v for y, v in zip(FOLDS, maes)}})
        if m < mejor_mae:
            mejor, mejor_mae = p, m
    print(f"  {nombre:18s} mejor MAE en validación cruzada: {mejor_mae:.3f} | {mejor}")
    return mejor


# %% [markdown]
# ## 2. Línea base y modelos lineales (Poisson y binomial negativa)
# Los GLM usan log(1 + rezagos) como predictores (mm3 se omite porque es combinación exacta de los rezagos)
# y una variable indicadora por tipo de cáncer. Poisson lleva penalización L2 (alpha) y la binomial negativa
# fija su parámetro de dispersión (disp) por validación cruzada.

# %%
def x_glm(df):
    X = pd.DataFrame({"l1": np.log1p(df["lag1"]), "l2": np.log1p(df["lag2"]), "l3": np.log1p(df["lag3"]),
                      "anio_idx": df["anio_idx"], "atipico_covid": df["atipico_covid"],
                      "sexo_masc": df["sexo_masc"]}, index=df.index)
    for c in COD[1:]:
        X[f"cod_{c}"] = (df["cod"] == c).astype(float)
    return X.astype(float)


def glm_ajusta(familia, df, p):
    X = x_glm(df)
    sc = StandardScaler().fit(X[CONT])
    Xs = X.copy()
    Xs[CONT] = sc.transform(X[CONT])
    y = df["muertes"].values.astype(float)
    if familia == "poisson":
        m = PoissonRegressor(alpha=p["alpha"], max_iter=3000).fit(Xs, y)
    else:
        m = sm.GLM(y, sm.add_constant(Xs, has_constant="add"),
                   family=sm.families.NegativeBinomial(alpha=p["disp"])).fit()
    return {"m": m, "sc": sc, "fam": familia, "cols": list(Xs.columns)}


def glm_predice(mod, df):
    X = x_glm(df)
    Xs = X.copy()
    Xs[CONT] = mod["sc"].transform(X[CONT])
    if mod["fam"] == "poisson":
        return mod["m"].predict(Xs)
    return np.asarray(mod["m"].predict(sm.add_constant(Xs, has_constant="add")))


def fp_glm(familia):
    return lambda tr, va, p: glm_predice(glm_ajusta(familia, d.loc[tr], p), d.loc[va])


print("Ajuste de hiperparámetros:")
mejores = {}
mejores["Poisson"] = busca("Poisson", ParameterGrid({"alpha": [1e-4, 1e-3, 1e-2, 1e-1, 1.0]}), fp_glm("poisson"))
mejores["Binomial negativa"] = busca("Binomial negativa",
                                     ParameterGrid({"disp": [0.01, 0.05, 0.1, 0.25, 0.5, 1.0]}), fp_glm("nb"))

# %% [markdown]
# ## 3. Random Forest y XGBoost

# %%
def fp_arbol(clase, fijos):
    def f(tr, va, p):
        m = clase(**fijos, **p).fit(d.loc[tr, FT], y_all[tr])
        return m.predict(d.loc[va, FT])
    return f


RF_FIJOS = {"n_estimators": 300, "random_state": SEED, "n_jobs": -1}
XGB_FIJOS = {"objective": "count:poisson", "tree_method": "hist", "random_state": SEED, "n_jobs": 1}
rej_rf = ParameterGrid({"max_depth": [3, 5, None], "min_samples_leaf": [1, 3, 5], "max_features": [0.5, 1.0]})
rej_xgb = ParameterSampler({"n_estimators": [100, 200, 400], "max_depth": [2, 3, 4], "learning_rate": [0.03, 0.1],
                            "subsample": [0.7, 1.0], "min_child_weight": [1, 3, 5]}, n_iter=25, random_state=SEED)
mejores["Random Forest"] = busca("Random Forest", rej_rf, fp_arbol(RandomForestRegressor, RF_FIJOS))
mejores["XGBoost"] = busca("XGBoost", list(rej_xgb), fp_arbol(XGBRegressor, XGB_FIJOS))

# %% [markdown]
# ## 4. LSTM
# Lee las ventanas de 3 años de lstm_secuencias.npz. La salida es log(lambda) y la pérdida es la verosimilitud
# de Poisson, así que la predicción es siempre no negativa. Se promedian varias semillas para estabilizar.
# Nota: el escalado (mu, sd) se calculó con años <= 2021, por lo que en los pliegues de validación
# hay una fuga mínima solo en el escalado; la prueba 2022-2025 no se ve afectada.

# %%
npz = np.load(PROC / "lstm_secuencias.npz")
clave = {(int(s), int(a)): i for i, (s, a) in enumerate(zip(npz["serie_id"], npz["anio"]))}
pos = np.array([clave[(int(s), int(a))] for s, a in zip(d["serie_id"], d["anio"])])
XS = torch.tensor(npz["X_seq"][pos])
XST = torch.tensor(npz["X_static"][pos])
N_SERIES = XST.shape[1]


class Red(nn.Module):
    def __init__(self, hidden, dropout=0.1):
        super().__init__()
        self.lstm = nn.LSTM(2, hidden, batch_first=True)
        self.drop = nn.Dropout(dropout)
        self.out = nn.Linear(hidden + N_SERIES, 1)

    def forward(self, xs, xst):
        _, (h, _) = self.lstm(xs)
        return self.out(torch.cat([self.drop(h[-1]), xst], dim=1)).squeeze(1)   # log(lambda)


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


def fp_lstm(n_semillas):
    return lambda tr, va, p: lstm_predice([lstm_entrena(tr, p, SEED + s) for s in range(n_semillas)], va)


rej_lstm = ParameterGrid({"hidden": [8, 16], "wd": [1e-4, 1e-2], "epocas": [100, 300]})
mejores["LSTM"] = busca("LSTM", rej_lstm, fp_lstm(2))

# %% [markdown]
# ## 5. Evaluación en la prueba 2022-2025
# Se reentrena cada modelo con todo el entrenamiento (2008-2021) y sus mejores hiperparámetros.

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
filas = []
for nombre, p in preds.items():
    filas.append({"modelo": nombre, "MAE": mae(y_te, p), "RMSE": rmse(y_te, p),
                  "mejora_MAE_vs_base_%": (1 - mae(y_te, p) / base_mae) * 100})
metricas = pd.DataFrame(filas).sort_values("MAE").reset_index(drop=True)
print("\nResultados en prueba 2022-2025 (ordenado por MAE):")
print(metricas.round(3).to_string(index=False))

ganador = metricas.iloc[0]["modelo"]
if ganador.startswith("Línea base"):
    print("\nAviso: ningún modelo supera la línea base en MAE; esto es un hallazgo válido y debe reportarse.")
else:
    print(f"\nModelo con menor MAE: {ganador}")

# MAE por año de prueba (estabilidad del desempeño)
por_anio = pd.DataFrame({n: pd.Series(np.abs(y_te - p)).groupby(d.loc[idx_test, "anio"].values).mean()
                         for n, p in preds.items()})
print("\nMAE por año de prueba:")
print(por_anio.round(3).to_string())

# %% [markdown]
# ## 6. Guardar resultados

# %%
largo = []
for nombre, p in preds.items():
    sub = d.loc[idx_test, ["serie_id", "cod", "tipo", "sexo", "anio", "muertes"]].copy()
    sub["modelo"], sub["prediccion"] = nombre, p
    largo.append(sub)
pd.concat(largo).to_csv(RES / "predicciones_test.csv", index=False)
metricas.to_csv(RES / "metricas_modelos.csv", index=False)
pd.DataFrame(cv_filas).to_csv(RES / "cv_resultados.csv", index=False)
with open(RES / "mejores_hiperparametros.json", "w", encoding="utf-8") as f:
    json.dump(mejores, f, indent=1, default=lambda o: o.item() if hasattr(o, "item") else str(o))
joblib.dump(finales, RES / "modelos.joblib")
torch.save([r.state_dict() for r in redes], RES / "lstm_pesos.pt")
print("\nGuardado en:", RES)
print("Siguiente: 05_importancia_variables.py")