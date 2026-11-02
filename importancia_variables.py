# %% [markdown]
# # 05 - Importancia de variables, error por serie y curva de aprendizaje (fase 5 de CRISP-DM)
# Usa lo que guardó 04_modelado.py en /resultados. Genera:
#   1) Coeficientes de Poisson como razones de tasa
#   2) Importancia de XGBoost (valores SHAP) y de Random Forest / XGBoost / LSTM por permutación
#   3) Error por tipo de cáncer y sexo: dónde el modelo ayuda y dónde no
#   4) Curva de aprendizaje: error en la prueba según cuántos años de entrenamiento se usan
# Requiere lo mismo que el 04 (scikit-learn, xgboost, statsmodels, torch, joblib) y matplotlib.
# Ejecuta cada celda (# %%) con Shift+Enter, o todo el archivo con el botón Run.

# %%
import json
import warnings
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.api as sm
import xgboost as xgb
from sklearn.ensemble import RandomForestRegressor
from sklearn.inspection import permutation_importance
from sklearn.linear_model import PoissonRegressor
from sklearn.preprocessing import StandardScaler
from xgboost import XGBRegressor

try:
    import torch
    import torch.nn as nn
    HAY_TORCH = True
except Exception:   # sin torch o instalación dañada: el script sigue sin el LSTM
    HAY_TORCH = False

warnings.filterwarnings("ignore")
SEED = 42
BASE = Path(__file__).resolve().parent
PROC = BASE / "data" / "processed"
RES = BASE / "resultados"
FIG = BASE / "figuras"
FIG.mkdir(exist_ok=True)
plt.rcParams.update({"figure.dpi": 110, "axes.spines.top": False, "axes.spines.right": False,
                     "axes.grid": True, "grid.alpha": 0.3, "font.size": 10})

SHORT = {201: "Estómago", 202: "Colon y recto", 203: "Otros digestivos", 204: "Tráquea, bronquios y pulmón",
         205: "Otros respiratorios", 206: "Mama (mujer)", 207: "Cuello uterino", 208: "Cuerpo del útero",
         209: "Útero no especificado", 210: "Próstata", 211: "Otros genitourinarios", 212: "Leucemia",
         213: "Linfático y hematopoyético", 214: "Otras localizaciones",
         215: "In situ / benignos / incierto"}


def guarda(nombre):
    plt.tight_layout()
    plt.savefig(FIG / nombre, bbox_inches="tight")
    plt.close()
    print("  figura guardada:", FIG / nombre)


# %% [markdown]
# ## 1. Datos, modelos entrenados y funciones de apoyo (mismas definiciones que el 04)

# %%
t = pd.read_csv(PROC / "tabla_modelo.csv")
d = t[t["conjunto"] != "sin_rezagos"].reset_index(drop=True)
idx_train = d.index[d["conjunto"] == "train"]
idx_test = d.index[d["conjunto"] == "test"]
y_all = d["muertes"].values.astype(float)
y_te = y_all[idx_test]
COD = sorted(d["cod"].unique())
FT = ["lag1", "lag2", "lag3", "mm3", "anio_idx", "atipico_covid", "sexo_masc", "cod"]
CONT = ["l1", "l2", "l3", "anio_idx"]
RF_FIJOS = {"n_estimators": 300, "random_state": SEED, "n_jobs": -1}
XGB_FIJOS = {"objective": "count:poisson", "tree_method": "hist", "random_state": SEED, "n_jobs": 1}

mejores = json.load(open(RES / "mejores_hiperparametros.json", encoding="utf-8"))
finales = joblib.load(RES / "modelos.joblib")


def mae(y, p):
    return float(np.mean(np.abs(np.asarray(y) - np.asarray(p))))


def rmse(y, p):
    return float(np.sqrt(np.mean((np.asarray(y) - np.asarray(p)) ** 2)))


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


# --- LSTM (misma red y entrenamiento que el 04); solo se define si PyTorch está disponible
usa_lstm = HAY_TORCH and bool(mejores.get("LSTM"))
if usa_lstm:
    npz = np.load(PROC / "lstm_secuencias.npz")
    clave = {(int(s), int(a)): i for i, (s, a) in enumerate(zip(npz["serie_id"], npz["anio"]))}
    pos = np.array([clave[(int(s), int(a))] for s, a in zip(d["serie_id"], d["anio"])])
    XS = torch.tensor(npz["X_seq"][pos])
    XST = torch.tensor(npz["X_static"][pos])

    class Red(nn.Module):
        def __init__(self, hidden, dropout=0.1):
            super().__init__()
            self.lstm = nn.LSTM(2, hidden, batch_first=True)
            self.drop = nn.Dropout(dropout)
            self.out = nn.Linear(hidden + XST.shape[1], 1)

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

    def lstm_predice(redes, xs, xst):
        with torch.no_grad():
            return np.mean([torch.exp(r(xs, xst).clamp(max=6)).numpy() for r in redes], axis=0)
else:
    print("Aviso: sin PyTorch o sin hiperparámetros del LSTM; se omite el LSTM en este script.")

# %% [markdown]
# ## 2. Poisson: coeficientes como razones de tasa
# exp(coeficiente) indica por cuánto se multiplica el número ESPERADO de muertes. Las variables continuas
# están estandarizadas (efecto de +1 desviación estándar); las binarias, de 0 a 1; los tipos de cáncer se
# comparan contra el primero (Estómago). Los tres rezagos son muy parecidos entre sí, así que conviene
# interpretarlos en conjunto y no cada uno por separado.

# %%
ETQ = {"l1": "Muertes 1 año antes (log)", "l2": "Muertes 2 años antes (log)", "l3": "Muertes 3 años antes (log)",
       "anio_idx": "Tendencia (año)", "atipico_covid": "Año atípico 2020-2022", "sexo_masc": "Sexo masculino"}
for c in COD[1:]:
    ETQ[f"cod_{c}"] = f"Tipo: {SHORT.get(c, c)} (vs {SHORT.get(COD[0], COD[0])})"

pois = finales["Poisson"]
coef = pd.Series(pois["m"].coef_, index=pois["cols"])
tab_pois = pd.DataFrame({"variable": [ETQ[c] for c in coef.index], "coeficiente": coef.values,
                         "razon_de_tasa": np.exp(coef.values)})
tab_pois = tab_pois.reindex(tab_pois["coeficiente"].abs().sort_values(ascending=False).index).reset_index(drop=True)
tab_pois.to_csv(RES / "importancia_poisson.csv", index=False)
print("Poisson, coeficientes de mayor a menor magnitud:")
print(tab_pois.head(12).round(3).to_string(index=False))

# %% [markdown]
# ## 3. Importancia en XGBoost (SHAP) y por permutación (Random Forest, XGBoost, LSTM)
# SHAP: contribución media absoluta de cada variable a la predicción (en escala logarítmica, que es la
# escala interna de XGBoost con objetivo Poisson). Permutación: cuánto aumenta el MAE de prueba al
# desordenar una variable; si casi no cambia, el modelo no depende de ella.

# %%
Xte = d.loc[idx_test, FT]
contrib = finales["XGBoost"].get_booster().predict(xgb.DMatrix(Xte), pred_contribs=True)[:, :-1]
shap_imp = pd.Series(np.abs(contrib).mean(axis=0), index=FT).sort_values(ascending=False)


def permuta(modelo):
    r = permutation_importance(modelo, Xte, y_te, scoring="neg_mean_absolute_error", n_repeats=30,
                               random_state=SEED)
    return pd.Series(r.importances_mean, index=FT)


perm_rf = permuta(finales["Random Forest"])
perm_xgb = permuta(finales["XGBoost"])
imp = pd.DataFrame({"XGBoost_SHAP": shap_imp, "XGBoost_permutacion": perm_xgb, "RandomForest_permutacion": perm_rf})
imp = imp.sort_values("XGBoost_SHAP", ascending=False)

if usa_lstm:
    redes = []
    for sd in torch.load(RES / "lstm_pesos.pt"):
        r = Red(mejores["LSTM"]["hidden"])
        r.load_state_dict(sd)
        redes.append(r.eval())
    xs_te, xst_te = XS[idx_test], XST[idx_test]
    mae_ref = mae(y_te, lstm_predice(redes, xs_te, xst_te))
    rng = np.random.default_rng(SEED)
    grupos = {"Muertes de los 3 años previos": "seq0", "Indicador de año atípico": "seq1", "Serie (tipo x sexo)": "est"}
    perm_lstm = {}
    for nombre, g in grupos.items():
        aumentos = []
        for _ in range(20):
            p = rng.permutation(len(idx_test))
            xs2, xst2 = xs_te.clone(), xst_te.clone()
            if g == "seq0":
                xs2[:, :, 0] = xs_te[p, :, 0]
            elif g == "seq1":
                xs2[:, :, 1] = xs_te[p, :, 1]
            else:
                xst2 = xst_te[p]
            aumentos.append(mae(y_te, lstm_predice(redes, xs2, xst2)) - mae_ref)
        perm_lstm[nombre] = float(np.mean(aumentos))
    perm_lstm = pd.Series(perm_lstm).sort_values(ascending=False)
    perm_lstm.to_csv(RES / "importancia_lstm_permutacion.csv", header=["aumento_MAE"])
    print("\nLSTM, aumento del MAE al desordenar cada grupo de entradas:")
    print(perm_lstm.round(3).to_string())

imp.to_csv(RES / "importancia_arboles.csv")
print("\nImportancia en modelos de árboles (SHAP medio absoluto y aumento del MAE por permutación):")
print(imp.round(3).to_string())

# %% [markdown]
# ## 4. Error por tipo de cáncer y sexo (Poisson frente a la línea base)

# %%
pr = pd.read_csv(RES / "predicciones_test.csv")
pr["err"] = (pr["muertes"] - pr["prediccion"]).abs()
BASE_N = "Línea base (prom. 3 años)"
piv = pr.pivot_table(index=["tipo", "sexo"], columns="modelo", values="err", aggfunc="mean")
tab = pd.DataFrame({"muertes_prom_anual": pr[pr["modelo"] == "Poisson"].groupby(["tipo", "sexo"])["muertes"].mean(),
                    "MAE_base": piv[BASE_N], "MAE_Poisson": piv["Poisson"]})
tab["mejora_%"] = np.where(tab["MAE_base"] > 0, (1 - tab["MAE_Poisson"] / tab["MAE_base"]) * 100, np.nan)
tab = tab.sort_values("muertes_prom_anual", ascending=False)
tab.to_csv(RES / "error_por_serie.csv")
print("Error por serie en prueba 2022-2025 (ordenado por mortalidad promedio):")
print(tab.round(2).to_string())
print(f"\nPoisson mejora a la línea base en {(tab['MAE_Poisson'] < tab['MAE_base']).sum()} de {len(tab)} series")
por_sexo = pr.groupby(["modelo", "sexo"])["err"].mean().unstack().loc[[BASE_N, "Poisson"]]
print("\nMAE por sexo:")
print(por_sexo.round(3).to_string())

# %% [markdown]
# ## 5. Curva de aprendizaje
# Se entrena con los últimos k años antes de 2022 (k = 4, 6, ..., 14) y se mide el error en la prueba
# 2022-2025 con los mismos hiperparámetros del 04. Si las curvas aún bajan en k = 14, más años ayudarían;
# si ya están planas, el límite es el ruido propio de los conteos. La línea base no depende de k.

# %%
KS = [4, 6, 8, 10, 12, 14]
base_pred = d.loc[idx_test, "mm3"].values
filas = []
for k in KS:
    tr = idx_train[d.loc[idx_train, "anio"] >= 2022 - k]
    dtr, dte = d.loc[tr], d.loc[idx_test]
    p = {"Poisson": glm_predice(glm_ajusta("poisson", dtr, mejores["Poisson"]), dte),
         "Binomial negativa": glm_predice(glm_ajusta("nb", dtr, mejores["Binomial negativa"]), dte),
         "Random Forest": RandomForestRegressor(**RF_FIJOS, **mejores["Random Forest"]).fit(
             dtr[FT], y_all[tr]).predict(dte[FT]),
         "XGBoost": XGBRegressor(**XGB_FIJOS, **mejores["XGBoost"]).fit(dtr[FT], y_all[tr]).predict(dte[FT])}
    if usa_lstm:
        redes_k = [lstm_entrena(tr, mejores["LSTM"], SEED + s) for s in range(3)]
        p["LSTM"] = lstm_predice(redes_k, XS[idx_test], XST[idx_test])
    for nombre, pred in p.items():
        filas.append({"k_anios": k, "filas_entrenamiento": len(tr), "modelo": nombre,
                      "MAE": mae(y_te, pred), "RMSE": rmse(y_te, pred)})
    print(f"  k = {k:2d} años ({len(tr)} filas) listo")

curva = pd.DataFrame(filas)
curva.to_csv(RES / "curva_aprendizaje.csv", index=False)
print("\nMAE en la prueba según años de entrenamiento:")
print(curva.pivot(index="k_anios", columns="modelo", values="MAE").round(3).to_string())
print(f"Línea base: MAE {mae(y_te, base_pred):.3f} | RMSE {rmse(y_te, base_pred):.3f}")

fig, axs = plt.subplots(1, 2, figsize=(12, 4.5))
for ax, met, ref in [(axs[0], "MAE", mae(y_te, base_pred)), (axs[1], "RMSE", rmse(y_te, base_pred))]:
    for nombre, g in curva.groupby("modelo"):
        ax.plot(g["k_anios"], g[met], marker="o", lw=2 if nombre == "Poisson" else 1.3, label=nombre)
    ax.axhline(ref, color="gray", ls="--", lw=1.3, label="Línea base")
    ax.set(title=f"{met} en la prueba 2022-2025", xlabel="Años de entrenamiento", ylabel=met)
    ax.set_xticks(KS)
axs[0].legend(fontsize=8)
guarda("fig8_curva_aprendizaje.png")

# %% [markdown]
# ## 6. Figura de importancia de variables

# %%
ETQ_ARB = {"lag1": "Muertes 1 año antes", "lag2": "Muertes 2 años antes", "lag3": "Muertes 3 años antes",
           "mm3": "Promedio 3 años", "anio_idx": "Tendencia (año)", "atipico_covid": "Año atípico",
           "sexo_masc": "Sexo masculino", "cod": "Tipo de cáncer"}
fig, axs = plt.subplots(1, 3, figsize=(15, 4.8))
top = tab_pois.head(10).iloc[::-1]
axs[0].barh(top["variable"], top["coeficiente"], color=np.where(top["coeficiente"] >= 0, "#1971c2", "#d6336c"))
axs[0].set(title="Poisson: coeficientes (10 mayores)", xlabel="Coeficiente (escala log)")
s = imp["XGBoost_SHAP"].iloc[::-1]
axs[1].barh([ETQ_ARB[i] for i in s.index], s.values, color="#4263eb")
axs[1].set(title="XGBoost: SHAP medio absoluto", xlabel="Contribución media (escala log)")
r = imp["RandomForest_permutacion"].sort_values().copy()
axs[2].barh([ETQ_ARB[i] for i in r.index], r.values, color="#2f9e44")
axs[2].set(title="Random Forest: permutación", xlabel="Aumento del MAE al desordenar")
guarda("fig9_importancia_variables.png")

print("\nListo. Siguiente: 06_despliegue.py (exportar a Power BI)")