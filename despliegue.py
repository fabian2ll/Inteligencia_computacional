# %% [markdown]
# # 06 - Despliegue (fase 6 de CRISP-DM): exportar a Power BI
# Lee lo que dejaron los pasos 01-05 (data/processed y resultados) y genera, en la carpeta /powerbi:
#   - Tablas CSV listas para Power BI (hechos, predicciones, métricas, importancia, contexto)
#   - cargar_en_powerbi.py: script para "Obtener datos > Script de Python" (ya trae tu ruta)
#   - visual_real_vs_predicho.py: código de ejemplo para un visual de Python
# El modelo del tablero es el de menor MAE en la prueba (se puede forzar con MODELO_MANUAL).
# Además pronostica el año siguiente (2026) con ese modelo reentrenado con 2008-2025.
# Ejecuta cada celda (# %%) con Shift+Enter, o todo el archivo con el botón Run.

# %%
import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import PoissonRegressor
from sklearn.preprocessing import StandardScaler
from xgboost import XGBRegressor

warnings.filterwarnings("ignore")
SEED = 42
BASE = Path(__file__).resolve().parent
PROC = BASE / "data" / "processed"
RES = BASE / "resultados"
BI = BASE / "powerbi"
BI.mkdir(exist_ok=True)

MODELO_MANUAL = None      # por ejemplo "Poisson"; None = el de menor MAE en la prueba
ANIO_PRON = 2026          # año a pronosticar (un año adelante con valores reales)
BASE_N = "Línea base (prom. 3 años)"


def guarda(df, nombre):
    df.to_csv(BI / nombre, index=False, encoding="utf-8-sig")   # utf-8-sig: Power BI lee bien las tildes
    print(f"  {nombre:32s} {len(df):5d} filas")


# %% [markdown]
# ## 1. Modelo seleccionado

# %%
met = pd.read_csv(RES / "metricas_modelos.csv")
cand = met[met["modelo"] != BASE_N].sort_values("MAE")
MODELO = MODELO_MANUAL or cand.iloc[0]["modelo"]
mejores = json.load(open(RES / "mejores_hiperparametros.json", encoding="utf-8"))
print(f"Modelo del tablero: {MODELO} (MAE prueba {met.loc[met['modelo'] == MODELO, 'MAE'].iloc[0]:.3f}"
      f" vs línea base {met.loc[met['modelo'] == BASE_N, 'MAE'].iloc[0]:.3f})")

# %% [markdown]
# ## 2. Tabla de hechos: real y predicho por serie y año

# %%
tabla = pd.read_csv(PROC / "tabla_modelo.csv")
series = (tabla[["serie_id", "cod", "tipo", "sexo"]].drop_duplicates()
          .rename(columns={"tipo": "tipo_cancer"}).sort_values("serie_id").reset_index(drop=True))
pr = pd.read_csv(RES / "predicciones_test.csv")

hechos = tabla[["serie_id", "cod", "tipo", "sexo", "anio", "muertes", "conjunto"]].rename(
    columns={"tipo": "tipo_cancer"})
hechos["periodo"] = hechos["conjunto"].map({"sin_rezagos": "2005-2007 (solo rezagos)",
                                            "train": "entrenamiento", "test": "prueba"})
hechos = hechos.drop(columns="conjunto")
sel = pr[pr["modelo"] == MODELO][["serie_id", "anio", "prediccion"]]
bas = pr[pr["modelo"] == BASE_N][["serie_id", "anio", "prediccion"]].rename(columns={"prediccion": "prediccion_linea_base"})
hechos = hechos.merge(sel, on=["serie_id", "anio"], how="left").merge(bas, on=["serie_id", "anio"], how="left")

# Control de coherencia: el MAE recalculado desde la tabla debe coincidir con el de metricas_modelos.csv
en_prueba = hechos[hechos["periodo"] == "prueba"]
mae_chk = float((en_prueba["muertes"] - en_prueba["prediccion"]).abs().mean())
mae_ref = float(met.loc[met["modelo"] == MODELO, "MAE"].iloc[0])
assert abs(mae_chk - mae_ref) < 1e-3, f"MAE no coincide: {mae_chk:.4f} vs {mae_ref:.4f}"
print(f"Control OK: MAE de {MODELO} recalculado = {mae_chk:.3f}")

# %% [markdown]
# ## 3. Pronóstico del año siguiente
# Se reentrena el modelo seleccionado con todo lo disponible (2008-2025) y se predice ANIO_PRON usando
# como rezagos los valores reales de los tres años anteriores. Es la misma regla de la prueba.

# %%
COD = sorted(tabla["cod"].unique())
CONT = ["l1", "l2", "l3", "anio_idx"]
FT = ["lag1", "lag2", "lag3", "mm3", "anio_idx", "atipico_covid", "sexo_masc", "cod"]


def x_glm(df):
    X = pd.DataFrame({"l1": np.log1p(df["lag1"]), "l2": np.log1p(df["lag2"]), "l3": np.log1p(df["lag3"]),
                      "anio_idx": df["anio_idx"], "atipico_covid": df["atipico_covid"],
                      "sexo_masc": df["sexo_masc"]}, index=df.index)
    for c in COD[1:]:
        X[f"cod_{c}"] = (df["cod"] == c).astype(float)
    return X.astype(float)


def pronostica(modelo, p, d_fit, d_new):
    y = d_fit["muertes"].values.astype(float)
    if modelo in ("Poisson", "Binomial negativa"):
        X, Xn = x_glm(d_fit), x_glm(d_new)
        sc = StandardScaler().fit(X[CONT])
        X[CONT], Xn[CONT] = sc.transform(X[CONT]), sc.transform(Xn[CONT])
        if modelo == "Poisson":
            return PoissonRegressor(alpha=p["alpha"], max_iter=3000).fit(X, y).predict(Xn)
        import statsmodels.api as sm
        m = sm.GLM(y, sm.add_constant(X, has_constant="add"),
                   family=sm.families.NegativeBinomial(alpha=p["disp"])).fit()
        return np.asarray(m.predict(sm.add_constant(Xn, has_constant="add")))
    if modelo == "Random Forest":
        m = RandomForestRegressor(n_estimators=300, random_state=SEED, n_jobs=-1, **p)
    elif modelo == "XGBoost":
        m = XGBRegressor(objective="count:poisson", tree_method="hist", random_state=SEED, n_jobs=1, **p)
    else:
        return None      # LSTM: requiere PyTorch; no se pronostica desde este script
    return m.fit(d_fit[FT], y).predict(d_new[FT])


d_fit = tabla[tabla["conjunto"] != "sin_rezagos"].reset_index(drop=True)
piv = tabla.pivot(index="serie_id", columns="anio", values="muertes")
ult = int(tabla["anio"].max())
nuevo = series.rename(columns={"tipo_cancer": "tipo"}).copy()
for k in (1, 2, 3):
    nuevo[f"lag{k}"] = piv[ult - k + 1].reindex(nuevo["serie_id"]).values
nuevo["mm3"] = nuevo[["lag1", "lag2", "lag3"]].mean(axis=1)
nuevo["anio"] = ANIO_PRON
nuevo["anio_idx"] = ANIO_PRON - int(tabla["anio"].min())
nuevo["atipico_covid"] = 0
nuevo["sexo_masc"] = (nuevo["sexo"] == "Masculino").astype(int)

pron = pronostica(MODELO, mejores[MODELO], d_fit, nuevo)
if pron is None:
    print(f"Aviso: el pronóstico de {MODELO} no está soportado en este script; se omite.")
else:
    fila = nuevo[["serie_id", "cod", "tipo", "sexo", "anio", "mm3"]].rename(
        columns={"tipo": "tipo_cancer", "mm3": "prediccion_linea_base"})
    fila["muertes"], fila["prediccion"], fila["periodo"] = np.nan, pron, "pronóstico"
    hechos = pd.concat([hechos, fila], ignore_index=True)
    print(f"Pronóstico {ANIO_PRON} ({MODELO}): {pron.sum():.0f} muertes en total "
          f"(real {ult}: {int(piv[ult].sum())})")

hechos["modelo_seleccionado"] = MODELO
hechos["error_abs"] = (hechos["muertes"] - hechos["prediccion"]).abs()
hechos = hechos.sort_values(["serie_id", "anio"]).reset_index(drop=True)

# %% [markdown]
# ## 4. Exportar las tablas

# %%
print("\nTablas en", BI)
guarda(hechos[["serie_id", "tipo_cancer", "cod", "sexo", "anio", "periodo", "muertes", "prediccion",
               "prediccion_linea_base", "modelo_seleccionado", "error_abs"]].round(3), "hechos_mortalidad.csv")
guarda(series[["serie_id", "cod", "tipo_cancer", "sexo"]], "dim_serie.csv")

pm = pr.rename(columns={"tipo": "tipo_cancer"}).copy()
pm["error_abs"] = (pm["muertes"] - pm["prediccion"]).abs()
pm["es_seleccionado"] = pm["modelo"] == MODELO
guarda(pm.round(3), "predicciones_modelos.csv")
guarda(met.round(4), "metricas_modelos.csv")
guarda(pd.read_csv(RES / "error_por_serie.csv").round(3), "error_por_serie.csv")
guarda(pd.read_csv(RES / "curva_aprendizaje.csv").round(4), "curva_aprendizaje.csv")

# Importancia de variables en una sola tabla larga (modelo, método, variable, valor)
ETQ_ARB = {"lag1": "Muertes 1 año antes", "lag2": "Muertes 2 años antes", "lag3": "Muertes 3 años antes",
           "mm3": "Promedio 3 años", "anio_idx": "Tendencia (año)", "atipico_covid": "Año atípico",
           "sexo_masc": "Sexo masculino", "cod": "Tipo de cáncer"}
imp = []
p = pd.read_csv(RES / "importancia_poisson.csv")
imp += [{"modelo": "Poisson", "metodo": "Coeficiente (escala log)", "variable": r.variable, "valor": r.coeficiente}
        for r in p.itertuples()]
a = pd.read_csv(RES / "importancia_arboles.csv", index_col=0)
for col, (mod, met_) in {"XGBoost_SHAP": ("XGBoost", "SHAP medio absoluto"),
                         "XGBoost_permutacion": ("XGBoost", "Permutación (aumento del MAE)"),
                         "RandomForest_permutacion": ("Random Forest", "Permutación (aumento del MAE)")}.items():
    imp += [{"modelo": mod, "metodo": met_, "variable": ETQ_ARB.get(v, v), "valor": x} for v, x in a[col].items()]
if (RES / "importancia_lstm_permutacion.csv").exists():
    l = pd.read_csv(RES / "importancia_lstm_permutacion.csv", index_col=0)
    imp += [{"modelo": "LSTM", "metodo": "Permutación (aumento del MAE)", "variable": v, "valor": x}
            for v, x in l["aumento_MAE"].items()]
imp = pd.DataFrame(imp)
imp["valor_abs"] = imp["valor"].abs()
guarda(imp.round(4), "importancia_variables.csv")

# Contexto descriptivo (no son variables del modelo)
tam = pd.read_csv(PROC / "tamizacion_eps_anio.csv")
tam["cumplimiento_pct"] = tam["cumplimiento"] * (100 if tam["cumplimiento"].max() <= 1.5 else 1)
guarda(tam.round(2), "contexto_tamizacion.csv")
fac = pd.read_csv(PROC / "factores_riesgo_grupo_edad.csv")
fac["porcentaje_pct"] = pd.to_numeric(fac["porcentaje"], errors="coerce") * (100 if fac["porcentaje"].max() <= 1.5 else 1)
fac["grupo_edad_corto"] = fac["grupo_edad"].str.split(" \\(").str[0].str.title()
guarda(fac.round(2), "contexto_factores_riesgo.csv")
guarda(pd.read_csv(PROC / "mortalidad_grandes_causas.csv"), "contexto_grandes_causas.csv")

# %% [markdown]
# ## 5. Scripts de Python para Power BI

# %%
TABLAS = ["hechos_mortalidad", "dim_serie", "predicciones_modelos", "metricas_modelos", "error_por_serie",
          "curva_aprendizaje", "importancia_variables", "contexto_tamizacion", "contexto_factores_riesgo",
          "contexto_grandes_causas"]
carga = ['# Power BI: Inicio > Obtener datos > Script de Python > pega este código > Aceptar.',
         '# Marca todas las tablas en el Navegador y pulsa Cargar. Requiere pandas en el Python que usa Power BI.',
         'import pandas as pd', 'from pathlib import Path', '', f'RUTA = Path(r"{BI}")', '']
carga += [f'{t} = pd.read_csv(RUTA / "{t}.csv", encoding="utf-8-sig")' for t in TABLAS]
(BI / "cargar_en_powerbi.py").write_text("\n".join(carga) + "\n", encoding="utf-8")

visual = '''# Visual de Python en Power BI: real vs. predicho por año.
# En "Valores" del visual arrastra: hechos_mortalidad[anio], [muertes], [prediccion] y [serie_id]
# (serie_id evita que Power BI elimine filas repetidas). Marca muertes y prediccion como "No resumir".
import matplotlib.pyplot as plt

d = dataset.groupby("anio", as_index=False)[["muertes", "prediccion"]].sum(min_count=1)
fig, ax = plt.subplots(figsize=(9, 4))
ax.plot(d["anio"], d["muertes"], marker="o", color="#212529", label="Real")
ax.plot(d["anio"], d["prediccion"], marker="s", ls="--", color="#d6336c", label="Predicho")
ax.set_title("Muertes por cáncer: real vs. predicho")
ax.set_xlabel("Año")
ax.set_ylabel("Muertes")
ax.legend()
ax.grid(alpha=0.3)
plt.tight_layout()
plt.show()
'''
(BI / "visual_real_vs_predicho.py").write_text(visual, encoding="utf-8")
print("\nScripts para Power BI: cargar_en_powerbi.py y visual_real_vs_predicho.py")
print("Listo. Abre Power BI Desktop y sigue los pasos de la guía.")