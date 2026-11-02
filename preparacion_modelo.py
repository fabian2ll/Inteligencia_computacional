# %% [markdown]
# # 03 - Preparación de la tabla del modelo (fase 3 de CRISP-DM)
# Lee data/processed/mortalidad_tipo_sexo_anio.csv y construye:
#   1) tabla_modelo.csv: una fila por (tipo de cáncer, sexo, año) con rezagos y promedios móviles
#   2) lstm_secuencias.npz: las mismas filas en forma de ventanas de 3 años, escaladas, para el LSTM
# Regla de oro: cada predictor de un año usa SOLO años anteriores (sin fuga de información).
# Ejecuta cada celda (# %%) con Shift+Enter, o todo el archivo con el botón Run.

# %%
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
BASE = Path(__file__).resolve().parent
PROC = BASE / "data" / "processed"

YEARS = list(range(2005, 2026))      # 21 años
LAGS = (1, 2, 3)                     # rezagos y ventana de la línea base
ANIO_TRAIN_FIN = 2021                # entrenamiento hasta 2021; prueba 2022-2025
ATIPICOS = (2020, 2022)              # rango de años atípicos (inclusive)

SHORT = {201: "Estómago", 202: "Colon y recto", 203: "Otros digestivos", 204: "Tráquea, bronquios y pulmón",
         205: "Otros respiratorios", 206: "Mama (mujer)", 207: "Cuello uterino", 208: "Cuerpo del útero",
         209: "Útero no especificado", 210: "Próstata", 211: "Otros genitourinarios", 212: "Leucemia",
         213: "Linfático y hematopoyético", 214: "Otras localizaciones",
         215: "In situ / benignos / incierto"}

# %% [markdown]
# ## 1. Panel completo: una fila por serie y año
# Una "serie" es una combinación tipo de cáncer × sexo. Los tipos de un solo sexo (mama, cuello uterino,
# cuerpo del útero, útero no especificado, próstata) solo generan una serie, por eso son 25 y no 30.
# Los años sin registro dentro de una serie se completan con 0 muertes.

# %%
m = pd.read_csv(PROC / "mortalidad_tipo_sexo_anio.csv")

series = (m[["cod", "sexo"]].drop_duplicates().sort_values(["cod", "sexo"]).reset_index(drop=True))
series["serie_id"] = range(len(series))
malla = series.merge(pd.DataFrame({"anio": YEARS}), how="cross")
panel = malla.merge(m[["cod", "sexo", "anio", "muertes"]], on=["cod", "sexo", "anio"], how="left")
n_vacias = int(panel["muertes"].isna().sum())
panel["muertes"] = panel["muertes"].fillna(0).astype(int)
panel = panel.sort_values(["serie_id", "anio"]).reset_index(drop=True)
print(f"Series (tipo x sexo): {len(series)} | filas del panel: {len(panel)} (esperado 25 x 21 = 525)")
print(f"Celdas completadas con 0 (no estaban en el CSV): {n_vacias}")

# %% [markdown]
# ## 2. Variables predictoras
# lag1, lag2, lag3: muertes de la misma serie 1, 2 y 3 años antes.
# mm3: promedio móvil de esos tres años (es exactamente la línea base del proyecto).
# anio_idx: tendencia lineal. atipico_covid: 1 en 2020-2022. sexo_masc y cod identifican la serie.

# %%
g = panel.groupby("serie_id")["muertes"]
for k in LAGS:
    panel[f"lag{k}"] = g.shift(k)
panel["mm3"] = panel[[f"lag{k}" for k in LAGS]].mean(axis=1, skipna=False)
panel["anio_idx"] = panel["anio"] - YEARS[0]
panel["atipico_covid"] = panel["anio"].between(*ATIPICOS).astype(int)
panel["sexo_masc"] = (panel["sexo"] == "Masculino").astype(int)
panel["tipo"] = panel["cod"].map(SHORT)

FEATURES = ["lag1", "lag2", "lag3", "mm3", "anio_idx", "atipico_covid", "sexo_masc", "cod"]
assert "muertes" not in FEATURES

# Conjunto de cada fila: los primeros 3 años de cada serie no tienen rezagos completos
panel["conjunto"] = np.select(
    [panel["lag3"].isna(), panel["anio"] <= ANIO_TRAIN_FIN],
    ["sin_rezagos", "train"], default="test")

# %% [markdown]
# ## 3. Control de calidad y verificación de no fuga

# %%
print("\nFilas por conjunto:")
print(panel["conjunto"].value_counts().reindex(["sin_rezagos", "train", "test"]).to_string())
print("Años sin rezagos completos:", sorted(int(a) for a in panel.loc[panel["conjunto"] == "sin_rezagos", "anio"].unique()))
print("Años de entrenamiento:", panel.loc[panel["conjunto"] == "train", "anio"].agg(["min", "max"]).tolist(),
      "| años de prueba:", panel.loc[panel["conjunto"] == "test", "anio"].agg(["min", "max"]).tolist())

# Recalcula cada rezago a mano desde la tabla ancha y compara: si coincide, no hay fuga
piv = panel.pivot(index="serie_id", columns="anio", values="muertes")
chk = panel[panel["conjunto"] != "sin_rezagos"]
for k in LAGS:
    esperado = np.array([piv.loc[s, a - k] for s, a in zip(chk["serie_id"], chk["anio"])])
    assert np.array_equal(chk[f"lag{k}"].values, esperado), f"lag{k} usa datos incorrectos"
esperado_mm = np.array([np.mean([piv.loc[s, a - k] for k in LAGS]) for s, a in zip(chk["serie_id"], chk["anio"])])
assert np.allclose(chk["mm3"].values, esperado_mm), "mm3 usa datos incorrectos"
print("Verificación de no fuga: OK (todos los predictores usan solo años anteriores)")

# Referencia: la línea base sobre el periodo de prueba (el mínimo que deben superar los modelos)
te = panel[panel["conjunto"] == "test"]
err = te["muertes"] - te["mm3"]
print(f"\nLínea base en prueba 2022-2025: MAE {err.abs().mean():.3f} | RMSE {np.sqrt((err ** 2).mean()):.3f}")
print(f"Celdas en cero en prueba: {(te['muertes'] == 0).mean() * 100:.0f} % | media {te['muertes'].mean():.2f}")

# %% [markdown]
# ## 4. Guardar la tabla del modelo

# %%
cols = ["serie_id", "cod", "tipo", "sexo", "anio", "muertes", "lag1", "lag2", "lag3", "mm3",
        "anio_idx", "atipico_covid", "sexo_masc", "conjunto"]
panel[cols].to_csv(PROC / "tabla_modelo.csv", index=False)
print("\nGuardado:", PROC / "tabla_modelo.csv", f"({len(panel)} filas)")

# %% [markdown]
# ## 5. Secuencias para el LSTM
# Cada ejemplo es la ventana de los 3 años anteriores (del más antiguo al más reciente), con dos canales
# por paso: log(1 + muertes) estandarizado y el indicador de año atípico. La serie se codifica en one-hot
# como variable estática. El escalado se ajusta SOLO con años <= 2021 para no filtrar información de la prueba.

# %%
W = len(LAGS)
previo = panel[panel["anio"] <= ANIO_TRAIN_FIN]
mu = float(np.log1p(previo["muertes"]).mean())
sd = float(np.log1p(previo["muertes"]).std())


def escala(v):
    return (np.log1p(v) - mu) / sd


filas = panel[panel["conjunto"] != "sin_rezagos"].reset_index(drop=True)
X_seq = np.zeros((len(filas), W, 2), dtype="float32")
for i, (s, a) in enumerate(zip(filas["serie_id"], filas["anio"])):
    for j, k in enumerate(range(W, 0, -1)):          # k = 3, 2, 1 -> de más antiguo a más reciente
        X_seq[i, j, 0] = escala(piv.loc[s, a - k])
        X_seq[i, j, 1] = int(ATIPICOS[0] <= a - k <= ATIPICOS[1])
X_static = np.eye(len(series), dtype="float32")[filas["serie_id"].values]

np.savez_compressed(PROC / "lstm_secuencias.npz",
                    X_seq=X_seq, X_static=X_static, y=filas["muertes"].values.astype("float32"),
                    anio=filas["anio"].values, serie_id=filas["serie_id"].values,
                    conjunto=filas["conjunto"].values.astype(str), mu=mu, sd=sd)
print(f"X_seq {X_seq.shape} | X_static {X_static.shape} | mu {mu:.3f} sd {sd:.3f}")
print("Guardado:", PROC / "lstm_secuencias.npz")
print("\nListo. Siguiente: 04_modelado.py")