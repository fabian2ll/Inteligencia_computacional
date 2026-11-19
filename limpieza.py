# %% [markdown]
# # 01 - Inventario y limpieza de datos
# Los datos de mortalidad vienen de los microdatos del DANE (via microdatos.py).
# Los contextos (grandes causas, tamización, factores) siguen viniendo de los Excel de data/raw.

# %%
import re
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from microdatos import construye_panel

warnings.filterwarnings("ignore")
BASE = Path(__file__).resolve().parent
RAW = BASE / "data" / "raw"
OUT = BASE / "data" / "processed"
OUT.mkdir(parents=True, exist_ok=True)

F1 = RAW / "1. COMPORTAMIENTO_CANCER.xlsx"
F2 = RAW / "2. FACTORES_RIESGO_Y_PROTECTORES_CA.xlsx"
F3 = RAW / "3. Cumplimiento_tamizacion_5_años.xlsx"
YEARS = list(range(2008, 2026))  # 18 años: 2008-2025


def limpia(s):
    return re.sub(r"\s+", " ", str(s).replace("\xa0", " ")).strip()


def a_numero(x):
    """Convierte '11,1%' -> 0.111 y '0,5' -> 0.5; deja los números como están."""
    if isinstance(x, str):
        t = x.strip()
        es_pct = "%" in t
        try:
            v = float(t.replace("%", "").replace(",", "."))
        except ValueError:
            return np.nan
        return v / 100 if es_pct else v
    return x


# %% [markdown]
# ## 1. Mortalidad por cáncer, sexo, grupo de edad y año (microdatos DANE)

# %%
micro = construye_panel()

# La tabla agregada se mantiene para los contextos y comparaciones con el Excel original
mort = (micro.groupby(["anio", "cod", "sexo"])["muertes"].sum().reset_index())
tot = mort.groupby("anio")["muertes"].sum()
print("Muertes por cáncer 2008-2025:", int(tot.sum()))
print("2008:", int(tot[2008]), "| 2025:", int(tot[2025]))
mort.to_csv(OUT / "mortalidad_tipo_sexo_anio.csv", index=False)

# %% [markdown]
# ## 2. Contexto: grandes causas de muerte (cáncer vs. otras causas)

# %%
g = pd.read_excel(F1, sheet_name=" MORTALIDAD GRANDES CAUSAS", header=None)
g_rows = []
for _, r in g.iloc[2:].iterrows():
    nombre = limpia(r[1]) if pd.notna(r[1]) else ""
    if not re.match(r"^\d{3} -|^NO DEFINIDO", nombre):
        continue
    vals = r.iloc[3:24].tolist()
    for y, v in zip(range(2005, 2026), vals):
        if pd.notna(v) and y >= 2008:
            g_rows.append({"gran_causa": nombre, "anio": y, "muertes": int(v)})
grandes = pd.DataFrame(g_rows)
grandes.to_csv(OUT / "mortalidad_grandes_causas.csv", index=False)

# %% [markdown]
# ## 3. Tamización 2021-2025 por EPS

# %%
t = pd.read_excel(F3, header=0).iloc[:, :7]
t.columns = ["indicador", "eps", 2021, 2022, 2023, 2024, 2025]
t["indicador"] = t["indicador"].map(limpia)
t["eps"] = t["eps"].map(limpia)
tam = t.melt(id_vars=["indicador", "eps"], var_name="anio", value_name="cumplimiento")
tam["cumplimiento"] = tam["cumplimiento"].map(a_numero)
tam.to_csv(OUT / "tamizacion_eps_anio.csv", index=False)

# %% [markdown]
# ## 4. Factores de riesgo por grupo de edad (foto única, sin serie temporal)

# %%
f = pd.read_excel(F2, sheet_name="FACTORES DE RIESGO PIC  ")
f.columns = ["grupo_edad", "indicador", "porcentaje", "numerador", "denominador"]
f["indicador"] = f["indicador"].map(limpia).str.replace(r"\)ntaje con actividad.*$", ")", regex=True)
f.to_csv(OUT / "factores_riesgo_grupo_edad.csv", index=False)

# %% [markdown]
# ## 5. Inventario (fase 2 de CRISP-DM)

# %%
for nombre, df in [("microdatos tipo×sexo×edad", micro), ("mortalidad agregada", mort),
                   ("grandes causas", grandes), ("tamización", tam), ("factores de riesgo", f)]:
    print(f"\n=== {nombre}: {df.shape[0]} filas x {df.shape[1]} columnas")
    print(df.isna().sum().to_string())
