# %% [markdown]
# # 01 - Inventario y limpieza de datos
# Lee los Excel de data/raw y genera tablas ordenadas (tidy) en data/processed.
# En VSCode puedes ejecutar cada celda (# %%) con Shift+Enter.

# %%
import re
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
BASE = Path(__file__).resolve().parent
RAW = BASE / "data" / "raw"
OUT = BASE / "data" / "processed"
OUT.mkdir(parents=True, exist_ok=True)

F1 = RAW / "1. COMPORTAMIENTO_CANCER.xlsx"
F2 = RAW / "2. FACTORES_RIESGO_Y_PROTECTORES_CA.xlsx"
F3 = RAW / "3. Cumplimiento_tamizacion_5_años.xlsx"
YEARS = list(range(2005, 2026))  # 21 años: 2005-2025


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
# ## 1. Mortalidad por tipo de cáncer, sexo y año (SISPRO / Estadísticas Vitales)

# %%
raw = pd.read_excel(F1, sheet_name="CAUSAS MORTALIDAD DETALLE", header=None)
filas, causa = [], None
for _, r in raw.iloc[2:].iterrows():
    if pd.notna(r[2]):
        causa = limpia(r[2])
    sexo = limpia(r[3]) if pd.notna(r[3]) else None
    if causa is None or sexo not in ("FEMENINO", "MASCULINO"):
        continue
    vals = r.iloc[5:26].tolist()  # 21 valores = 2005..2025
    for y, v in zip(YEARS, vals):
        filas.append({"tipo_cancer": causa, "sexo": sexo.title(), "anio": y,
                      "muertes": 0 if pd.isna(v) else int(v)})
mort = pd.DataFrame(filas)
mort["cod"] = mort["tipo_cancer"].str.extract(r"^(\d+)").astype(int)

# Control de calidad contra los totales del propio archivo
tot = mort.groupby("anio")["muertes"].sum()
print("Muertes por cáncer 2005-2025 (suma tidy):", int(tot.sum()), "| referencia archivo: 2917")
print("2005:", tot[2005], "(ref 108) | 2025:", tot[2025], "(ref 187)")
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
    for y, v in zip(YEARS, vals):
        if pd.notna(v):
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
for nombre, df in [("mortalidad", mort), ("grandes causas", grandes),
                   ("tamización", tam), ("factores de riesgo", f)]:
    print(f"\n=== {nombre}: {df.shape[0]} filas x {df.shape[1]} columnas")
    print(df.isna().sum().to_string())