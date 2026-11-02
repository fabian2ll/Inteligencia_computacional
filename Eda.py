# %% [markdown]
# # 02 - Análisis exploratorio (EDA)
# Lee los CSV de data/processed, imprime los hallazgos clave y guarda las figuras en /figuras.
# Ejecuta cada celda (# %%) con Shift+Enter, o todo el archivo con el botón Run.

# %%
import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
BASE = Path(__file__).resolve().parent
PROC = BASE / "data" / "processed"
FIG = BASE / "figuras"
FIG.mkdir(exist_ok=True)

plt.rcParams.update({"figure.dpi": 110, "axes.spines.top": False, "axes.spines.right": False,
                     "axes.grid": True, "grid.alpha": 0.3, "font.size": 10})
COL_F, COL_M, COL_T = "#d6336c", "#1971c2", "#212529"

SHORT = {201: "Estómago", 202: "Colon y recto", 203: "Otros digestivos", 204: "Tráquea, bronquios y pulmón",
         205: "Otros respiratorios", 206: "Mama (mujer)", 207: "Cuello uterino", 208: "Cuerpo del útero",
         209: "Útero no especificado", 210: "Próstata", 211: "Otros genitourinarios", 212: "Leucemia",
         213: "Linfático y hematopoyético", 214: "Otras localizaciones",
         215: "In situ / benignos / incierto"}

m = pd.read_csv(PROC / "mortalidad_tipo_sexo_anio.csv")
m["tipo"] = m["cod"].map(SHORT)
g = pd.read_csv(PROC / "mortalidad_grandes_causas.csv")
tam = pd.read_csv(PROC / "tamizacion_eps_anio.csv")
fac = pd.read_csv(PROC / "factores_riesgo_grupo_edad.csv")


def guarda(nombre):
    plt.tight_layout()
    plt.savefig(FIG / nombre, bbox_inches="tight")
    plt.close()
    print("  figura guardada:", FIG / nombre)


# %% [markdown]
# ## 1. Resumen numérico

# %%
total_anio = m.groupby("anio")["muertes"].sum()
por_sexo = m.groupby(["anio", "sexo"])["muertes"].sum().unstack()
tot = int(total_anio.sum())
print(f"Muertes por cáncer 2005-2025: {tot}")
print("Por sexo:", m.groupby("sexo")["muertes"].sum().to_dict())
print(f"Promedio anual: {total_anio.mean():.1f} | mínimo {total_anio.min()} ({total_anio.idxmin()}) "
      f"| máximo {total_anio.max()} ({total_anio.idxmax()})")

pend, _ = np.polyfit(total_anio.index, total_anio.values, 1)
ini, fin = total_anio.loc[2005:2009].mean(), total_anio.loc[2021:2025].mean()
print(f"Tendencia lineal: {pend:+.1f} muertes por año")
print(f"Promedio 2005-2009: {ini:.1f} -> 2021-2025: {fin:.1f} ({(fin / ini - 1) * 100:+.0f} %)")
pre, cov = total_anio.loc[2017:2019].mean(), total_anio.loc[2020:2022].mean()
print(f"Promedio 2017-2019: {pre:.1f} vs 2020-2022: {cov:.1f} ({(cov / pre - 1) * 100:+.0f} %)")

print("\nTop 6 tipos de cáncer (total 2005-2025):")
top = m.groupby("tipo")["muertes"].sum().sort_values(ascending=False)
for t, v in top.head(6).items():
    print(f"  {t:35s} {v:4d}  ({v / tot * 100:4.1f} %)")

print("\nDistribución de la variable a predecir (muertes por tipo-sexo-año):")
print(f"  media {m['muertes'].mean():.2f} | máximo {m['muertes'].max()} | "
      f"celdas en cero {(m['muertes'] == 0).mean() * 100:.0f} %")
print(f"  varianza/media = {m['muertes'].var() / m['muertes'].mean():.1f} (1 = Poisson puro; >1 = sobredispersión)")

# %% [markdown]
# ## 2. Figura 1: evolución anual por sexo

# %%
fig, ax = plt.subplots(figsize=(10, 4.8))
ax.axvspan(2019.5, 2022.5, color="#ffd8a8", alpha=0.5, label="2020-2022")
ax.plot(total_anio.index, total_anio.values, color=COL_T, lw=2.5, marker="o", label="Total")
ax.plot(por_sexo.index, por_sexo["Femenino"], color=COL_F, lw=1.8, marker="s", ms=4, label="Femenino")
ax.plot(por_sexo.index, por_sexo["Masculino"], color=COL_M, lw=1.8, marker="^", ms=4, label="Masculino")
ax.plot(total_anio.index, np.polyval(np.polyfit(total_anio.index, total_anio.values, 1), total_anio.index),
        "--", color="gray", lw=1.2, label=f"Tendencia total ({pend:+.1f}/año)")
ax.set(title="Muertes por cáncer en Sogamoso por año (2005-2025)", xlabel="Año", ylabel="Muertes")
ax.set_xticks(range(2005, 2026, 2))
ax.legend(ncol=3, fontsize=9, loc="upper left")
guarda("fig1_muertes_por_anio.png")

# %% [markdown]
# ## 3. Figura 2: tipos de cáncer y sexo

# %%
piv = m.pivot_table(index="tipo", columns="sexo", values="muertes", aggfunc="sum").fillna(0)
piv = piv.loc[piv.sum(axis=1).sort_values().index]
fig, ax = plt.subplots(figsize=(10, 6))
ax.barh(piv.index, piv["Femenino"], color=COL_F, label="Femenino")
ax.barh(piv.index, piv["Masculino"], left=piv["Femenino"], color=COL_M, label="Masculino")
for i, (_, r) in enumerate(piv.iterrows()):
    ax.text(r.sum() + 5, i, int(r.sum()), va="center", fontsize=9)
ax.set(title="Muertes acumuladas 2005-2025 por tipo de cáncer y sexo", xlabel="Muertes")
ax.legend(loc="lower right")
guarda("fig2_tipos_cancer_sexo.png")

# %% [markdown]
# ## 4. Figura 3: mapa de calor tipo × año (8 tipos principales)

# %%
top8 = top.head(8).index
hm = m[m["tipo"].isin(top8)].pivot_table(index="tipo", columns="anio", values="muertes", aggfunc="sum")
hm = hm.loc[top8]
fig, ax = plt.subplots(figsize=(12, 4.8))
im = ax.imshow(hm.values, cmap="YlOrRd", aspect="auto")
ax.set_xticks(range(hm.shape[1]), hm.columns, rotation=60)
ax.set_yticks(range(hm.shape[0]), hm.index)
ax.grid(False)
for i in range(hm.shape[0]):
    for j in range(hm.shape[1]):
        ax.text(j, i, int(hm.values[i, j]), ha="center", va="center", fontsize=7)
fig.colorbar(im, ax=ax, label="Muertes")
ax.set_title("Muertes por año en los 8 tipos de cáncer con más casos")
guarda("fig3_mapa_calor_tipo_anio.png")

# %% [markdown]
# ## 5. Figura 4: cáncer frente a otras grandes causas

# %%
tot_g = g.groupby("anio")["muertes"].sum()
neo = g[g["gran_causa"].str.contains("NEOPLASIAS")].set_index("anio")["muertes"]
share = neo / tot_g * 100
rank = g.groupby("gran_causa")["muertes"].sum().sort_values(ascending=False)
print("\nGrandes causas de muerte 2005-2025 (total):")
for k, v in rank.items():
    print(f"  {k:62s} {v:6d}")
print(f"Participación del cáncer en el total de muertes: {share.min():.1f} % - {share.max():.1f} %")

fig, ax = plt.subplots(figsize=(10, 4))
ax.plot(share.index, share.values, color="#862e9c", lw=2.5, marker="o")
ax.set(title="Participación de las neoplasias en el total de muertes", xlabel="Año", ylabel="% del total")
ax.set_xticks(range(2005, 2026, 2))
guarda("fig4_participacion_cancer.png")

# %% [markdown]
# ## 6. Figura 5: distribución de la variable a predecir

# %%
fig, ax = plt.subplots(figsize=(8, 4))
ax.hist(m["muertes"], bins=range(0, int(m["muertes"].max()) + 2), color="#4dabf7", edgecolor="white")
ax.set(title="Muertes por tipo de cáncer, sexo y año (una celda = un dato del modelo)",
       xlabel="Muertes en el año", ylabel="Número de celdas")
ax.text(0.97, 0.9, f"media {m['muertes'].mean():.1f}\nceros {(m['muertes'] == 0).mean() * 100:.0f} %",
        transform=ax.transAxes, ha="right", va="top")
guarda("fig5_distribucion_muertes.png")

# %% [markdown]
# ## 7. Contexto: tamización por EPS (solo TOTAL MUNICIPIO, 2021-2025)

# %%
tm = tam[tam["eps"] == "TOTAL MUNICIPIO"].pivot_table(index="anio", columns="indicador", values="cumplimiento")
if tm.max().max() > 1.5:   # por si vienen en escala 0-100
    tm = tm / 100
fig, ax = plt.subplots(figsize=(10, 5))
for c in tm.columns:
    agregado = c in ("Tamizaje cancer de cuello uterino", "Tamizaje cancer de prostata")
    ax.plot(tm.index, tm[c] * 100, marker="o", lw=2.2 if agregado else 1.4,
            ls="--" if agregado else "-", label=c.replace("Tamizaje cancer de ", "").replace("Tamizacion cancer de ", ""))
ax.set(title="Cumplimiento de tamización, total municipio", xlabel="Año", ylabel="% de cumplimiento")
ax.set_xticks(range(2021, 2026))
ax.legend(fontsize=8, loc="upper left", bbox_to_anchor=(1, 1))
guarda("fig6_tamizacion.png")

# %% [markdown]
# ## 8. Contexto: factores de riesgo por grupo de edad

# %%
fac["porcentaje"] = pd.to_numeric(fac["porcentaje"], errors="coerce")
if fac["porcentaje"].max() <= 1.5:
    fac["porcentaje"] = fac["porcentaje"] * 100
claves = {"Sobrepeso": "Sobrepeso", "Obesidad Grado I": "Obesidad grado I", "que fuman": "Fuma",
          "antecedentes de diabetes": "Antecedente diabetes", "azúcar altos": "Azúcar alto",
          "colesterol alto": "Colesterol alto", "bebidas alcohólicas": "Consume alcohol",
          "actividad física": "Actividad física regular"}
filas = []
for k, nombre in claves.items():
    sub = fac[fac["indicador"].str.contains(k, regex=False)]
    for _, r in sub.iterrows():
        filas.append({"indicador": nombre, "grupo_edad": r["grupo_edad"].split(" (")[0].title(),
                      "porcentaje": r["porcentaje"]})
fp = pd.DataFrame(filas).pivot_table(index="indicador", columns="grupo_edad", values="porcentaje")
fp = fp[[c for c in ["Juventud", "Adultez", "Vejez"] if c in fp.columns]].loc[list(claves.values())]
print("\nFactores de riesgo (% por grupo de edad):")
print(fp.round(1).to_string())

fig, ax = plt.subplots(figsize=(10, 5))
fp.plot.barh(ax=ax, color=["#74c0fc", "#4263eb", "#364fc7"][:fp.shape[1]], width=0.8)
ax.invert_yaxis()
ax.set(title="Factores de riesgo y protectores por grupo de edad", xlabel="% de la población evaluada", ylabel="")
guarda("fig7_factores_riesgo.png")

print("\nListo. Revisa la carpeta:", FIG)