# Visual de Python en Power BI: real vs. predicho por año.
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
