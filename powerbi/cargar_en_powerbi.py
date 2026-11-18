# Power BI: Inicio > Obtener datos > Script de Python > pega este código > Aceptar.
# Marca todas las tablas en el Navegador y pulsa Cargar. Requiere pandas en el Python que usa Power BI.
import pandas as pd
from pathlib import Path

RUTA = Path(r"C:\Users\USER\Documents\UNIVERSIDAD\Inteligencia Computacional\proyecto_final\powerbi")

hechos_mortalidad = pd.read_csv(RUTA / "hechos_mortalidad.csv", encoding="utf-8-sig")
dim_serie = pd.read_csv(RUTA / "dim_serie.csv", encoding="utf-8-sig")
predicciones_modelos = pd.read_csv(RUTA / "predicciones_modelos.csv", encoding="utf-8-sig")
metricas_modelos = pd.read_csv(RUTA / "metricas_modelos.csv", encoding="utf-8-sig")
error_por_serie = pd.read_csv(RUTA / "error_por_serie.csv", encoding="utf-8-sig")
curva_aprendizaje = pd.read_csv(RUTA / "curva_aprendizaje.csv", encoding="utf-8-sig")
importancia_variables = pd.read_csv(RUTA / "importancia_variables.csv", encoding="utf-8-sig")
contexto_tamizacion = pd.read_csv(RUTA / "contexto_tamizacion.csv", encoding="utf-8-sig")
contexto_factores_riesgo = pd.read_csv(RUTA / "contexto_factores_riesgo.csv", encoding="utf-8-sig")
contexto_grandes_causas = pd.read_csv(RUTA / "contexto_grandes_causas.csv", encoding="utf-8-sig")
