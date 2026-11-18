# %% [markdown]
# # 00 -Microdatos DANE 2008-2025 -> panel por tipo, sexo, grupo de edad y año
# Convierte los archivos individuales de defunciones no fetales a la misma
# categorización CAUSA_667/666 (cod 201-215) que ya usa el proyecto.

# %%
from pathlib import Path

import pandas as pd

BASE = Path(__file__).resolve().parent
RAW = BASE / "data" / "raw"
OUT = BASE / "data" / "processed"
OUT.mkdir(parents=True, exist_ok=True)

ANIOS = range(2008, 2026)
COLS = ["COD_DPTO", "COD_MUNIC", "CODPTORE", "CODMUNRE", "ANO", "SEXO",
        "GRU_ED2", "C_BAS1", "CAUSA_667", "CAUSA_666"]
EDAD = {"01": "0-14", "02": "0-14", "03": "0-14", "04": "15-44",
        "05": "45-64", "06": "65+", "07": "Desconocida"}
SHORT = {201: "Estómago", 202: "Colon y recto", 203: "Otros digestivos", 204: "Tráquea, bronquios y pulmón",
         205: "Otros respiratorios", 206: "Mama (mujer)", 207: "Cuello uterino", 208: "Cuerpo del útero",
         209: "Útero no especificado", 210: "Próstata", 211: "Otros genitourinarios", 212: "Leucemia",
         213: "Linfático y hematopoyético", 214: "Otras localizaciones",
         215: "In situ / benignos / incierto"}
COD = [str(c) for c in SHORT]


# %%
def archivo(anio):
    if anio in (2012, 2013):
        return RAW / f"Defun_{anio}.sav", "CAUSA_666"
    if anio >= 2023:
        return RAW / f"BD-EEVV-Defuncionesnofetales-{anio}.csv", "CAUSA_667"
    return RAW / f"nofetal{anio}.csv", "CAUSA_667"


def normaliza(x, causa, anio):
    x = x.copy()
    x.columns = [str(c).upper() for c in x.columns]
    x = x[[c for c in COLS if c in x.columns]]
    for c in x.columns:
        x[c] = x[c].astype(str).str.strip()

    x["fuente"] = "csv" if causa == "CAUSA_667" else "sav"
    if causa == "CAUSA_666":
        x = x[x["CODPTORE"].str.contains("Boyac", case=False, na=False)]
        x["CODPTORE"] = "15"
        x["sexo"] = x["SEXO"].str[:1].map({"M": "Masculino", "F": "Femenino"})
        e = x["GRU_ED2"].str.lower()
        x["GRU_ED2"] = "07"
        for codigo, patron in [("01", "menor de 1"), ("02", "1 a 4"), ("03", "5 a 14"),
                               ("04", "15 a 44"), ("05", "45 a 64"), ("06", "65")]:
            x.loc[e.str.contains(patron, regex=False, na=False), "GRU_ED2"] = codigo
        x["en_cancer"] = x[causa].isin(COD)
        x["cod"] = None
    else:
        x["sexo"] = x["SEXO"].map({"1": "Masculino", "2": "Femenino"})
        x["en_cancer"] = x[causa].isin(COD)
        x["cod"] = x[causa].where(x["en_cancer"]).map(lambda v: int(v) if pd.notna(v) else None)

    x["CODMUNRE"] = x["CODMUNRE"].str.zfill(3)
    x = x[(x["CODPTORE"] == "15") & (x["CODMUNRE"] == "759")]
    x["C_BAS1"] = x["C_BAS1"].str.upper()
    x["anio"] = anio
    x["grupo_edad"] = x["GRU_ED2"].map(EDAD)
    cols = ["anio", "cod", "sexo", "grupo_edad", "C_BAS1", "en_cancer", "fuente"]
    return x[cols].dropna(subset=["sexo", "grupo_edad", "C_BAS1"])


# %%
partes = []
for anio in ANIOS:
    ruta, causa = archivo(anio)
    crudo = pd.read_spss(ruta) if ruta.suffix == ".sav" else pd.read_csv(
        ruta, usecols=lambda c: c.upper() in COLS, dtype=str, encoding="latin-1")
    x = normaliza(crudo, causa, anio)
    partes.append(x)
    print(f"{anio}: {x['en_cancer'].sum():4d} muertes por cáncer en Sogamoso")

# Mapa CIE-10 -> cod aprendido de los años con CAUSA_667 (2012-2013 solo tenían CAUSA_666)
csv = pd.concat([p for p in partes if p["fuente"].iloc[0] == "csv" and p["en_cancer"].any()])
p = csv[["C_BAS1", "cod"]].dropna()
p["pref"] = p["C_BAS1"].str[:3]
mapa = p.groupby("C_BAS1")["cod"].agg(lambda s: int(s.value_counts().idxmax()))
prefijo = p.groupby("pref")["cod"].agg(lambda s: int(s.value_counts().idxmax()))

for i, x in enumerate(partes):
    if x["fuente"].iloc[0] != "sav":
        continue
    cie = x["C_BAS1"]
    cod = cie.map(mapa).fillna(cie.str[:3].map(prefijo)).fillna(cie.str[:2].map(prefijo))
    cod = cod.where(~cie.str.startswith("D"), 215)
    cod = cod.where(~cie.str.startswith("C11"), 205)
    cod = cod.where(~cie.str.startswith("C37"), 214)
    partes[i] = x.assign(cod=cod.where(x["en_cancer"]))

micro = pd.concat(partes, ignore_index=True)
micro = micro.dropna(subset=["cod"])
micro["cod"] = micro["cod"].astype(int)
micro["tipo"] = micro["cod"].map(SHORT)
micro = (micro.groupby(["anio", "cod", "tipo", "sexo", "grupo_edad"])
              .size().rename("muertes").reset_index())
micro.to_csv(OUT / "mortalidad_microdatos_edad.csv", index=False)

# %%
vieja = pd.read_csv(OUT / "mortalidad_tipo_sexo_anio.csv")
chk = (micro.groupby(["anio", "cod", "sexo"]).muertes.sum().rename("micro").reset_index()
            .merge(vieja.groupby(["anio", "cod", "sexo"]).muertes.sum().rename("excel").reset_index(),
                   on=["anio", "cod", "sexo"], how="outer").fillna(0))
chk["dif"] = chk["micro"] - chk["excel"]
chk = chk[chk["anio"] >= 2008]
print("\nComparación contra la serie original 2008-2025 (solo diferencias):")
print(chk[chk["dif"] != 0].to_string(index=False))
print(f"\nFilas: {len(micro)} | series tipo×sexo×edad: "
      f"{micro[['cod','sexo','grupo_edad']].drop_duplicates().shape[0]}")
print("Guardado:", OUT / "mortalidad_microdatos_edad.csv")
