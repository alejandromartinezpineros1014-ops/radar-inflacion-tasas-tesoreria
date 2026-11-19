"""
motor.py - Funciones del modelo para la aplicación Streamlit.

Es el MISMO modelo de modelo_completo_v3.py (mismos parámetros, misma semilla y misma
lógica), organizado en funciones para que la app pueda reutilizarlo y actualizarlo con
meses nuevos. Si se cambia algo del modelo, debe cambiarse en los dos archivos.
"""
from pathlib import Path

import numpy as np
import pandas as pd
from statsmodels.tsa.api import VAR

COLS_VAR = ["inflacion_anual", "tasa_interes", "var_anual_trm", "var_anual_brent"]
FASES = ["bajando", "estable", "subiendo"]

REZAGOS = 3
HORIZONTE = 6
N_SIM = 5000
UMBRAL_TASA = 0.5
UMBRAL_PROB = 0.40
SPREAD_CDT = 0.28
BLOQUE = 3
SEMILLA = 2026

TEXTO = {
    "AMARRAR": "CDT a plazo largo (6+ meses) y, como criterio condicionado, crédito a tasa variable.",
    "RENOVAR CORTO": "CDT a 90 días y, como criterio condicionado, crédito a tasa fija.",
    "ESCALONAR": "Repartir plazos: parte en CDT corto y parte en CDT largo.",
}

EQUIVALENCIAS_EXCEL = {
    "Inflación anual (%)": "inflacion_anual",
    "Tasa de interés (%)": "tasa_interes",
    "Var. anual TRM (log, %)": "var_anual_trm",
    "Var. anual Brent (log, %)": "var_anual_brent",
    "Fase ciclo de tasas": "fase_ciclo_tasa",
    "Estado inflación vs meta": "estado_inflacion",
}


# ---------------------------------------------------------------------------
# Datos
# ---------------------------------------------------------------------------
def cargar_base(carpeta="datos"):
    """Lee la base del modelo (CSV o, si falta, el Excel) y verifica que esté completa."""
    carpeta = Path(carpeta)
    if (carpeta / "base_inflacion_tasas.csv").exists():
        base = pd.read_csv(carpeta / "base_inflacion_tasas.csv", index_col=0, parse_dates=True)
    else:
        base = pd.read_excel(carpeta / "base_inflacion_tasas.xlsx", sheet_name="Base_modelo",
                             index_col=0, parse_dates=True).rename(columns=EQUIVALENCIAS_EXCEL)
    base.index.name = "fecha"
    assert not base.index.duplicated().any(), "Hay fechas duplicadas"
    assert (base.index == pd.date_range(base.index[0], periods=len(base), freq="MS")).all(), \
        "Las fechas no son mensuales consecutivas"
    assert base[COLS_VAR + ["fase_ciclo_tasa"]].notna().all().all(), "Hay datos faltantes"
    base.index.freq = "MS"
    return base


def cargar_originales(carpeta="datos"):
    """Datos originales (2003-2026) en niveles: IPC, tasa, TRM y Brent."""
    orig = pd.read_csv(Path(carpeta) / "datos_originales.csv", index_col=0, parse_dates=True)
    orig.index.name = "fecha"
    return orig


def fase_de_cambio(cambio):
    """Misma regla que la base: <= -0,25 bajando; >= +0,25 subiendo; si no, estable."""
    cambio = round(cambio, 4)
    if cambio <= -0.25:
        return "bajando"
    if cambio >= 0.25:
        return "subiendo"
    return "estable"


def estado_inflacion(inf):
    if inf <= 4:
        return "en la meta (<=4%)"
    if inf <= 6:
        return "por encima (>4% y <=6%)"
    return "muy por encima (>6%)"


def agregar_meses(base, originales, nuevos):
    """
    Agrega meses nuevos a la base. `nuevos` es un DataFrame con índice de fechas (inicio de
    mes, consecutivas después del último mes de la base) y columnas:
      inflacion_anual (%), tasa_interes (%), trm (pesos por dólar), brent (USD por barril).
    Las variaciones anuales de TRM y Brent se calculan contra el mismo mes del año anterior
    (tomado de los datos originales) y la fase con el cambio de la tasa en 3 meses.
    """
    base = base.copy()
    tasa = base["tasa_interes"].copy()
    for fecha, fila in nuevos.sort_index().iterrows():
        esperado = base.index[-1] + pd.DateOffset(months=1)
        if fecha != esperado:
            raise ValueError(f"El mes {fecha:%Y-%m} no sigue al último mes de la base ({base.index[-1]:%Y-%m}).")
        hace_12 = fecha - pd.DateOffset(months=12)
        trm_12 = originales.loc[hace_12, "trm_promedio"]
        brent_12 = originales.loc[hace_12, "brent_usd"]
        tasa_3 = tasa.loc[fecha - pd.DateOffset(months=3)]
        nueva = {
            "inflacion_anual": float(fila["inflacion_anual"]),
            "tasa_interes": float(fila["tasa_interes"]),
            "var_anual_trm": 100 * np.log(float(fila["trm"]) / trm_12),
            "var_anual_brent": 100 * np.log(float(fila["brent"]) / brent_12),
            "fase_ciclo_tasa": fase_de_cambio(float(fila["tasa_interes"]) - tasa_3),
            "estado_inflacion": estado_inflacion(float(fila["inflacion_anual"])),
        }
        base.loc[fecha] = pd.Series(nueva)
        tasa.loc[fecha] = nueva["tasa_interes"]
    base.index.freq = "MS"
    return base


# ---------------------------------------------------------------------------
# Modelo
# ---------------------------------------------------------------------------
def matriz_transicion(estados):
    """Cuenta cuántas veces se pasó de cada fase a cada fase y lo vuelve probabilidad."""
    conteo = np.ones((3, 3)) * 0.01
    for a, b in zip(estados[:-1], estados[1:]):
        conteo[a, b] += 1
    return conteo / conteo.sum(axis=1, keepdims=True)


def simular(var_mod, historia, fases_hist, P_mark, fase_inicial, n_sim, horizonte, rng, bloque=BLOQUE):
    """Monte Carlo: VAR + sorpresas históricas por bloques, elegidas según la fase de Markov
    (fase del mes ANTERIOR a cada residuo). Igual que en modelo_completo_v3.py."""
    A, c = var_mod.coefs, var_mod.intercept
    p = A.shape[0]
    resid = var_mod.resid.values
    fase_resid = fases_hist[p - 1:-1]
    assert len(fase_resid) == len(resid)
    ultimo_inicio = len(resid) - bloque
    bolsas = [np.where((fase_resid == k) & (np.arange(len(resid)) <= ultimo_inicio))[0] for k in range(3)]
    bolsas = [b if len(b) > 5 else np.arange(ultimo_inicio + 1) for b in bolsas]

    rezagos = np.repeat(historia[-p:][None, :, :], n_sim, axis=0)
    fase = np.full(n_sim, fase_inicial)
    salida = np.zeros((n_sim, horizonte, historia.shape[1]))
    idx = np.zeros(n_sim, dtype=int)
    for h in range(horizonte):
        if h % bloque == 0:
            for k in range(3):
                m = fase == k
                idx[m] = rng.choice(bolsas[k], size=m.sum())
        else:
            idx = idx + 1
        y = c + sum(rezagos[:, -(j + 1)] @ A[j].T for j in range(p)) + resid[idx]
        salida[:, h] = y
        rezagos = np.concatenate([rezagos[:, 1:], y[:, None]], axis=1)
        u = rng.random(n_sim)
        fase = (u[:, None] > P_mark[fase].cumsum(axis=1)).sum(axis=1)
    assert np.isfinite(salida).all(), "La simulación produjo valores no finitos"
    return salida


def recomendar(p_baja, p_alza, umbral_prob=UMBRAL_PROB):
    if p_baja >= umbral_prob and p_baja > p_alza:
        return "AMARRAR"
    if p_alza >= umbral_prob and p_alza > p_baja:
        return "RENOVAR CORTO"
    return "ESCALONAR"


def correr_modelo(base, horizonte=HORIZONTE, umbral_tasa=UMBRAL_TASA, n_sim=N_SIM, semilla=SEMILLA):
    """Estima VAR y Markov, simula Monte Carlo y aplica la regla de decisión.
    Devuelve un diccionario con todo lo que muestra la app."""
    datos = base[COLS_VAR]
    var = VAR(datos).fit(REZAGOS)
    codigo_fase = base["fase_ciclo_tasa"].map({f: i for i, f in enumerate(FASES)}).values
    assert np.isin(codigo_fase, [0, 1, 2]).all(), "Hay fases no reconocidas en fase_ciclo_tasa."
    P = matriz_transicion(codigo_fase)
    fase_hoy = int(codigo_fase[-1])

    sims = simular(var, datos.values, codigo_fase, P, fase_hoy, n_sim, horizonte, np.random.default_rng(semilla))
    tasa_hoy = float(datos["tasa_interes"].iloc[-1])
    inf_hoy = float(datos["inflacion_anual"].iloc[-1])
    tasa_fin, inf_fin = sims[:, -1, 1], sims[:, -1, 0]
    p_baja = float((tasa_fin <= tasa_hoy - umbral_tasa).mean())
    p_alza = float((tasa_fin >= tasa_hoy + umbral_tasa).mean())

    # Sensibilidad al remuestreo (residuos sueltos, bloques de 3 y de 6 meses)
    filas = []
    for b in [1, 3, 6]:
        sb = simular(var, datos.values, codigo_fase, P, fase_hoy, n_sim, horizonte,
                     np.random.default_rng(semilla), bloque=b)
        tf = sb[:, -1, 1]
        pb, pa = float((tf <= tasa_hoy - umbral_tasa).mean()), float((tf >= tasa_hoy + umbral_tasa).mean())
        filas.append({"remuestreo": "residuos sueltos" if b == 1 else f"bloques de {b} meses",
                      "P(baja) %": 100 * pb, "P(sube) %": 100 * pa, "decisión": recomendar(pb, pa)})
    tabla_bloques = pd.DataFrame(filas).set_index("remuestreo")

    reco = recomendar(p_baja, p_alza)
    fragil = tabla_bloques["decisión"].nunique() > 1
    reco_final = "ESCALONAR" if fragil else reco

    fechas = pd.date_range(datos.index[-1] + pd.DateOffset(months=1), periods=horizonte, freq="MS")
    def resumen(k):
        return pd.DataFrame({
            "p5": np.percentile(sims[:, :, k], 5, axis=0),
            "p25": np.percentile(sims[:, :, k], 25, axis=0),
            "mediana": np.median(sims[:, :, k], axis=0),
            "p75": np.percentile(sims[:, :, k], 75, axis=0),
            "p95": np.percentile(sims[:, :, k], 95, axis=0),
        }, index=fechas)

    return {
        "var": var, "P": P, "fase_hoy": fase_hoy, "codigo_fase": codigo_fase,
        "prob_markov": {h: np.linalg.matrix_power(P, h)[fase_hoy] for h in (3, 6, 12)},
        "tasa_hoy": tasa_hoy, "inf_hoy": inf_hoy, "fecha_hoy": datos.index[-1],
        "p_baja": p_baja, "p_alza": p_alza, "p_meta": float((inf_fin <= 4).mean()),
        "inf_mediana": float(np.median(inf_fin)),
        "tabla_bloques": tabla_bloques, "reco": reco, "fragil": fragil, "reco_final": reco_final,
        "fan_inflacion": resumen(0), "fan_tasa": resumen(1),
        "pronostico_var": pd.DataFrame(var.forecast(datos.values[-REZAGOS:], horizonte),
                                       index=fechas, columns=COLS_VAR),
    }


def rentabilidad_real(tasa_nominal, inflacion):
    """Rentabilidad real (%) = [(1 + tasa/100) / (1 + inflación/100) - 1] x 100."""
    return 100 * ((1 + tasa_nominal / 100) / (1 + inflacion / 100) - 1)
