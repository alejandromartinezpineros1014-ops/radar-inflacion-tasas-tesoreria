"""
=====================================================================
 PROYECTO INTEGRADOR - Inflación y tasas de interés en Colombia
 Pronóstico para decisiones de tesorería (CDT y crédito), 2004-2026
 Versión 3 (correcciones C1-C13 + remuestreo por bloques, advertencias
 visibles y prueba de McNemar)
=====================================================================
Qué hace este script, en orden:
  PARTE 1. Carga la base (CSV, o Excel si falta el CSV) y la verifica.
  PARTE 2. Estacionariedad y cointegración (ADF, KPSS, Johansen).
  PARTE 3. VAR: rezagos, estimación, diagnósticos, Granger,
           impulso-respuesta (con sensibilidad al orden) y pronóstico.
  PARTE 4. Cadena de Markov sobre el ciclo de tasas (descriptiva).
  PARTE 5. Monte Carlo (VAR + fases de Markov) y chequeos de alineación.
  PARTE 6. Recomendación para la tesorería (decide SOLO Monte Carlo).
  PARTE 7. Validación fuera de muestra: dirección, señales falsas,
           subperiodos, VAR en niveles vs diferencias y estrategias CDT.

Requisitos:  pip install pandas numpy matplotlib statsmodels openpyxl scipy
Ejecutar:    python modelo_completo_v3.py
Los gráficos y tablas se guardan en la carpeta  resultados/
=====================================================================
"""

from pathlib import Path
import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.api import VAR
from statsmodels.tsa.stattools import adfuller, kpss
from statsmodels.tsa.vector_ar.vecm import coint_johansen
from statsmodels.stats.diagnostic import het_arch

# NO se ocultan las advertencias: se registran todas y al final se muestra un
# resumen (qué advertencia, cuántas veces) para revisar si alguna es grave.
warnings.simplefilter("always")
REGISTRO_AVISOS = []
warnings.showwarning = lambda msg, cat, *a, **k: REGISTRO_AVISOS.append(f"{cat.__name__}: {str(msg)[:110]}")

RUTA_CSV = Path("datos/base_inflacion_tasas.csv")
RUTA_XLSX = Path("datos/base_inflacion_tasas.xlsx")   # respaldo si no está el CSV
CARPETA = Path("resultados")
CARPETA.mkdir(exist_ok=True)

# ---- Parámetros del proyecto (decisiones de diseño, ver ficha) ----
REZAGOS = 3                 # meses hacia atrás que mira el VAR
HORIZONTE = 6               # meses a pronosticar para la decisión
N_SIM = 5000                # número de trayectorias de Monte Carlo
UMBRAL_TASA = 0.5           # movimiento de tasa (puntos) que consideramos relevante
UMBRAL_PROB = 0.40          # probabilidad mínima para recomendar amarrar o renovar
# SUPUESTO (C9): un CDT paga la tasa interbancaria + 0,28 puntos.
# Origen: promedio de (CDT 90 días - tasa interbancaria), FRED, 2004-2023 (mediana 0,25;
# desde 2015 promedio 0,24). Es un supuesto: la tasa interbancaria NO es la tasa que
# ofrece un banco comercial a una empresa en particular.
SPREAD_CDT = 0.28
BLOQUE = 3                  # remuestreo por bloques: meses consecutivos de residuos (1 = residuos sueltos)
MONTO = 200_000_000         # monto ilustrativo de la tesorería (COP)
CORRER_VALIDACION = True    # pon False si quieres que el script corra más rápido

COLS_VAR = ["inflacion_anual", "tasa_interes", "var_anual_trm", "var_anual_brent"]
FASES = ["bajando", "estable", "subiendo"]


def titulo(texto):
    print("\n" + "=" * 70 + f"\n{texto}\n" + "=" * 70)


# =====================================================================
# PARTE 1. CARGAR Y VERIFICAR LA BASE  (C10)
# =====================================================================
titulo("PARTE 1. BASE DE DATOS")
if RUTA_CSV.exists():
    base = pd.read_csv(RUTA_CSV, index_col=0, parse_dates=True)
    origen = RUTA_CSV
elif RUTA_XLSX.exists():
    base = pd.read_excel(RUTA_XLSX, sheet_name="Base_modelo", index_col=0, parse_dates=True)
    # Homologar los nombres del Excel con los nombres que utiliza el modelo
    equivalencias = {
        "Inflación anual (%)": "inflacion_anual",
        "Tasa de interés (%)": "tasa_interes",
        "Var. anual TRM (log, %)": "var_anual_trm",
        "Var. anual Brent (log, %)": "var_anual_brent",
        "Fase ciclo de tasas": "fase_ciclo_tasa",
        "Estado inflación vs meta": "estado_inflacion",
    }
    base = base.rename(columns=equivalencias)
    base.index.name = "fecha"
    origen = RUTA_XLSX
else:
    raise FileNotFoundError("No encuentro la base: pon base_inflacion_tasas.csv (o .xlsx) en la carpeta datos/")

# Verificaciones: si algo falla, el script se detiene y dice por qué
esperadas = COLS_VAR + ["fase_ciclo_tasa", "estado_inflacion"]
faltan = [c for c in esperadas if c not in base.columns]
assert not faltan, f"Faltan columnas: {faltan}"
assert not base.index.duplicated().any(), "Hay fechas duplicadas"
assert (base.index == pd.date_range(base.index[0], periods=len(base), freq="MS")).all(), \
    "Las fechas no son mensuales consecutivas (falta o sobra algún mes)"
assert base[esperadas].notna().all().all(), "Hay datos faltantes"
if len(base) != 273:
    print(f"AVISO: la base tiene {len(base)} filas (la versión entregada tiene 273).")
base.index.freq = "MS"
print(f"Archivo: {origen}")
print(f"Periodo: {base.index[0]:%Y-%m} a {base.index[-1]:%Y-%m}  |  Meses: {len(base)}  |  "
      "sin duplicados, sin huecos, sin faltantes: OK")

datos = base[COLS_VAR]
print("\nEstadísticas descriptivas:\n", datos.describe().round(2).to_string())

fig, ejes = plt.subplots(4, 1, figsize=(10, 10), sharex=True)
nombres = ["Inflación anual (%)", "Tasa de interés (%)", "Var. anual TRM (log, %)", "Var. anual Brent (log, %)"]
for eje, col, nom in zip(ejes, COLS_VAR, nombres):
    eje.plot(datos.index, datos[col], lw=1.2)
    eje.set_title(nom, fontsize=10)
    eje.grid(alpha=0.3)
ejes[0].axhspan(2, 4, color="green", alpha=0.1, label="Rango meta BanRep (2%-4%)")
ejes[0].legend(fontsize=8)
plt.tight_layout()
plt.savefig(CARPETA / "1_series.png", dpi=150)
plt.close()


# =====================================================================
# PARTE 2. ESTACIONARIEDAD Y COINTEGRACIÓN  (C5)
# =====================================================================
titulo("PARTE 2. ESTACIONARIEDAD (ADF y KPSS) Y COINTEGRACIÓN (Johansen)")
print("ADF:  H0 = raíz unitaria (p < 0,05 -> estacionaria)")
print("KPSS: H0 = estacionaria  (p < 0,05 -> NO estacionaria)")
filas = []
for col in COLS_VAR:
    filas.append({
        "variable": col,
        "ADF nivel p": adfuller(datos[col], autolag="AIC")[1],
        "ADF diferencia p": adfuller(datos[col].diff().dropna(), autolag="AIC")[1],
        "KPSS nivel p": kpss(datos[col], regression="c", nlags="auto")[1],
    })
tabla_est = pd.DataFrame(filas).set_index("variable").round(3)
print(tabla_est.to_string())
print("(KPSS reporta p entre 0,01 y 0,10; 0,100 significa 'p >= 0,10')")

def traza_johansen_real(x, k):
    """Recalcula la traza de Johansen con valores propios garantizados reales
    (problema simétrico generalizado). Sirve para comprobar que la ComplexWarning
    de statsmodels es solo redondeo numérico."""
    from scipy.linalg import eigh
    from statsmodels.tsa.tsatools import lagmat
    dx = np.diff(x, axis=0)
    z = np.column_stack([np.ones(len(dx) - k), lagmat(dx, k)[k:]])
    r0 = dx[k:] - z @ np.linalg.lstsq(z, dx[k:], rcond=None)[0]
    rk = x[k:-1] - z @ np.linalg.lstsq(z, x[k:-1], rcond=None)[0]
    n = len(r0)
    s00, s0k, skk = r0.T @ r0 / n, r0.T @ rk / n, rk.T @ rk / n
    lam = eigh(s0k.T @ np.linalg.inv(s00) @ s0k, skk, eigvals_only=True)[::-1]
    return np.array([-n * np.log(1 - lam[i:]).sum() for i in range(len(lam))])


for nombre, cols in [("inflación-tasa", ["inflacion_anual", "tasa_interes"]), ("4 variables", COLS_VAR)]:
    j = coint_johansen(datos[cols], det_order=0, k_ar_diff=2)
    rango = int((j.lr1 > j.cvt[:, 1]).sum())
    print(f"Johansen {nombre}: traza {np.round(j.lr1, 2)} vs crítico 95% {np.round(j.cvt[:, 1], 2)} -> rango {rango}")
    control = traza_johansen_real(datos[cols].values, 2)
    assert np.allclose(control, j.lr1, atol=1e-6), "La traza de Johansen NO coincide con el cálculo de control: revisar"
    print(f"   control con valores propios reales: {np.round(control, 2)} -> coincide")
print("""Lectura: las pruebas ADF y KPSS presentan resultados no concluyentes para algunas
variables, especialmente inflación y tasa de interés (ADF no rechaza raíz unitaria y KPSS
tampoco rechaza estacionariedad).
El rango COMPLETO obtenido en Johansen no demuestra por sí mismo cointegración entre
variables integradas de orden uno (rango completo es lo que se esperaría si las series
fueran estacionarias). Por tanto, NO afirmamos que se haya demostrado una relación de
equilibrio de largo plazo.
Se conserva el VAR en niveles como especificación principal para analizar la dinámica
conjunta, pero es una decisión de modelación que se contrasta con un VAR en primeras
diferencias mediante validación fuera de muestra (Parte 7). Los resultados deben
interpretarse considerando las limitaciones de estacionariedad, autocorrelación y
especificación del modelo.""")


# =====================================================================
# PARTE 3. MODELO VAR
# =====================================================================
titulo("PARTE 3. MODELO VAR")
modelo = VAR(datos)

# 3.1 Selección de rezagos
seleccion = modelo.select_order(maxlags=12)
print("Rezagos sugeridos por cada criterio:", seleccion.selected_orders)
print(f"Usamos {REZAGOS} rezagos (punto intermedio entre BIC y AIC/HQIC).")

# 3.2 Estimación
var = modelo.fit(REZAGOS)

# 3.3 Diagnósticos (C4)
print("\nDiagnósticos de los residuos:")
diag = [
    ("Estabilidad (raíz máx. < 1)", f"{max(abs(var.roots) ** -1):.3f}", "Cumple: el modelo es estable"),
    ("Autocorrelación Portmanteau(12)", f"p = {var.test_whiteness(nlags=12).pvalue:.4f}", ""),
    ("Autocorrelación Portmanteau(24)", f"p = {var.test_whiteness(nlags=24).pvalue:.4f}", ""),
    ("Normalidad Jarque-Bera", f"p = {var.test_normality().pvalue:.4f}", ""),
]
for col in COLS_VAR:
    diag.append((f"ARCH(12) {col}", f"p = {het_arch(var.resid[col], nlags=12)[1]:.4f}", ""))
tabla_diag = pd.DataFrame(diag, columns=["prueba", "resultado", "nota"])
print(tabla_diag.to_string(index=False))
pm_lags = {p: VAR(datos).fit(p).test_whiteness(nlags=24).pvalue for p in [3, 6, 12]}
print("Portmanteau(24) con más rezagos:", {k: round(v, 4) for k, v in pm_lags.items()})
print("""Consecuencias:
  - Autocorrelación: no desaparece ni con 12 rezagos. Es en buena parte mecánica: las
    variables son variaciones ANUALES medidas cada mes (ventanas de 12 meses que se
    traslapan). Los errores estándar e intervalos analíticos del VAR no son confiables.
  - No normalidad y presencia de efectos ARCH en algunas variables: por eso los rangos
    no se obtienen suponiendo errores normales, sino mediante simulación Monte Carlo con
    residuos históricos. Los residuos se remuestrean por bloques de meses consecutivos
    para conservar parcialmente su dependencia temporal. La fase de Markov condiciona la
    selección del inicio de cada bloque; los residuos siguientes se incorporan
    consecutivamente, aunque la fase simulada pueda cambiar durante ese bloque.
  - Lo que sí validamos con rigor es el desempeño fuera de muestra (Parte 7).""")
tabla_diag.to_csv(CARPETA / "diagnosticos_var.csv", index=False)

# 3.4 Granger (C6): capacidad predictiva, no causalidad
print("\nCausalidad de Granger = capacidad predictiva / precedencia temporal")
print("(p < 0,05: los rezagos de X mejoran el pronóstico de Y; NO prueba que X cause Y):")
for x in ["tasa_interes", "var_anual_trm", "var_anual_brent"]:
    p = var.test_causality("inflacion_anual", [x], kind="f").pvalue
    print(f"  {x:16s} -> inflación   p = {p:.4f}")
for x in ["inflacion_anual", "var_anual_trm", "var_anual_brent"]:
    p = var.test_causality("tasa_interes", [x], kind="f").pvalue
    print(f"  {x:16s} -> tasa        p = {p:.4f}")
print("Ojo: el BanRep mira muchas más cosas (expectativas, brecha del producto, tasa de la Fed,")
print("crédito). Al no estar en el modelo, parte de su efecto queda en los residuos.")

# 3.5 Impulso-respuesta y sensibilidad al orden (C7)
irf = var.irf(24)
fig = irf.plot(orth=True, response="inflacion_anual", figsize=(10, 7))
fig.suptitle("Respuesta de la inflación a un choque de 1 desviación estándar", fontsize=11)
plt.tight_layout()
plt.savefig(CARPETA / "2_impulso_respuesta.png", dpi=150)
plt.close()

ordenes = {
    "inflación > tasa > TRM > Brent (base)": COLS_VAR,
    "Brent > TRM > inflación > tasa": ["var_anual_brent", "var_anual_trm", "inflacion_anual", "tasa_interes"],
    "Brent > TRM > tasa > inflación": ["var_anual_brent", "var_anual_trm", "tasa_interes", "inflacion_anual"],
    "tasa > inflación > TRM > Brent": ["tasa_interes", "inflacion_anual", "var_anual_trm", "var_anual_brent"],
}
filas = []
for nombre, orden in ordenes.items():
    r = VAR(datos[orden]).fit(REZAGOS).irf(24).orth_irfs[:, orden.index("inflacion_anual"), orden.index("tasa_interes")]
    filas.append({"orden de Cholesky": nombre, "6m": r[6], "12m": r[12], "18m": r[18], "24m": r[24]})
tabla_orden = pd.DataFrame(filas).set_index("orden de Cholesky").round(2)
print("\nRespuesta de la inflación (pts) a un choque de tasa, según el orden:")
print(tabla_orden.to_string())
print("""Lectura (price puzzle): con los 4 órdenes la inflación SUBE tras un alza de tasa y se
apaga hacia los 24 meses. No es un problema del orden: el VAR confunde reacción con causa
(el BanRep sube la tasa CUANDO ve venir inflación) y le faltan variables de expectativas.
Por eso NO usamos el impulso-respuesta para medir el efecto de la política monetaria;
el proyecto usa el VAR para PRONOSTICAR.""")
tabla_orden.to_csv(CARPETA / "sensibilidad_orden.csv")

# 3.6 Pronóstico puntual
pron = var.forecast(datos.values[-REZAGOS:], steps=12)
fechas_fut = pd.date_range(datos.index[-1] + pd.DateOffset(months=1), periods=12, freq="MS")
pron = pd.DataFrame(pron, index=fechas_fut, columns=COLS_VAR)
print("\nPronóstico puntual del VAR:")
print(pron.loc[:, ["inflacion_anual", "tasa_interes"]].iloc[[0, 2, 5, 11]].round(2).to_string())


# =====================================================================
# PARTE 4. CADENA DE MARKOV SOBRE EL CICLO DE TASAS  (C3)
# =====================================================================
titulo("PARTE 4. MARKOV: FASES DEL CICLO DE TASAS")
print("Fase de cada mes = cambio de la tasa en los últimos 3 meses:")
print("  <= -0,25 pts 'bajando' | > -0,25 y < +0,25 'estable' | >= +0,25 'subiendo'")


def matriz_transicion(estados):
    """Cuenta cuántas veces se pasó de cada fase a cada fase y lo vuelve probabilidad."""
    conteo = np.ones((3, 3)) * 0.01            # +0.01 evita divisiones por cero
    for a, b in zip(estados[:-1], estados[1:]):
        conteo[a, b] += 1
    return conteo / conteo.sum(axis=1, keepdims=True)


codigo_fase = base["fase_ciclo_tasa"].map({f: i for i, f in enumerate(FASES)}).values
assert np.isin(codigo_fase, [0, 1, 2]).all(), (
    "Hay fases no reconocidas en fase_ciclo_tasa. Revisar las categorías de la base de datos.")
# Las 273 fases son válidas, incluidas las de ene-mar 2004: la base se construyó con datos
# desde 2003 (hoja Datos_originales), así que ene-2004 se compara con oct-2003, feb-2004
# con nov-2003 y mar-2004 con dic-2003. Por eso no se excluye ningún mes.
# (Prueba de robustez hecha: excluir esos 3 meses da P(baja) 35% y P(sube) 17%,
# la misma decisión ESCALONAR.)
P = matriz_transicion(codigo_fase)
print("\nMatriz de transición (%): fila = fase de este mes, columna = fase del próximo mes")
print(pd.DataFrame((100 * P).round(1), index=FASES, columns=FASES).to_string())

# Persistencia según cómo se define la fase
filas = []
for meses, umbral in [(3, 0.25), (1, 0.10), (1, 0.25)]:
    # round(4): igual que la fórmula del Excel (evita que 0,25 se lea como 0,2499999)
    cambio = (datos["tasa_interes"] - datos["tasa_interes"].shift(meses)).dropna().round(4)
    s = np.where(cambio <= -umbral, 0, np.where(cambio >= umbral, 2, 1))
    Pm = matriz_transicion(s)
    filas.append({"definición": f"cambio {meses}m, umbral {umbral}",
                  "seguir bajando %": 100 * Pm[0, 0], "seguir estable %": 100 * Pm[1, 1],
                  "seguir subiendo %": 100 * Pm[2, 2], "subiendo->bajando %": 100 * Pm[2, 0]})
tabla_pers = pd.DataFrame(filas).set_index("definición").round(0)
print("\nPersistencia según la definición de fase:")
print(tabla_pers.to_string())
print("""Lectura: con la ventana de 3 meses la diagonal de la matriz de transición puede ser
alta en parte por construcción, debido a que dos meses consecutivos comparten 2 de los 3
meses utilizados para calcular el cambio de la tasa.
Al utilizar cambios de un mes, la persistencia disminuye. Por tanto, los resultados
muestran persistencia en los estados construidos a partir de la tasa interbancaria, pero
su magnitud depende de la definición utilizada.
No se interpreta esta persistencia como una demostración directa de la inercia de las
decisiones del Banco de la República.""")

print("\nDuración promedio de cada fase = 1 / (1 - prob. de quedarse)  [DESCRIPTIVA]:")
for i, f in enumerate(FASES):
    print(f"  {f:9s} {1 / (1 - P[i, i]):.1f} meses")

fase_hoy = codigo_fase[-1]
prob_markov = np.linalg.matrix_power(P, HORIZONTE)[fase_hoy]
print(f"\nFase actual ({base.index[-1]:%Y-%m}): {FASES[fase_hoy].upper()}")
for h in [3, 6]:
    prob = np.linalg.matrix_power(P, h)[fase_hoy]
    print(f"  Fase dentro de {h} meses: " + ", ".join(f"{FASES[i]} {100 * prob[i]:.0f}%" for i in range(3)))
print("Nota: Markov dice en qué FASE estará el ciclo, no CUÁNTO se moverá la tasa. Por eso")
print("se reporta aparte y no se mezcla con la probabilidad de Monte Carlo (ver Parte 6).")


# =====================================================================
# PARTE 5. MONTE CARLO (VAR + fases de Markov)  (C2)
# =====================================================================
titulo(f"PARTE 5. MONTE CARLO: {N_SIM} trayectorias de {HORIZONTE} meses")


def simular(var_mod, historia, fases_hist, P_mark, fase_inicial, n_sim, horizonte, rng,
            devolver_fases=False, bloque=BLOQUE):
    """
    Simula trayectorias futuras:
      1) El VAR da el valor esperado del próximo mes.
      2) Se le suma una 'sorpresa' (residuo real). Al inicio de cada bloque se
         selecciona un mes histórico cuya fase del mes anterior coincide con la fase
         actual de la trayectoria. Después se incorporan los residuos de los meses
         siguientes de manera consecutiva. Durante el bloque, la fase simulada puede
         cambiar, pero los residuos continúan perteneciendo al bloque histórico
         seleccionado (block bootstrap). Con bloque=1 son residuos sueltos.
      3) Markov decide la fase del mes siguiente.
    Devuelve un arreglo (n_sim, horizonte, 4) y, si se pide, las fases simuladas.
    """
    A, c = var_mod.coefs, var_mod.intercept
    p = A.shape[0]
    resid = var_mod.resid.values               # residuo k corresponde al mes p+k
    fase_resid = fases_hist[p - 1:-1]          # fase del mes ANTERIOR a cada residuo
    assert len(fase_resid) == len(resid)
    ultimo_inicio = len(resid) - bloque          # el bloque completo debe caber en la historia
    bolsas = [np.where((fase_resid == k) & (np.arange(len(resid)) <= ultimo_inicio))[0] for k in range(3)]
    bolsas = [b if len(b) > 5 else np.arange(ultimo_inicio + 1) for b in bolsas]

    rezagos = np.repeat(historia[-p:][None, :, :], n_sim, axis=0)
    fase = np.full(n_sim, fase_inicial)
    salida = np.zeros((n_sim, horizonte, historia.shape[1]))
    fases_sim = np.zeros((n_sim, horizonte), dtype=int)
    idx = np.zeros(n_sim, dtype=int)
    for h in range(horizonte):
        if h % bloque == 0:                       # empieza un bloque nuevo
            for k in range(3):
                m = fase == k
                idx[m] = rng.choice(bolsas[k], size=m.sum())
        else:                                     # sigue el bloque: mes siguiente
            idx = idx + 1
        y = c + sum(rezagos[:, -(j + 1)] @ A[j].T for j in range(p)) + resid[idx]
        salida[:, h] = y
        rezagos = np.concatenate([rezagos[:, 1:], y[:, None]], axis=1)
        u = rng.random(n_sim)
        fase = (u[:, None] > P_mark[fase].cumsum(axis=1)).sum(axis=1)
        fases_sim[:, h] = fase
    # Control: ninguna trayectoria puede tener valores infinitos o NaN. Si las
    # advertencias de 'matmul' fueran reales, el programa se detendría aquí.
    assert np.isfinite(salida).all(), "La simulación produjo valores no finitos: revisar"
    return (salida, fases_sim) if devolver_fases else salida


# Chequeo de alineación: ¿por qué la fase del mes ANTERIOR?
res_tasa = var.resid["tasa_interes"].values
fase_mismo = codigo_fase[REZAGOS:]
fase_ant = codigo_fase[REZAGOS - 1:-1]
print("Sorpresa promedio de la tasa (pts) según cómo se empareja con la fase:")
print(f"  {'fase':9s} {'mismo mes':>10s} {'mes anterior':>13s}")
for k in range(3):
    print(f"  {FASES[k]:9s} {res_tasa[fase_mismo == k].mean():+10.3f} {res_tasa[fase_ant == k].mean():+13.3f}")
print("""La fase de un mes se calcula con el cambio de tasa que INCLUYE ese mismo mes, es decir,
incluye la sorpresa que queremos simular. Emparejar con la fase del mismo mes es circular
(usa el resultado para elegir la sorpresa) y exagera las sorpresas. La fase del mes
anterior sí se conoce al momento de pronosticar: por eso la usamos.""")

rng = np.random.default_rng(2026)
sims, fases_sim = simular(var, datos.values, codigo_fase, P, fase_hoy, N_SIM, HORIZONTE, rng, devolver_fases=True)
tasa_hoy = datos["tasa_interes"].iloc[-1]
inf_hoy = datos["inflacion_anual"].iloc[-1]
tasa_fin = sims[:, -1, 1]
inf_fin = sims[:, -1, 0]

# Coherencia: fase de Markov al final vs fase que implica la tasa simulada
cambio_3m = sims[:, -1, 1] - sims[:, -4, 1]
fase_implicita = np.where(cambio_3m <= -0.25, 0, np.where(cambio_3m >= 0.25, 2, 1))
print(f"Coherencia: en el {100 * (fase_implicita == fases_sim[:, -1]).mean():.0f}% de las trayectorias la fase de Markov")
print("coincide con la que implica la tasa simulada. Markov y el VAR van por caminos")
print("distintos; por eso Markov solo elige el TIPO de sorpresa y no se promedia con el VAR.")

p_baja_mc = (tasa_fin <= tasa_hoy - UMBRAL_TASA).mean()
p_alza_mc = (tasa_fin >= tasa_hoy + UMBRAL_TASA).mean()
print(f"\nRemuestreo principal: bloques de {BLOQUE} meses")
print(f"Tasa hoy {tasa_hoy:.2f}%  |  Inflación hoy {inf_hoy:.2f}%")
print(f"Tasa en {HORIZONTE} meses: mediana {np.median(tasa_fin):.2f}%  rango 90% [{np.percentile(tasa_fin, 5):.2f}, {np.percentile(tasa_fin, 95):.2f}]")
print(f"Inflación en {HORIZONTE} meses: mediana {np.median(inf_fin):.2f}%  rango 90% [{np.percentile(inf_fin, 5):.2f}, {np.percentile(inf_fin, 95):.2f}]")
print(f"P(tasa baja >= {UMBRAL_TASA} pts) = {100 * p_baja_mc:.0f}%   P(tasa sube >= {UMBRAL_TASA} pts) = {100 * p_alza_mc:.0f}%   P(se queda) = {100 * (1 - p_baja_mc - p_alza_mc):.0f}%")
print(f"P(inflación en la meta, <= 4%) = {100 * (inf_fin <= 4).mean():.0f}%")

# Sensibilidad al tamaño del bloque
print("\nSensibilidad al remuestreo (mismas 5.000 trayectorias, otra forma de tomar las sorpresas):")
filas = []
for b in [1, 3, 6]:
    sb = simular(var, datos.values, codigo_fase, P, fase_hoy, N_SIM, HORIZONTE, np.random.default_rng(2026), bloque=b)
    tf, inf_b = sb[:, -1, 1], sb[:, -1, 0]
    filas.append({"remuestreo": "residuos sueltos" if b == 1 else f"bloques de {b} meses",
                  "P(baja) %": 100 * (tf <= tasa_hoy - UMBRAL_TASA).mean(),
                  "P(sube) %": 100 * (tf >= tasa_hoy + UMBRAL_TASA).mean(),
                  "tasa p5": np.percentile(tf, 5), "tasa p95": np.percentile(tf, 95),
                  "inflación p5": np.percentile(inf_b, 5), "inflación p95": np.percentile(inf_b, 95)})
tabla_bloques = pd.DataFrame(filas).set_index("remuestreo").round(1)
print(tabla_bloques.to_string())
tabla_bloques.to_csv(CARPETA / "sensibilidad_bloques.csv")
print("""Interpretación de sensibilidad: la recomendación de tesorería puede cambiar
dependiendo del método de remuestreo de los residuos. El remuestreo individual supone
una selección independiente de residuos, mientras que el remuestreo por bloques conserva
parte de la dependencia temporal entre observaciones consecutivas. Cuando las
recomendaciones obtenidas con estos métodos no coinciden, se considera que existe
sensibilidad del resultado al procedimiento de simulación. Por esta razón, se adopta la
estrategia escalonada como decisión prudencial, evitando depender de un único método.""")

fig, ejes = plt.subplots(1, 2, figsize=(12, 4.5))
for eje, k, nom in [(ejes[0], 0, "Inflación anual (%)"), (ejes[1], 1, "Tasa de interés (%)")]:
    hist = datos.iloc[-36:, k]
    eje.plot(hist.index, hist.values, color="black", lw=1.5, label="Histórico")
    fut = pd.date_range(datos.index[-1] + pd.DateOffset(months=1), periods=HORIZONTE, freq="MS")
    for lo, hi, a in [(5, 95, 0.15), (25, 75, 0.3)]:
        eje.fill_between(fut, np.percentile(sims[:, :, k], lo, axis=0),
                         np.percentile(sims[:, :, k], hi, axis=0), color="tab:blue", alpha=a)
    eje.plot(fut, np.median(sims[:, :, k], axis=0), color="tab:blue", lw=1.5, label="Mediana simulada")
    eje.set_title(nom)
    eje.grid(alpha=0.3)
    eje.legend(fontsize=8)
plt.tight_layout()
plt.savefig(CARPETA / "3_abanico_montecarlo.png", dpi=150)
plt.close()


# =====================================================================
# PARTE 6. RECOMENDACIÓN PARA LA TESORERÍA  (C1, C13)
# =====================================================================
titulo("PARTE 6. RECOMENDACIÓN")


def recomendar(p_baja, p_alza):
    if p_baja >= UMBRAL_PROB and p_baja > p_alza:
        return "AMARRAR"
    if p_alza >= UMBRAL_PROB and p_alza > p_baja:
        return "RENOVAR CORTO"
    return "ESCALONAR"


TEXTO = {
    "AMARRAR": "CDT a plazo largo (6+ meses) y crédito a tasa VARIABLE.",
    "RENOVAR CORTO": "CDT a 90 días y crédito a tasa FIJA.",
    "ESCALONAR": "ninguna señal domina; repartir plazos (parte corto, parte largo).",
}
print("La DECISIÓN usa solo Monte Carlo: P(tasa baja/sube al menos 0,5 pts en 6 meses).")
print(f"  P(baja) = {100 * p_baja_mc:.0f}%   P(sube) = {100 * p_alza_mc:.0f}%   regla: >= {100 * UMBRAL_PROB:.0f}% y mayor que la otra")
reco = recomendar(p_baja_mc, p_alza_mc)
print(f"\n>>> RECOMENDACIÓN: {reco}: {TEXTO[reco]}")
print(f"\nContexto (no decide): Markov ve el ciclo en '{FASES[fase_hoy]}' y en {HORIZONTE} meses lo ubica en")
print("  " + ", ".join(f"{FASES[i]} {100 * prob_markov[i]:.0f}%" for i in range(3)))
if reco == "ESCALONAR":
    print("  -> Monte Carlo no da señal clara; Markov solo describe la fase actual del ciclo.")
elif (reco == "AMARRAR" and prob_markov[2] > prob_markov[0]) or (reco == "RENOVAR CORTO" and prob_markov[0] > prob_markov[2]):
    print("  -> Las dos señales NO coinciden: la recomendación es de convicción BAJA.")
    print("     Sugerencia prudente: ejecutarla de forma parcial / escalonada y revisar el próximo mes.")
else:
    print("  -> Las dos señales apuntan en la misma dirección.")

# ¿La decisión depende de cómo se remuestrean los residuos?
decisiones = {r: recomendar(f["P(baja) %"] / 100, f["P(sube) %"] / 100) for r, f in tabla_bloques.iterrows()}
print("\nDecisión según el remuestreo:", ", ".join(f"{r}: {d}" for r, d in decisiones.items()))
if len(set(decisiones.values())) > 1:
    reco = "ESCALONAR"
    print("  -> La decisión CAMBIA según el remuestreo: la probabilidad está cerca del umbral de 40%.")
    print("     Por prudencia la decisión final es ESCALONAR (y volver a correr el modelo el próximo mes).")
print(f"\n>>> DECISIÓN FINAL: {reco}: {TEXTO[reco]}")

tasa_cdt = tasa_hoy + SPREAD_CDT
inf_esp = np.median(inf_fin)
real = 100 * ((1 + tasa_cdt / 100) / (1 + inf_esp / 100) - 1)
print(f"\nRentabilidad real esperada de un CDT (~{tasa_cdt:.2f}% nominal = tasa + {SPREAD_CDT} [supuesto],")
print(f"inflación esperada {inf_esp:.2f}%): {real:.2f}% anual")


# =====================================================================
# PARTE 7. VALIDACIÓN FUERA DE MUESTRA  (C5, C8, C9)
# =====================================================================
if CORRER_VALIDACION:
    titulo("PARTE 7. VALIDACIÓN HISTÓRICA (2011 en adelante, tarda 2-3 minutos)")
    dif = pd.DataFrame({"d_inf": datos["inflacion_anual"].diff(), "d_tasa": datos["tasa_interes"].diff(),
                        "trm": datos["var_anual_trm"], "brent": datos["var_anual_brent"]}).dropna()
    dif.index.freq = "MS"
    filas = []
    rng_v = np.random.default_rng(7)
    for t in datos.loc["2011-01-01":].index:
        i = datos.index.get_loc(t)
        if i + HORIZONTE >= len(datos):
            break
        d_hist = datos.iloc[:i + 1]
        f_hist = codigo_fase[:i + 1]
        v = VAR(d_hist).fit(REZAGOS)
        P_h = matriz_transicion(f_hist)
        s = simular(v, d_hist.values, f_hist, P_h, f_hist[-1], 1000, HORIZONTE, rng_v)
        s1 = simular(v, d_hist.values, f_hist, P_h, f_hist[-1], 1000, HORIZONTE, rng_v, bloque=1)
        hoy_t = d_hist["tasa_interes"].iloc[-1]
        pb = (s[:, -1, 1] <= hoy_t - UMBRAL_TASA).mean()
        pa = (s[:, -1, 1] >= hoy_t + UMBRAL_TASA).mean()
        pm_h = np.linalg.matrix_power(P_h, HORIZONTE)[f_hist[-1]]
        pron_n = v.forecast(d_hist.values[-REZAGOS:], HORIZONTE)[-1]
        hd = dif.loc[:t]
        pron_d = VAR(hd).fit(REZAGOS).forecast(hd.values[-REZAGOS:], HORIZONTE)
        filas.append({
            "fecha": t,
            "tasa_hoy": hoy_t,
            "tasa_en_3m": datos["tasa_interes"].iloc[i + 3],
            "cambio_real_tasa": datos["tasa_interes"].iloc[i + HORIZONTE] - hoy_t,
            "p_baja": pb, "p_alza": pa,
            "decision": recomendar(pb, pa),
            "markov_bajando": pm_h[0], "markov_subiendo": pm_h[2],
            "señal_modelo": np.sign(pa - pb),
            "señal_modelo_sueltos": np.sign((s1[:, -1, 1] >= hoy_t + UMBRAL_TASA).mean() - (s1[:, -1, 1] <= hoy_t - UMBRAL_TASA).mean()),
            "señal_tendencia": np.sign(hoy_t - d_hist["tasa_interes"].iloc[-4]),
            "tasa_pron_niveles": pron_n[1],
            "tasa_pron_dif": hoy_t + pron_d[:, 1].sum(),
            "inf_pron_niveles": pron_n[0],
            "inf_pron_dif": d_hist["inflacion_anual"].iloc[-1] + pron_d[:, 0].sum(),
            "inf_hoy": d_hist["inflacion_anual"].iloc[-1],
            "inf_real": datos["inflacion_anual"].iloc[i + HORIZONTE],
        })
    val = pd.DataFrame(filas).set_index("fecha")
    val["tasa_real"] = val["tasa_hoy"] + val["cambio_real_tasa"]

    # --- 7.1 Dirección de la tasa: modelo vs regla de tendencia (C8) ---
    grandes = val[val["cambio_real_tasa"].abs() >= UMBRAL_TASA]
    dir_real = np.sign(grandes["cambio_real_tasa"])
    acierto_mod = (grandes["señal_modelo"] == dir_real).mean()
    acierto_ten = (grandes["señal_tendencia"] == dir_real).mean()
    print(f"Pronósticos evaluados: {len(val)} (uno por mes)  |  movimientos de tasa >= {UMBRAL_TASA} pts: {len(grandes)}")
    print(f"Las ventanas de {HORIZONTE} meses se TRASLAPAN: meses seguidos comparten casi todo su futuro,")
    print(f"así que hay aprox. {len(val) // HORIZONTE} pronósticos realmente independientes, no {len(val)}.")
    ok_mod = grandes["señal_modelo"] == dir_real
    ok_ten = grandes["señal_tendencia"] == dir_real
    print(f"\nDirección en los {len(grandes)} movimientos grandes:   acertados  fallidos  acierto")
    print(f"  Modelo (bloques de {BLOQUE} meses)           {ok_mod.sum():5d}  {(~ok_mod).sum():8d}  {100 * ok_mod.mean():6.0f}%")
    ok_s = grandes["señal_modelo_sueltos"] == dir_real
    print(f"  Modelo (residuos sueltos)            {ok_s.sum():5d}  {(~ok_s).sum():8d}  {100 * ok_s.mean():6.0f}%")
    print(f"  Regla 'sigue la tendencia de 3m'     {ok_ten.sum():5d}  {(~ok_ten).sum():8d}  {100 * ok_ten.mean():6.0f}%")
    # Prueba de McNemar: compara dos métodos sobre LOS MISMOS casos
    from statsmodels.stats.contingency_tables import mcnemar
    solo_mod = int((ok_mod & ~ok_ten).sum())
    solo_ten = int((~ok_mod & ok_ten).sum())
    p_mc = mcnemar([[int((ok_mod & ok_ten).sum()), solo_mod], [solo_ten, int((~ok_mod & ~ok_ten).sum())]], exact=True).pvalue
    print(f"McNemar modelo vs tendencia: el modelo acierta y la tendencia falla en {solo_mod} casos;")
    print(f"al revés en {solo_ten}. p = {p_mc:.2f} -> {'diferencia significativa' if p_mc < 0.05 else 'NO hay diferencia significativa'}.")
    print("(Y como las ventanas se traslapan, este p-valor es incluso optimista.)")

    # Señales de la REGLA DE DECISIÓN (40%): falsas, acertadas y movimientos no anticipados
    senal = val["decision"].map({"AMARRAR": -1, "RENOVAR CORTO": 1, "ESCALONAR": 0})
    real_dir = np.sign(val["cambio_real_tasa"]).where(val["cambio_real_tasa"].abs() >= UMBRAL_TASA, 0)
    con_senal = senal != 0
    print(f"\nRegla de decisión (umbral {100 * UMBRAL_PROB:.0f}%): dio señal en {con_senal.sum()} de {len(val)} meses")
    print(f"  Señal correcta (la tasa se movió >= 0,5 en esa dirección): {(con_senal & (senal == real_dir)).sum()}")
    print(f"  Señal falsa: la tasa se movió al REVÉS:                   {(con_senal & (senal == -real_dir) & (real_dir != 0)).sum()}")
    print(f"  Señal falsa: la tasa casi no se movió:                    {(con_senal & (real_dir == 0)).sum()}")
    print(f"  Movimientos grandes NO anticipados (escalonar):           {((senal == 0) & (real_dir != 0)).sum()} de {(real_dir != 0).sum()}")

    # ¿Qué pasa cuando Monte Carlo y Markov no coinciden?
    señal_markov = np.sign(val["markov_subiendo"] - val["markov_bajando"])
    coincide = con_senal & (senal == señal_markov)
    choca = con_senal & (senal == -señal_markov)
    for nombre, m in [("Monte Carlo y Markov COINCIDEN", coincide), ("Monte Carlo y Markov CHOCAN  ", choca)]:
        if m.sum():
            print(f"  {nombre}: {m.sum():3d} señales, correctas {100 * (senal[m] == real_dir[m]).mean():.0f}%, "
                  f"al revés {100 * ((senal[m] == -real_dir[m]) & (real_dir[m] != 0)).mean():.0f}%")

    # Subperiodos
    print("\nPor subperiodos (acierto en movimientos grandes):")
    sub = []
    for nombre, a, b in [("2011-2015", "2011", "2015"), ("2016-2020", "2016", "2020"), ("2021-2026", "2021", "2026")]:
        g = grandes.loc[a:b]
        if len(g):
            dr = np.sign(g["cambio_real_tasa"])
            sub.append({"periodo": nombre, "movimientos": len(g),
                        "modelo %": 100 * (g["señal_modelo"] == dr).mean(),
                        "tendencia %": 100 * (g["señal_tendencia"] == dr).mean()})
    print(pd.DataFrame(sub).set_index("periodo").round(0).to_string())
    print("Lectura: el modelo y la regla de tendencia aciertan parecido. NO afirmamos que el modelo")
    print("sea superior en dirección; su aporte es la probabilidad, el rango y el pronóstico de inflación.")

    # --- 7.2 Errores: niveles vs diferencias vs ingenuo (C5) ---
    print(f"\nError absoluto medio a {HORIZONTE} meses (puntos):")
    errores = pd.DataFrame({
        "VAR niveles": [(val["inf_pron_niveles"] - val["inf_real"]).abs().mean(), (val["tasa_pron_niveles"] - val["tasa_real"]).abs().mean()],
        "VAR diferencias": [(val["inf_pron_dif"] - val["inf_real"]).abs().mean(), (val["tasa_pron_dif"] - val["tasa_real"]).abs().mean()],
        "'se queda igual'": [(val["inf_hoy"] - val["inf_real"]).abs().mean(), val["cambio_real_tasa"].abs().mean()],
    }, index=["inflación", "tasa"]).round(2)
    print(errores.to_string())
    print("""Lectura: la comparación fuera de muestra muestra que el VAR en niveles presenta menor
error para el pronóstico de inflación, mientras que el VAR en diferencias presenta menor
error para la tasa de interés. Por tanto, no existe una especificación superior para todas
las variables. Se mantiene el VAR en niveles como modelo integrado principal para estudiar
conjuntamente la inflación y las tasas, pero se reconoce que el VAR en diferencias ofrece
mejor desempeño en el pronóstico de la tasa durante el periodo de validación.
Esta decisión es un compromiso entre el análisis conjunto de las variables y la precisión
predictiva individual. No se afirma que el VAR en niveles sea superior en todos los
pronósticos ni que Johansen haya demostrado cointegración.""")

    # --- 7.3 Estrategias de CDT (C9): EJERCICIO ILUSTRATIVO ---
    # No usa la serie histórica de CDT a 90 días (esa serie solo sirvió para estimar el
    # spread de 0,28). Aquí la tasa de un CDT = tasa interbancaria + SPREAD_CDT.
    # Lo pidió el plan de correcciones (C9 y punto 13) y lo cita la ficha (Alcance e Integración).
    # Supuestos: CDT = tasa interbancaria + SPREAD_CDT para cualquier plazo (sin prima por plazo),
    # sin impuestos ni comisiones; rendimiento de 6 meses en puntos porcentuales anualizados.
    fija = val["tasa_hoy"] + SPREAD_CDT                                    # amarrar 6 meses
    renov = (val["tasa_hoy"] + val["tasa_en_3m"]) / 2 + SPREAD_CDT          # dos CDT de 90 días
    escal = (fija + renov) / 2                                              # mitad y mitad
    modelo_estr = np.select([val["decision"] == "AMARRAR", val["decision"] == "RENOVAR CORTO"], [fija, renov], escal)
    estr = pd.DataFrame({"siempre amarrar": fija, "siempre renovar": renov,
                         "siempre escalonar": escal, "según el modelo": modelo_estr}, index=val.index)
    resumen = pd.DataFrame({
        "tasa media %": estr.mean(),
        "vs escalonar (pts)": estr.mean() - escal.mean(),
        f"COP por año sobre {MONTO / 1e6:.0f} M": (estr.mean() - escal.mean()) / 100 * MONTO,
    }).round(3)
    print("\nComparación de estrategias de CDT (2011 en adelante, supuestos en el código):")
    print(resumen.to_string())
    print("Ojo: la diferencia es pequeña y la evaluación usa la tasa interbancaria como referencia,")
    print("no las tasas que ofrece cada banco. Es una comparación ilustrativa, no una garantía.")

    val.to_csv(CARPETA / "validacion.csv")
    errores.to_csv(CARPETA / "errores_pronostico.csv")
    resumen.to_csv(CARPETA / "estrategias_cdt.csv")

titulo("ADVERTENCIAS REGISTRADAS DURANTE LA EJECUCIÓN")
if REGISTRO_AVISOS:
    print(pd.Series(REGISTRO_AVISOS).value_counts().to_string())
    print("""Cómo leerlas:
  - InterpolationWarning (KPSS): el p-valor real está fuera de la tabla (p > 0,10 o
    p < 0,01); statsmodels reporta el límite. No es un error: se lee 'p >= 0,10'.
  - ValueWarning de fechas/frecuencia: solo informa que no encontró la frecuencia; la
    base se verificó en la Parte 1 como mensual y consecutiva.
  - ComplexWarning (Johansen): al calcular los valores propios aparecen partes imaginarias
    de redondeo. No se ignora: en cada ejecución la Parte 2 recalcula la traza con valores
    propios garantizados reales y el programa se DETIENE si no coincide. Si llegó hasta
    aquí, la comprobación pasó y la advertencia es solo numérica.
  - FutureWarning: statsmodels avisa que cambiará el formato de salida de adfuller, kpss
    y het_arch en versiones futuras. No afecta los resultados de esta versión.
  - RuntimeWarning de 'matmul' (divide by zero / overflow / invalid value): apareció al
    ejecutar en Mac y no en Linux. No se da por irrelevante: cada simulación verifica que
    todos sus valores sean finitos y el programa se DETIENE si no lo son. Además, en nuestra
    comparación (mismo código, misma base y misma semilla) todas las cifras impresas
    coincidieron entre el Mac y Linux. Si aparece con otros datos o en otro equipo, se debe
    revisar su origen y comprobar que no afecte los resultados.
  - PyparsingDeprecationWarning: viene de matplotlib (al dibujar los gráficos) por usar
    nombres antiguos de la librería pyparsing. No afecta ningún cálculo.
  - Cualquier otra (convergencia, matrices singulares) SÍ debe revisarse.""")
else:
    print("Ninguna.")

print("\nListo. Revisa los gráficos y tablas en la carpeta 'resultados/'.")
