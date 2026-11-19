"""
Radar de inflación y tasas para tesorería
Aplicación Streamlit del proyecto integrador (Análisis e Interpretación de Datos a través de Python).
Ejecutar:  streamlit run app.py
"""
from pathlib import Path

import altair as alt
import numpy as np
import pandas as pd
import streamlit as st

import motor as m

st.set_page_config(page_title="Radar de inflación y tasas para tesorería", page_icon="📈", layout="wide")

CARPETA_DATOS = Path("datos")
CARPETA_RESULTADOS = Path("resultados")
NOMBRES = {"inflacion_anual": "Inflación anual (%)", "tasa_interes": "Tasa de interés interbancaria (%)",
           "var_anual_trm": "Cambio anual del dólar – TRM (%)", "var_anual_brent": "Cambio anual del petróleo Brent (%)"}
MESES_ES = ["enero", "febrero", "marzo", "abril", "mayo", "junio", "julio", "agosto",
            "septiembre", "octubre", "noviembre", "diciembre"]
COLOR_RECO = {"AMARRAR": "🟢", "RENOVAR CORTO": "🟠", "ESCALONAR": "🔵"}
QUE_HACER = {
    "AMARRAR": "invertir en **CDT a plazo largo** (6 meses o más) para asegurar la tasa de hoy, porque es probable que baje.",
    "RENOVAR CORTO": "invertir en **CDT cortos** (90 días) para poder renovar a una tasa más alta, porque es probable que suba.",
    "ESCALONAR": "**repartir la plata** entre CDT cortos y largos, porque ninguna dirección de la tasa es lo bastante clara.",
}
SECCIONES = {
    "🏠 Inicio": "Qué hace la herramienta y qué recomienda hoy",
    "🔮 Pronosticar próximos meses": "Escribe datos nuevos y mira qué se espera para 2027",
    "📊 Los datos": "Las cuatro variables y de dónde salen",
    "🎲 Monte Carlo: 5.000 futuros": "De dónde salen las probabilidades",
    "🔁 Markov: etapa del ciclo": "¿Las tasas vienen subiendo, quietas o bajando?",
    "📈 VAR: cómo se mueven juntas": "El modelo que hace los pronósticos",
    "✅ ¿Qué tan confiable es?": "Prueba con 15 años de historia y conclusiones",
}


def mes(fecha):
    """Nombre del mes en español, por ejemplo 'enero 2027'."""
    return f"{MESES_ES[fecha.month - 1]} {fecha.year}"


def en_pocas_palabras(texto):
    st.info("**En pocas palabras.** " + texto, icon="💡")


# ---------------------------------------------------------------------------
# Datos y modelo (en caché para que la app sea rápida)
# ---------------------------------------------------------------------------
@st.cache_data
def base_oficial():
    return m.cargar_base(CARPETA_DATOS)


@st.cache_data
def originales():
    return m.cargar_originales(CARPETA_DATOS)


@st.cache_resource(show_spinner="Calculando: el modelo está simulando 5.000 futuros posibles...")
def modelo(base, horizonte, umbral_tasa):
    return m.correr_modelo(base, horizonte=horizonte, umbral_tasa=umbral_tasa)


def grafico_abanico(historia, fan, titulo):
    """Datos reales (línea negra) + valor esperado (línea azul) y rangos probables (bandas)."""
    hist = historia.rename("valor").reset_index()
    hist.columns = ["fecha", "valor"]
    f = fan.reset_index().rename(columns={"index": "fecha"})
    eje_x = alt.X("fecha:T", title=None, axis=alt.Axis(format="%m/%Y"))
    base = alt.Chart(f).encode(x=eje_x)
    banda90 = base.mark_area(opacity=0.18, color="#1f77b4").encode(y=alt.Y("p5:Q", title="%"), y2="p95:Q")
    banda50 = base.mark_area(opacity=0.35, color="#1f77b4").encode(y="p25:Q", y2="p75:Q")
    mediana = base.mark_line(color="#1f77b4", strokeWidth=2.5).encode(
        y="mediana:Q", tooltip=[alt.Tooltip("fecha:T", title="Mes", format="%m/%Y"),
                                alt.Tooltip("mediana:Q", title="Valor esperado", format=".2f"),
                                alt.Tooltip("p5:Q", title="Escenario bajo", format=".2f"),
                                alt.Tooltip("p95:Q", title="Escenario alto", format=".2f")])
    linea = alt.Chart(hist).mark_line(color="black").encode(
        x=eje_x, y=alt.Y("valor:Q", scale=alt.Scale(zero=False)),
        tooltip=[alt.Tooltip("fecha:T", title="Mes", format="%m/%Y"), alt.Tooltip("valor:Q", title="Dato real", format=".2f")])
    return (banda90 + banda50 + linea + mediana).properties(title=titulo, height=320)


def mostrar_grafico(lugar, grafico):
    """Muestra un gráfico Altair a todo el ancho; compatible con Streamlit 1.50 (Mac con
    Python 3.9) y con las versiones nuevas (Streamlit Cloud)."""
    try:
        lugar.altair_chart(grafico, width="stretch")
    except TypeError:
        lugar.altair_chart(grafico, use_container_width=True)


def como_leer_grafico():
    st.caption("**Cómo leer el gráfico:** la línea **negra** son los datos reales. La línea **azul** es el valor "
               "esperado (el del medio de los 5.000 futuros simulados). La banda **oscura** es donde cae la mitad de "
               "los futuros, y la banda **clara**, 9 de cada 10. Mientras más lejos en el tiempo, más ancha la banda: "
               "hay más incertidumbre.")


def tarjeta_decision(r, umbral_tasa, horizonte):
    reco = r["reco_final"]
    with st.container(border=True):
        st.markdown(f"### {COLOR_RECO[reco]} Recomendación de hoy: **{reco}**")
        st.markdown(f"Lo que conviene: {QUE_HACER[reco]}")
        c1, c2, c3 = st.columns(3)
        c1.metric(f"Chance de que la tasa BAJE ≥ {umbral_tasa} pts", f"{100 * r['p_baja']:.0f}%")
        c2.metric(f"Chance de que la tasa SUBA ≥ {umbral_tasa} pts", f"{100 * r['p_alza']:.0f}%")
        c3.metric(f"Tasa de interés hoy ({mes(r['fecha_hoy'])})", f"{r['tasa_hoy']:.2f}%")
        st.caption(f"**Por qué:** en los próximos {horizonte} meses, de cada 100 futuros simulados la tasa baja en "
                   f"{100 * r['p_baja']:.0f} y sube en {100 * r['p_alza']:.0f}. La regla del proyecto: actuar solo si "
                   "una de las dos llega al **40%** y supera a la otra.")
        if r["fragil"]:
            st.warning("**Señal débil:** la recomendación cambia según el método con que se simulan los futuros "
                       "(la probabilidad está cerca del 40%). Por prudencia se recomienda **escalonar** y volver a "
                       "revisar el mes siguiente.")
        if horizonte != 6:
            st.caption("La regla del proyecto está pensada a **6 meses**; con otro horizonte, tómalo como orientación.")


# ---------------------------------------------------------------------------
# Barra lateral
# ---------------------------------------------------------------------------
st.sidebar.title("📈 Radar de inflación y tasas")
st.sidebar.caption("Proyecto integrador · Diego Alejandro Martínez Piñeros y María Juliana Brand Ardila")
seccion = st.sidebar.radio("Ir a:", list(SECCIONES), captions=list(SECCIONES.values()))
st.sidebar.divider()
st.sidebar.markdown("**Ajustes del modelo**")
horizonte = st.sidebar.select_slider("¿A cuántos meses mirar?", options=[3, 6, 12], value=6,
                                     help="Cuántos meses hacia adelante se simulan. El proyecto usa 6.")
umbral_tasa = st.sidebar.select_slider("¿Qué movimiento de la tasa es importante? (puntos)", options=[0.25, 0.5, 1.0],
                                       value=0.5, help="Se cuenta como 'baja' o 'sube' solo si la tasa se mueve al "
                                                       "menos esto. El proyecto usa 0,5 puntos.")
st.sidebar.caption("Valores del proyecto: 6 meses y 0,5 puntos.")
with st.sidebar.expander("📖 Glosario"):
    st.markdown(
        "- **CDT:** certificado de depósito a término; dejas plata en el banco un plazo fijo y te paga una tasa.\n"
        "- **Tasa interbancaria:** la tasa a la que se prestan plata los bancos entre sí por un día. Se mueve casi "
        "igual que la tasa del Banco de la República y que los CDT.\n"
        "- **Inflación anual:** cuánto subieron los precios frente al mismo mes del año anterior.\n"
        "- **TRM:** precio del dólar en pesos. **Brent:** precio internacional del barril de petróleo.\n"
        "- **Punto porcentual (pts):** pasar de 12% a 12,5% es subir 0,5 puntos.\n"
        "- **Amarrar:** CDT largo para asegurar la tasa de hoy. **Renovar corto:** CDT de 90 días. "
        "**Escalonar:** repartir entre cortos y largos.\n"
        "- **VAR:** modelo que aprende cómo se mueven juntas las cuatro variables y con eso proyecta.\n"
        "- **Markov:** clasifica en qué etapa está el ciclo de tasas (bajando, quieta o subiendo).\n"
        "- **Monte Carlo:** simula miles de futuros posibles y cuenta qué pasa en cada uno.\n"
        "- **Rango probable:** entre qué valores cae el resultado en 9 de cada 10 futuros simulados.")

base = base_oficial()
r = modelo(base, horizonte, umbral_tasa)
datos = base[m.COLS_VAR]

# ---------------------------------------------------------------------------
# INICIO
# ---------------------------------------------------------------------------
if seccion == "🏠 Inicio":
    st.title("Radar de inflación y tasas para tesorería")
    st.markdown("##### Una herramienta para decidir **a qué plazo invertir en CDT**, según hacia dónde van la inflación "
                "y las tasas de interés en Colombia.")
    c1, c2, c3 = st.columns(3)
    with c1.container(border=True):
        st.markdown("**🎯 ¿Qué pregunta responde?**\n\n¿Las tasas de interés van a **bajar**, **subir** o "
                    "**quedarse igual** en los próximos meses? ¿Y qué me conviene hacer con mi plata en CDT?")
    with c2.container(border=True):
        st.markdown("**👥 ¿Para quién es?**\n\nPara la **tesorería de una empresa** (o cualquier persona) que tiene "
                    "plata para invertir en CDT y debe elegir entre un plazo corto o largo.")
    with c3.container(border=True):
        st.markdown("**📦 ¿Qué te entrega?**\n\nLa **probabilidad** de que la tasa baje o suba, el **pronóstico** de "
                    "la tasa y la inflación mes a mes y una **recomendación** clara.")

    st.subheader("La respuesta de hoy")
    tarjeta_decision(r, umbral_tasa, horizonte)
    with st.expander("¿Qué significa cada recomendación?"):
        st.markdown(
            "| Recomendación | Cuándo | Qué hacer | Ejemplo con 200 millones |\n|---|---|---|---|\n"
            "| 🟢 **Amarrar** | Es probable que la tasa **baje** | CDT a plazo largo | Los 200 M a 6 meses o más, a la tasa alta de hoy |\n"
            "| 🟠 **Renovar corto** | Es probable que la tasa **suba** | CDT a 90 días | Los 200 M a 90 días, y renovar a una tasa mejor |\n"
            "| 🔵 **Escalonar** | No está claro | Repartir | 100 M a 90 días y 100 M a 6 meses |")

    st.subheader("¿Cómo usar esta página?")
    st.markdown(
        "1. **Mira la recomendación de hoy** (arriba).\n"
        "2. **Ve a 🔮 Pronosticar próximos meses** para ver qué se espera en 2027. Ya trae un ejemplo y puedes cambiar los "
        "datos (por ejemplo, el precio del petróleo) para ver cómo cambia el resultado.\n"
        "3. **Si quieres entender cómo se calcula**, revisa las secciones de Monte Carlo, Markov y VAR.\n"
        "4. **En ✅ ¿Qué tan confiable es?** está la prueba del modelo con 15 años de historia.")

    st.subheader(f"Qué se espera en los próximos {horizonte} meses")
    c1, c2 = st.columns(2)
    mostrar_grafico(c1, grafico_abanico(datos["inflacion_anual"].iloc[-36:], r["fan_inflacion"], "Inflación anual (%)"))
    mostrar_grafico(c2, grafico_abanico(datos["tasa_interes"].iloc[-36:], r["fan_tasa"], "Tasa de interés (%)"))
    como_leer_grafico()
    st.caption("Datos: DANE y FRED, mensuales, de 2004 a septiembre de 2026. Herramienta académica de apoyo a la decisión; "
               "no es una recomendación de inversión.")

# ---------------------------------------------------------------------------
# PRONOSTICAR PRÓXIMOS MESES
# ---------------------------------------------------------------------------
elif seccion == "🔮 Pronosticar próximos meses":
    orig = originales()
    ultimo = base.index[-1]
    meses = pd.date_range(ultimo + pd.DateOffset(months=1), periods=3, freq="MS")

    st.title("🔮 Pronosticar los próximos meses")
    en_pocas_palabras(
        f"Nuestros datos llegan hasta **{mes(ultimo)}**. Aquí agregas los meses que siguen y el modelo te dice qué "
        "tasa de interés e inflación esperar más adelante, y qué conviene hacer con los CDT. **Ya viene un ejemplo "
        "cargado**: baja hasta el resultado para verlo, o cambia cualquier número y todo se recalcula solo.")
    with st.expander("¿Cómo funciona por dentro?"):
        st.markdown(
            "1. Para proyectar, el modelo necesita los **últimos 3 meses** de cuatro datos: inflación, tasa de interés, "
            "precio del dólar y precio del petróleo.\n"
            "2. Con los datos que escribas, el modelo **se vuelve a ajustar** y simula **5.000 futuros posibles** para "
            "los meses siguientes.\n"
            "3. El **valor esperado** es el del medio de esos 5.000 futuros; el **rango probable** es donde caen 9 de cada 10.\n\n"
            "**Ejemplo:** si subes el precio del petróleo, el modelo entiende que se encarecen el transporte y la "
            "energía; eso suele empujar la inflación hacia arriba y, con ella, las tasas.")

    # ---- Valores de ejemplo (octubre 2026 = datos reales publicados; nov y dic = ejemplo) ----
    ejemplo = {}
    if meses[0] in orig.index:
        o = orig.loc[meses[0]]
        ipc_12 = orig.loc[meses[0] - pd.DateOffset(months=12), "ipc_indice"]
        ejemplo[meses[0]] = [round(100 * (o["ipc_indice"] / ipc_12 - 1), 2), float(o["tasa_interbancaria"]),
                             float(o["trm_promedio"]), 100.0]
    ult = [float(base["inflacion_anual"].iloc[-1]), float(base["tasa_interes"].iloc[-1]),
           float(orig["trm_promedio"].dropna().iloc[-1]), float(orig["brent_usd"].dropna().iloc[-1])]
    for k, fecha in enumerate(meses):
        if fecha not in ejemplo:
            prev = ejemplo.get(meses[k - 1], ult) if k > 0 else ult
            ejemplo[fecha] = [round(prev[0] - 0.1, 2), prev[1], prev[2], prev[3]]

    st.subheader("Paso 1 · Escribe los datos de los últimos meses")
    c_n, c_h, c_b = st.columns([2, 2, 1])
    n_meses = c_n.radio("¿Cuántos meses nuevos vas a escribir?", [1, 2, 3], index=2, horizontal=True,
                        help="Con 3 meses (octubre a diciembre de 2026) el pronóstico empieza en enero de 2027.")
    h_local = c_h.radio("¿Hasta dónde quieres pronosticar?", [6, 12], horizontal=True,
                        format_func=lambda n: f"{n} meses")
    if c_b.button("↺ Volver al ejemplo", help="Borra tus cambios y vuelve a los valores de ejemplo"):
        for clave in list(st.session_state):
            if str(clave).startswith("dato_"):
                del st.session_state[clave]
        st.rerun()

    columnas = st.columns(n_meses)
    filas = {}
    for col, fecha in zip(columnas, meses[:n_meses]):
        with col.container(border=True):
            real = fecha == meses[0] and meses[0] in orig.index
            st.markdown(f"#### {mes(fecha).capitalize()}")
            st.caption("✅ Datos reales publicados (el Brent es de ejemplo)" if real else "✏️ Valores de ejemplo: cámbialos por los reales")
            e = ejemplo[fecha]
            clave = fecha.strftime("%Y%m")
            inf = st.number_input("📈 Inflación anual (%)", -2.0, 25.0, e[0], 0.05, key=f"dato_inf_{clave}",
                                  help="Cuánto subieron los precios frente al mismo mes del año anterior. Lo publica "
                                       "el DANE cada mes (boletín del IPC).")
            tasa = st.number_input("🏦 Tasa de interés interbancaria (%)", 0.0, 25.0, e[1], 0.05, key=f"dato_tasa_{clave}",
                                   help="Promedio del mes de la tasa a la que se prestan los bancos. Fuente: FRED, "
                                        "serie COLIRSTCI01STM.")
            trm = st.number_input("💵 Precio del dólar – TRM ($)", 1000.0, 10000.0, e[2], 10.0, key=f"dato_trm_{clave}",
                                  help="Pesos por dólar, promedio del mes.")
            brent = st.number_input("🛢️ Petróleo Brent (US$ por barril)", 10.0, 250.0, e[3], 1.0, key=f"dato_brent_{clave}",
                                    help="Precio promedio del barril de petróleo en el mes (FRED, POILBREUSDM).")
            filas[fecha] = {"inflacion_anual": inf, "tasa_interes": tasa, "trm": trm, "brent": brent}

    for fecha, f in filas.items():
        if not 0.7 * ult[2] <= f["trm"] <= 1.3 * ult[2]:
            st.warning(f"{mes(fecha)}: el dólar ({f['trm']:,.0f}) está más de 30% lejos del último dato ({ult[2]:,.0f}). ¿Está bien escrito?")
        if not 0.5 * ult[3] <= f["brent"] <= 1.5 * ult[3]:
            st.warning(f"{mes(fecha)}: el petróleo ({f['brent']:.0f}) está más de 50% lejos del último dato ({ult[3]:.0f}). ¿Está bien escrito?")

    nuevos = pd.DataFrame.from_dict(filas, orient="index")
    base_nueva = m.agregar_meses(base, orig, nuevos)
    r2 = modelo(base_nueva, h_local, umbral_tasa)
    fan_t, fan_i = r2["fan_tasa"], r2["fan_inflacion"]
    fechas = fan_t.index

    st.subheader(f"Paso 2 · Elige el mes que quieres ver ({mes(fechas[0])} a {mes(fechas[-1])})")
    elegido = st.select_slider("Mueve la barra para escoger el mes:", options=list(fechas),
                               value=fechas[min(5, len(fechas) - 1)], format_func=lambda f: mes(f))
    t, i = fan_t.loc[elegido], fan_i.loc[elegido]
    c1, c2 = st.columns(2)
    with c1.container(border=True):
        st.markdown(f"**🏦 Tasa de interés en {mes(elegido)}**")
        st.metric("Valor esperado", f"{t['mediana']:.2f}%",
                  f"{t['mediana'] - r2['tasa_hoy']:+.2f} pts frente a {mes(r2['fecha_hoy'])} ({r2['tasa_hoy']:.2f}%)",
                  delta_color="off")
        st.markdown(f"Lo más probable es que quede **entre {t['p5']:.2f}% y {t['p95']:.2f}%** "
                    f"(así pasa en 9 de cada 10 futuros simulados).")
    with c2.container(border=True):
        st.markdown(f"**📈 Inflación anual en {mes(elegido)}**")
        st.metric("Valor esperado", f"{i['mediana']:.2f}%",
                  f"{i['mediana'] - r2['inf_hoy']:+.2f} pts frente a {mes(r2['fecha_hoy'])} ({r2['inf_hoy']:.2f}%)",
                  delta_color="off")
        st.markdown(f"Lo más probable es que quede **entre {i['p5']:.2f}% y {i['p95']:.2f}%**. "
                    f"{'Estaría dentro de la meta del Banco de la República (2% a 4%).' if i['mediana'] <= 4 else 'Seguiría por encima de la meta del Banco de la República (2% a 4%).'}")

    st.subheader("Paso 3 · ¿Qué conviene hacer con los CDT?")
    reco = r2["reco_final"]
    with st.container(border=True):
        st.markdown(f"### {COLOR_RECO[reco]} {reco}")
        st.markdown(
            f"De cada **100 futuros** simulados para los próximos {h_local} meses, la tasa **baja** al menos "
            f"{umbral_tasa} puntos en **{100 * r2['p_baja']:.0f}** y **sube** al menos {umbral_tasa} puntos en "
            f"**{100 * r2['p_alza']:.0f}**. En los demás se queda más o menos igual.\n\n"
            f"👉 Por eso conviene {QUE_HACER[reco]}")
        if r2["fragil"]:
            st.caption("⚠️ Señal débil: la recomendación cambia según cómo se simulan los futuros; por prudencia se escalona.")
        if h_local != 6:
            st.caption("La regla del proyecto está pensada a 6 meses; a 12 meses tómalo como orientación.")
        tasa_cdt = t["mediana"] + m.SPREAD_CDT
        real = m.rentabilidad_real(tasa_cdt, i["mediana"])
        st.markdown(f"💰 **Para tener una idea:** si en {mes(elegido)} un CDT paga cerca de **{tasa_cdt:.2f}%** al año "
                    f"(tasa esperada + 0,28) y la inflación es {i['mediana']:.2f}%, la ganancia **real** (descontando la "
                    f"inflación) sería de **{real:.2f}% al año**: unos **{200_000_000 * real / 100:,.0f} pesos** sobre "
                    "200 millones.".replace(",", "."))

    st.subheader("Todos los meses")
    tabla = pd.DataFrame({
        "Mes": [mes(f) for f in fechas],
        "Tasa esperada (%)": fan_t["mediana"].round(2).values,
        "Tasa: escenario bajo – alto": [f"{a:.2f} a {b:.2f}" for a, b in zip(fan_t["p5"], fan_t["p95"])],
        "Inflación esperada (%)": fan_i["mediana"].round(2).values,
        "Inflación: escenario bajo – alto": [f"{a:.2f} a {b:.2f}" for a, b in zip(fan_i["p5"], fan_i["p95"])],
    })
    st.dataframe(tabla, hide_index=True, width="stretch")
    c1, c2 = st.columns(2)
    mostrar_grafico(c1, grafico_abanico(base_nueva["inflacion_anual"].iloc[-36:], fan_i, "Inflación anual (%)"))
    mostrar_grafico(c2, grafico_abanico(base_nueva["tasa_interes"].iloc[-36:], fan_t, "Tasa de interés (%)"))
    como_leer_grafico()
    st.caption("Es un pronóstico con rangos, no una certeza: no puede anticipar sorpresas como crisis o pandemias.")

# ---------------------------------------------------------------------------
# DATOS
# ---------------------------------------------------------------------------
elif seccion == "📊 Los datos":
    st.title("📊 Los datos")
    en_pocas_palabras(
        f"El modelo usa **cuatro datos mensuales** de Colombia, de {mes(base.index[0])} a {mes(base.index[-1])} "
        f"(**{len(base)} meses**, sin datos faltantes). Se eligieron porque se mueven juntos: si sube el petróleo o "
        "el dólar, suben los precios; si suben los precios, el Banco de la República sube las tasas.")
    st.dataframe(pd.DataFrame({
        "Dato": ["📈 Inflación anual", "🏦 Tasa de interés interbancaria", "💵 Dólar (TRM)", "🛢️ Petróleo Brent"],
        "¿Qué es?": ["Cuánto subieron los precios en un año", "Tasa a la que se prestan los bancos (sigue a la del BanRep)",
                     "Precio del dólar en pesos", "Precio internacional del barril de petróleo"],
        "¿Por qué importa?": ["Define cuánto gana de verdad un CDT", "Es la que mueve el rendimiento de los CDT",
                              "Un dólar caro encarece lo importado", "Petróleo caro encarece transporte y energía"],
        "Fuente": ["DANE (IPC)", "FRED (OCDE)", "FRED (OCDE)", "FRED"],
        "Cómo entra al modelo": ["% frente al año anterior", "Tal cual (%)", "% de cambio frente al año anterior",
                                 "% de cambio frente al año anterior"],
    }), hide_index=True, width="stretch")
    variable = st.selectbox("Elige qué dato quieres ver en el gráfico:", list(NOMBRES), format_func=NOMBRES.get)
    st.line_chart(datos[variable], height=320)
    st.caption("Se ven claramente los ciclos: la inflación alta de 2021–2023 y cómo el Banco de la República subió las "
               "tasas para frenarla.")
    with st.expander("Ver estadísticas (promedio, mínimo, máximo…)"):
        st.dataframe(datos.describe().round(2).rename(columns=NOMBRES), width="stretch")
    st.caption("La tasa interbancaria es una aproximación: no es exactamente la tasa del Banco de la República ni la que "
               "paga cada banco. Para estimar un CDT se le suman 0,28 puntos (diferencia promedio histórica 2004–2023).")
    st.download_button("⬇️ Descargar los datos (CSV)", base.to_csv().encode("utf-8"), "base_inflacion_tasas.csv", "text/csv")

# ---------------------------------------------------------------------------
# MONTE CARLO
# ---------------------------------------------------------------------------
elif seccion == "🎲 Monte Carlo: 5.000 futuros":
    st.title("🎲 Monte Carlo: 5.000 futuros posibles")
    en_pocas_palabras(
        "Nadie sabe exactamente qué va a pasar. Por eso el modelo **imagina 5.000 futuros distintos**: parte de lo que "
        "espera (el pronóstico del VAR) y le suma **sorpresas** sacadas de la historia real, como las que ya han pasado. "
        "Después **cuenta**: ¿en cuántos de esos futuros bajó la tasa? ¿en cuántos subió? De ahí salen las probabilidades.")
    tarjeta_decision(r, umbral_tasa, horizonte)
    c1, c2 = st.columns(2)
    mostrar_grafico(c1, grafico_abanico(datos["inflacion_anual"].iloc[-36:], r["fan_inflacion"], "Inflación anual (%)"))
    mostrar_grafico(c2, grafico_abanico(datos["tasa_interes"].iloc[-36:], r["fan_tasa"], "Tasa de interés (%)"))
    como_leer_grafico()

    st.subheader(f"¿Dónde terminan los 5.000 futuros dentro de {horizonte} meses?")
    fin = pd.DataFrame({"Inflación (%)": r["fan_inflacion"].iloc[-1], "Tasa de interés (%)": r["fan_tasa"].iloc[-1]}).T
    fin = fin.rename(columns={"p5": "Escenario bajo", "p25": "Bajo-medio", "mediana": "Valor esperado",
                              "p75": "Medio-alto", "p95": "Escenario alto"}).round(2)
    st.dataframe(fin, width="stretch")
    st.caption("Escenario bajo y alto: entre ellos caen 9 de cada 10 futuros. Bajo-medio y medio-alto: entre ellos cae la mitad.")
    st.write(f"Probabilidad de que la inflación vuelva a la meta del Banco de la República (4% o menos): "
             f"**{100 * r['p_meta']:.0f}%**.")

    st.subheader("¿Cambia el resultado si las sorpresas se simulan de otra forma?")
    st.markdown("Las sorpresas pueden tomarse **mes por mes sueltas** o en **tramos de meses seguidos** (más realista, "
                "porque en la vida real una sorpresa suele seguir la siguiente). Si las tres formas no dan la misma "
                "recomendación, la señal es débil y se recomienda escalonar.")
    st.dataframe(r["tabla_bloques"].round(1).rename(columns={"P(baja) %": "Chance de baja (%)", "P(sube) %": "Chance de subida (%)",
                                                              "decisión": "Recomendación"}), width="stretch")

    st.subheader("💰 Calculadora: ¿cuánto gana de verdad un CDT?")
    st.markdown("Escribe un monto y la tasa que te ofrecen; la calculadora descuenta la inflación esperada.")
    c1, c2 = st.columns(2)
    monto = c1.number_input("Monto a invertir (pesos)", min_value=1_000_000, value=200_000_000, step=10_000_000, format="%d")
    tasa_cdt = c2.number_input("Tasa que paga el CDT (% al año)", min_value=0.0, max_value=30.0,
                               value=round(r["tasa_hoy"] + m.SPREAD_CDT, 2), step=0.05)
    real = m.rentabilidad_real(tasa_cdt, r["inf_mediana"])
    st.metric("Ganancia real esperada (% al año)", f"{real:.2f}%",
              help=f"Descuenta una inflación esperada de {r['inf_mediana']:.2f}%.")
    st.write(f"En un año, {monto:,.0f} pesos ganarían unos **{monto * real / 100:,.0f} pesos por encima de la inflación**."
             .replace(",", "."))
    st.caption("Por defecto: tasa interbancaria de hoy + 0,28 puntos (supuesto del proyecto). No incluye impuestos ni comisiones.")

# ---------------------------------------------------------------------------
# MARKOV
# ---------------------------------------------------------------------------
elif seccion == "🔁 Markov: etapa del ciclo":
    st.title("🔁 Markov: ¿en qué etapa está el ciclo de tasas?")
    en_pocas_palabras(
        "Las tasas se mueven por **etapas**: un tiempo suben, luego se quedan quietas, luego bajan. Cada mes se "
        "clasifica según cuánto cambió la tasa en los últimos 3 meses: **bajando** (bajó 0,25 puntos o más), "
        "**estable** (se movió menos de 0,25) o **subiendo** (subió 0,25 o más). La cadena de Markov cuenta, con la "
        "historia, qué tan seguido se pasa de una etapa a otra.")
    st.subheader(f"Hoy ({mes(r['fecha_hoy'])}) el ciclo está: **{m.FASES[r['fase_hoy']].upper()}**")
    prob = pd.DataFrame({f"En {h} meses": 100 * r["prob_markov"][h] for h in (3, 6, 12)}, index=m.FASES).round(0)
    prob.index = ["seguir/pasar a bajando", "seguir/pasar a estable", "seguir/pasar a subiendo"]
    st.markdown("**¿Dónde podría estar el ciclo más adelante? (% de probabilidad)**")
    st.dataframe(prob, width="stretch")

    st.subheader("¿Qué tan seguido se pasa de una etapa a otra?")
    st.caption("Cómo leerlo: cada fila es la etapa de **este mes** y cada columna la del **mes siguiente**. Por ejemplo, la "
               "fila 'subiendo' dice que, cuando las tasas vienen subiendo, el mes siguiente siguen subiendo casi 9 de cada 10 veces.")
    st.dataframe(pd.DataFrame(100 * r["P"], index=[f"Este mes: {f}" for f in m.FASES],
                              columns=[f"Siguiente: {f}" for f in m.FASES]).round(1), width="stretch")
    st.caption("Duración promedio de cada etapa: " + ", ".join(
        f"{f} {1 / (1 - r['P'][i, i]):.1f} meses" for i, f in enumerate(m.FASES)) + ". Es solo descriptiva.")
    st.warning("**Ojo:** que las etapas duren tanto es en parte por cómo se miden (dos meses seguidos comparten 2 de los "
               "3 meses con que se calculan). Por eso Markov solo **describe** la etapa; la recomendación la decide Monte Carlo.")

# ---------------------------------------------------------------------------
# VAR
# ---------------------------------------------------------------------------
elif seccion == "📈 VAR: cómo se mueven juntas":
    st.title("📈 VAR: cómo se mueven juntas las cuatro variables")
    en_pocas_palabras(
        "El VAR es el modelo que hace los pronósticos. **Aprende de 22 años de historia** cómo se mueven juntas la "
        "inflación, la tasa, el dólar y el petróleo, mirando siempre los **3 meses anteriores**. Con eso proyecta los "
        "meses que vienen. Se usa para **pronosticar**, no para decir qué causa qué.")
    var = r["var"]
    st.subheader("¿Qué datos ayudan a anticipar a cuáles?")
    st.caption("Prueba de Granger: dice si conocer el pasado de un dato **ayuda a predecir** otro. No prueba que uno cause el otro.")
    filas = []
    for y in ["inflacion_anual", "tasa_interes"]:
        for x in [c for c in m.COLS_VAR if c != y]:
            p = var.test_causality(y, [x], kind="f").pvalue
            filas.append({"Pregunta": f"¿{NOMBRES[x]} ayuda a anticipar {NOMBRES[y]}?",
                          "Respuesta": "✅ Sí" if p < 0.05 else "❌ No", "p-valor": round(p, 4)})
    st.dataframe(pd.DataFrame(filas), hide_index=True, width="stretch")

    st.subheader(f"Pronóstico directo del VAR a {horizonte} meses")
    pv = r["pronostico_var"][["inflacion_anual", "tasa_interes"]].rename(columns=NOMBRES).round(2)
    pv.index = [mes(f) for f in pv.index]
    st.dataframe(pv, width="stretch")
    st.caption("Es un solo número por mes, sin rango. Las probabilidades y rangos los añade Monte Carlo.")

    with st.expander("Para expertos: diagnósticos y respuesta a choques"):
        ruta = CARPETA_RESULTADOS / "diagnosticos_var.csv"
        if ruta.exists():
            st.dataframe(pd.read_csv(ruta)[["prueba", "resultado"]], hide_index=True, width="stretch")
        st.markdown("El modelo es **estable**, pero sus errores tienen autocorrelación, no son normales y su varianza "
                    "cambia (ARCH). Por eso los rangos salen de Monte Carlo con errores reales y no de una fórmula.")
        irf = var.irf(24).orth_irfs[:, 0, :]
        st.markdown("**Cómo responde la inflación a una sorpresa en cada variable (meses):**")
        st.line_chart(pd.DataFrame(irf, columns=[NOMBRES[c] for c in m.COLS_VAR]).rename_axis("meses"), height=280)
        st.warning("**Price puzzle:** tras una subida de la tasa, el modelo muestra la inflación subiendo el primer año. No "
                   "significa que subir tasas cause inflación: el Banco de la República sube la tasa **cuando ve venir** "
                   "inflación, y el modelo confunde la reacción con la causa.")
        ruta = CARPETA_RESULTADOS / "sensibilidad_orden.csv"
        if ruta.exists():
            st.dataframe(pd.read_csv(ruta, index_col=0), width="stretch")

# ---------------------------------------------------------------------------
# VALIDACIÓN Y CONCLUSIONES
# ---------------------------------------------------------------------------
elif seccion == "✅ ¿Qué tan confiable es?":
    st.title("✅ ¿Qué tan confiable es el modelo?")
    en_pocas_palabras(
        "Pusimos a prueba el modelo con el pasado, **sin dejarlo ver el futuro**: desde 2011, cada mes lo entrenamos "
        "solo con los datos que existían hasta ese momento, le pedimos un pronóstico a 6 meses y lo comparamos con lo "
        "que de verdad pasó. Lo hicimos **183 veces**.")
    ruta_val = CARPETA_RESULTADOS / "validacion.csv"
    if not ruta_val.exists():
        st.error("No encuentro resultados/validacion.csv. Corre primero: python modelo_completo_v3.py")
    else:
        val = pd.read_csv(ruta_val, index_col=0, parse_dates=True)
        grandes = val[val["cambio_real_tasa"].abs() >= 0.5]
        dr = np.sign(grandes["cambio_real_tasa"])
        ok_m, ok_t = grandes["señal_modelo"] == dr, grandes["señal_tendencia"] == dr
        st.subheader("1. ¿Acierta si la tasa sube o baja?")
        c1, c2, c3 = st.columns(3)
        c1.metric("Veces que la tasa se movió 0,5 pts o más", len(grandes))
        c2.metric("El modelo acertó la dirección", f"{ok_m.sum()} veces ({100 * ok_m.mean():.0f}%)")
        c3.metric("Regla simple 'si viene subiendo, sigue'", f"{ok_t.sum()} veces ({100 * ok_t.mean():.0f}%)")
        st.markdown("**Lectura honesta:** el modelo acierta 8 de cada 10 veces, pero **lo mismo que la regla simple**. "
                    "El Banco de la República mueve las tasas despacio y por etapas, así que mirar la tendencia ya funciona "
                    "bastante bien. No decimos que el modelo sea mejor en esto.")
        ruta_err = CARPETA_RESULTADOS / "errores_pronostico.csv"
        if ruta_err.exists():
            st.subheader("2. ¿Qué tanto se equivoca en el número?")
            err = pd.read_csv(ruta_err, index_col=0)
            err.columns = ["Modelo (VAR)", "Otra versión del modelo (en diferencias)", "Suponer que nada cambia"]
            st.dataframe(err, width="stretch")
            st.caption("Error promedio a 6 meses, en puntos porcentuales: más bajo es mejor.")
            st.markdown(f"**Lectura:** en la **inflación** el modelo se equivoca en promedio **{err.iloc[0, 0]:.2f}** puntos, "
                        f"contra **{err.iloc[0, 2]:.2f}** de suponer que no cambia: es **{100 * (1 - err.iloc[0, 0] / err.iloc[0, 2]):.0f}% "
                        "mejor**. Ahí está su mayor aporte.")
        ruta_est = CARPETA_RESULTADOS / "estrategias_cdt.csv"
        if ruta_est.exists():
            with st.expander("3. ¿Habría ganado más plata siguiendo el modelo? (ejercicio ilustrativo)"):
                st.dataframe(pd.read_csv(ruta_est, index_col=0).round(3), width="stretch")
                st.caption("Sí, pero poco: unos 0,1 puntos al año más que escalonar siempre (unos 200.000 pesos sobre 200 "
                           "millones). Supone CDT = tasa interbancaria + 0,28, sin impuestos ni comisiones.")
    st.subheader("Conclusiones")
    st.markdown(
        "- ✅ **Pronostica la inflación** mejor que suponer que no cambia, y eso define cuánto gana de verdad un CDT.\n"
        "- ✅ **Da probabilidades y rangos**, no solo un número: ayuda a decidir con la incertidumbre a la vista.\n"
        "- ✅ **Sabe cuándo no está seguro:** si la señal es débil, recomienda escalonar en vez de apostar.\n"
        "- 🟰 **Para saber si la tasa sube o baja**, empata con una regla simple de seguir la tendencia.\n"
        f"- 📌 **Hoy ({mes(r['fecha_hoy'])}):** {r['reco_final'].lower()}. Es más probable una baja ({100 * r['p_baja']:.0f}%) "
        f"que una subida ({100 * r['p_alza']:.0f}%), pero sin llegar al 40%.")
    with st.expander("Limitaciones"):
        st.markdown(
            "- La tasa interbancaria es una aproximación; no es la tasa del Banco de la República ni la de cada banco.\n"
            "- El costo de un CDT se estima con un supuesto (+0,28 puntos); no incluye impuestos ni comisiones.\n"
            "- El modelo no anticipa sorpresas como crisis, pandemias o decisiones inesperadas.\n"
            "- Los resultados de Monte Carlo cambian un poco según cómo se simulen las sorpresas.\n"
            "- Las etapas de Markov parecen más largas en parte por cómo se miden.\n\n"
            "Herramienta académica de apoyo a la decisión; no es una recomendación de inversión.")
    st.caption("Fuentes: DANE (IPC) y FRED – Federal Reserve Bank of St. Louis (COLIRSTCI01STM, COLCCUSMA02STM, "
               "POILBREUSDM, COLIR3TCD01STM).")
