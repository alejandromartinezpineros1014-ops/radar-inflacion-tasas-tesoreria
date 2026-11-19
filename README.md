# Radar de inflación y tasas para tesorería

Proyecto integrador – *Análisis e Interpretación de Datos a través de Python* (tercer corte).
Diego Alejandro Martínez Piñeros y María Juliana Brand Ardila.

**Aplicación publicada:** https://radar-inflacion-tasas.streamlit.app

**Pregunta:** ¿cómo permiten las relaciones dinámicas entre inflación, tasa de interés, TRM y Brent en
Colombia, junto con Markov y Monte Carlo, anticipar la inflación y la tasa a 3 y 6 meses y orientar las
decisiones de plazo de los CDT de una tesorería?

## Contenido del repositorio

| Archivo | Qué es |
|---|---|
| `app.py` | Aplicación Streamlit: Inicio, Pronosticar próximos meses, Los datos, Monte Carlo, Markov, VAR y ¿Qué tan confiable es? |
| `motor.py` | Funciones del modelo que usa la app (mismo modelo que `modelo_completo_v3.py`) |
| `modelo_completo_v3.py` | Análisis completo con todas las pruebas; genera la carpeta `resultados/` |
| `datos/base_inflacion_tasas.csv` | Base del modelo: 273 meses (2004-01 a 2026-09) |
| `datos/base_inflacion_tasas.xlsx` | La misma base con diccionario de variables, datos originales y fuentes |
| `datos/datos_originales.csv` | Datos en niveles 2003–2026 (IPC, tasa, TRM, Brent), para actualizar con meses nuevos |
| `resultados/` | Tablas y gráficos generados por `modelo_completo_v3.py` (la app lee la validación de aquí) |
| `requirements.txt` | Librerías necesarias |

## Fuentes de datos

- DANE – Índice de Precios al Consumidor, serie de empalme (base dic-2018).
- FRED (Federal Reserve Bank of St. Louis): COLIRSTCI01STM (tasa interbancaria), COLCCUSMA02STM (TRM),
  POILBREUSDM (Brent), COLIR3TCD01STM (CDT 90 días, solo para el supuesto de spread).

## Cómo ejecutarlo en el computador

```bash
python -m venv venv
source venv/bin/activate          # En Windows: venv\Scripts\activate
pip install -r requirements.txt
python modelo_completo_v3.py      # análisis completo; crea/actualiza resultados/
streamlit run app.py              # abre la app en el navegador
```

## Cómo publicarlo en Streamlit Community Cloud

1. Subir esta carpeta a un repositorio público de GitHub (sin la carpeta `venv`).
2. Entrar a https://share.streamlit.io con la cuenta de GitHub y elegir **Create app**.
3. Seleccionar el repositorio, la rama `main` y el archivo principal `app.py`. Publicar.

## Resultado vigente (base hasta septiembre de 2026)

P(tasa baja ≥ 0,5 puntos en 6 meses) = 35 %, P(sube) = 18 %. La decisión cambia según el método de
remuestreo, así que la recomendación prudente es **escalonar**. Validación 2011–2026: el modelo y la regla de
tendencia aciertan la dirección en el 80 % de los movimientos grandes (sin diferencia significativa); el VAR
pronostica la inflación un 11 % mejor que suponer que no cambia.

*Herramienta académica de apoyo a la decisión; no es una recomendación de inversión.*
