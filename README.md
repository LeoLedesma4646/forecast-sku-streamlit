# Forecast de ventas semanales por SKU

Proyecto académico en una sola app Streamlit. La arquitectura se mantiene simple por fuera y más rigurosa por dentro: **10 SKU prioritarios**, **4 modelos candidatos**, validación temporal común y persistencia de modelos ya entrenados.

## Enfoque actualizado

Los SKU se agregan a nivel semanal en el intervalo común de sus combinaciones canal–región. Se consideran admisibles si tienen al menos **104 semanas**, equivalentes a **2 ciclos completos** de un periodo estacional anual de **52 semanas**. Entre los admisibles se seleccionan los 10 de mayor ingreso histórico (`units_sold × price_unit`).

Para cada SKU se comparan exactamente los mismos bloques de backtest:

1. **Seasonal Naive (52)**: baseline; repite el valor observado 52 semanas atrás.
2. **Fourier + Ridge**: tendencia `t`, `t²`, Fourier anual (K=2), precio, promoción y calendario. `alpha` se calibra por SKU.
3. **SARIMAX estacional**: modelo estacional genuino con `s=52` + precio, promoción y calendario. **No usa Fourier**.
4. **Fourier + ARMA (DHR)**: regresión armónica dinámica; usa tendencia + Fourier + exógenas y añade dependencia ARMA en los errores.

La selección final por SKU usa **WAPE como criterio principal** y **MAE como métrica secundaria**. No se fuerza que gane el modelo más complejo; Seasonal Naive puede ser el seleccionado si tiene menor error fuera de muestra.

## Separación entre calibración y evaluación

Antes del backtest externo se reserva un pequeño bloque temporal de calibración para elegir parámetros sin mirar las 12 semanas usadas posteriormente para comparar los cuatro modelos. De esta manera los parámetros no se eligen usando el mismo bloque con el que se reporta el desempeño final.

- Ridge: `alpha ∈ {0.1, 1, 10, 100}`.
- DHR: errores AR(1), MA(1) o ARMA(1,1).
- SARIMAX estacional: alternativas parsimoniosas con efecto estacional AR(1) o MA(1) a 52 semanas.
- Backtest externo: 3 folds × 4 semanas = 12 semanas evaluadas.

## Variables futuras

Se utilizan `price_unit`, `promotion_flag`, `is_summer`, `is_winter` e `is_holiday_week`. `stock_available` y `delivery_days` **no** se utilizan como predictores del forecast futuro.

## Archivos principales

- `01_modelamiento_forecast.ipynb`: laboratorio reproducible: selección, EDA, STL, calibración, backtesting, métricas, entrenamiento final y guardado.
- `01_modelamiento_forecast_ejecutado.ipynb`: versión ejecutada con resultados.
- `forecast_core.py`: lógica de datos, features, modelos, backtest y forecast.
- `app.py`: interfaz Streamlit.
- `artifacts/`: tablas de selección, métricas, predicciones, parámetros y coeficientes.
- `models/`: modelos `.pkl` y metadata por SKU.
- `data/weekly_df_final_for_modeling.xlsx`: copia local de la base.

## Papers de referencia

**Principal**  
Kačmáry, P., Bindzár, P., Kovalčík, J., & Ondov, M. (2024). *Forecast of sales of selected food products in retail using Fourier series analysis and non-linear regression*. Foresight, 26(3), 487–504. DOI: 10.1108/FS-12-2022-0168.

**Apoyo**  
de Castro Moraes, T., Yuan, X.-M., & Chew, E. P. (2024). *Hybrid convolutional long short-term memory models for sales forecasting in retail*. Journal of Forecasting, 43(5), 1278–1293. DOI: 10.1002/for.3073.

El proyecto es una **adaptación metodológica** al dataset del curso; no pretende reproducir literalmente la especificación completa de los papers.

## Ejecución

1. Activar el entorno `ledesma_mpd`.
2. Ejecutar una vez `01_modelamiento_forecast.ipynb` si cambió la base o la metodología.
3. Iniciar Streamlit con `run_app.bat` o:

```bash
python -m streamlit run app.py --server.address localhost --server.port 8501
```

Streamlit reutiliza los modelos guardados, por lo que no reentrena los 10 SKU cada vez que se abre.

## Nota de la base

La copia incluida conserva la fuente disponible en el proyecto. Existe una observación no numérica (`q`) en `units_sold` para un SKU que **no pertenece a los 10 SKU seleccionados**. Si actualizas la base, reemplaza el Excel y vuelve a ejecutar el notebook para regenerar resultados y modelos.
