from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from forecast_core import (
    MIN_WEEKS, TOP_N, PERIOD, MODEL_NAMES, MODEL_NAIVE, MODEL_RIDGE,
    MODEL_SARIMAX, MODEL_DHR, EXOG_BASE,
    load_data, sku_summary, select_top_skus, aggregate_sku,
    correlation_table, decompose_stl, future_scenario,
    load_saved_models, fit_and_save_sku, rolling_backtest,
    select_hyperparameters, forecast_from_saved, add_fourier_features,
    ridge_coefficients,
)

st.set_page_config(page_title="Forecast de ventas por SKU", page_icon="📈", layout="wide")

ROOT = Path(__file__).resolve().parent
DEFAULT_DATA = ROOT / "data" / "weekly_df_final_for_modeling.xlsx"
ART = ROOT / "artifacts"
MODELS = ROOT / "models"


@st.cache_data(show_spinner=False)
def cached_load(path):
    return load_data(path)


@st.cache_data(show_spinner=False)
def cached_summary(df):
    return sku_summary(df)


def load_source():
    if DEFAULT_DATA.exists():
        return cached_load(DEFAULT_DATA)
    up = st.sidebar.file_uploader("Cargar Excel", type=["xlsx"])
    if up is None:
        st.info("Carga el Excel para comenzar.")
        st.stop()
    temp = ROOT / "_uploaded.xlsx"
    temp.write_bytes(up.getvalue())
    return load_data(temp)


def get_evaluation(series, sku):
    metrics_path = ART / "model_metrics.csv"
    preds_path = ART / "backtest_predictions.csv"
    params_path = ART / "selected_parameters.csv"

    if metrics_path.exists() and preds_path.exists():
        metrics_all = pd.read_csv(metrics_path)
        preds_all = pd.read_csv(preds_path, parse_dates=["week"])
        metrics = metrics_all[metrics_all["sku"].eq(sku)].sort_values(["WAPE", "MAE"])
        preds = preds_all[preds_all["sku"].eq(sku)].copy()
        params = None
        if params_path.exists():
            p = pd.read_csv(params_path)
            p = p[p["sku"].eq(sku)]
            if not p.empty:
                params = p.iloc[0].to_dict()
        if not metrics.empty and not preds.empty:
            return preds, metrics, params

    params = select_hyperparameters(series)
    preds, metrics = rolling_backtest(series, params=params)
    preds["sku"] = sku
    metrics["sku"] = sku
    return preds, metrics, params


df = load_source()
summary = cached_summary(df)
top10 = select_top_skus(summary, TOP_N)
selected_skus = top10["sku"].tolist()

st.sidebar.title("Forecast por SKU")
page = st.sidebar.radio("Navegación", ["Inicio", "Explorar SKU", "Modelos", "Forecast"])
sku = st.sidebar.selectbox("SKU seleccionado", selected_skus)

st.title("Forecast de ventas semanales")
st.caption("10 SKU prioritarios · periodo anual de 52 semanas · 4 modelos candidatos · backtesting temporal")

if page == "Inicio":
    st.subheader("1. Selección de la muestra")
    st.write(
        "Se consideran admisibles los SKU con al menos 104 semanas comunes: "
        "dos ciclos completos de un periodo estacional anual de 52 semanas."
    )

    s = summary.sort_values("common_weeks", ascending=False).copy()
    s["estado"] = np.where(s["admissible"], "Admisible", "No admisible")
    fig = px.bar(
        s, x="sku", y="common_weeks", color="estado",
        color_discrete_map={"Admisible": "#4E8F68", "No admisible": "#D9534F"},
        labels={"common_weeks": "Semanas comunes", "sku": "SKU"},
    )
    fig.add_hline(y=MIN_WEEKS, line_dash="dash", annotation_text="104 semanas = 2 ciclos")
    fig.update_layout(legend_title_text="", height=430)
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("2. Pareto de ingresos")
    p = summary.sort_values("total_revenue", ascending=False).copy()
    p["cum_pct"] = 100 * p["total_revenue"].cumsum() / p["total_revenue"].sum()
    fig = go.Figure()
    fig.add_bar(x=p["sku"], y=p["total_revenue"], name="Ingresos históricos")
    fig.add_scatter(x=p["sku"], y=p["cum_pct"], name="% acumulado", yaxis="y2", mode="lines+markers")
    fig.update_layout(
        yaxis=dict(title="Ingresos"),
        yaxis2=dict(title="% acumulado", overlaying="y", side="right", range=[0, 105]),
        legend=dict(orientation="h"), height=450,
    )
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("10 SKU seleccionados")
    view = top10[["sku", "common_weeks", "seasonal_cycles", "total_units", "total_revenue"]].copy()
    view["seasonal_cycles"] = view["seasonal_cycles"].round(2)
    view.columns = ["SKU", "Semanas", "Ciclos de 52 semanas", "Unidades históricas", "Ingresos históricos"]
    st.dataframe(view, hide_index=True, use_container_width=True)

elif page == "Explorar SKU":
    series = aggregate_sku(df, sku)
    st.subheader(f"Comportamiento de {sku}")

    c1, c2, c3 = st.columns(3)
    c1.metric("Semanas", len(series))
    c2.metric("Ciclos anuales", f"{len(series) / PERIOD:.2f}")
    c3.metric("Venta semanal media", f"{series['units_sold'].mean():,.0f} u.")

    fig = px.line(series, x="week", y="units_sold", labels={"week": "Semana", "units_sold": "Unidades"})
    fig.update_layout(height=420)
    st.plotly_chart(fig, use_container_width=True)

    with st.expander("Ver tendencia, estacionalidad y ruido", expanded=False):
        dec = decompose_stl(series, PERIOD)
        parts = pd.DataFrame({
            "week": dec.observed.index,
            "Observado": dec.observed.values,
            "Tendencia": dec.trend.values,
            "Estacionalidad": dec.seasonal.values,
            "Ruido / residuo": dec.resid.values,
        })
        for col in ["Tendencia", "Estacionalidad", "Ruido / residuo"]:
            f = px.line(parts, x="week", y=col)
            f.update_layout(height=280, margin=dict(t=20, b=20))
            st.plotly_chart(f, use_container_width=True)

    with st.expander("Ver relación con precio, promoción y calendario", expanded=False):
        corr = correlation_table(series)
        corr["interpretación"] = corr["spearman_rho"].apply(
            lambda r: "Sin relación clara" if pd.isna(r) or abs(r) < .2
            else ("Relación moderada" if abs(r) < .5 else "Relación fuerte")
        )
        st.dataframe(
            corr[["variable", "spearman_rho", "p_value", "interpretación"]],
            hide_index=True, use_container_width=True,
        )
        st.caption("La correlación describe asociación; no demuestra causalidad.")

elif page == "Modelos":
    series = aggregate_sku(df, sku)
    with st.spinner("Cargando validación temporal…"):
        preds, metrics, params = get_evaluation(series, sku)

    winner = metrics.sort_values(["WAPE", "MAE"]).iloc[0]
    st.subheader("Comparación de modelos")
    st.success(
        f"Modelo seleccionado para {sku}: **{winner['model']}** · "
        f"WAPE {winner['WAPE']:.2f}% · MAE {winner['MAE']:.1f} unidades."
    )

    show = metrics[["model", "WAPE", "MAE", "rank"]].copy()
    show.columns = ["Modelo", "WAPE (%)", "MAE (unidades)", "Orden"]
    st.dataframe(show, hide_index=True, use_container_width=True)

    st.subheader("Real vs. Predicción")
    chosen = st.selectbox("Modelo a visualizar", metrics["model"].tolist())
    p = preds[preds["model"].eq(chosen)].sort_values("week")
    fig = go.Figure()
    fig.add_scatter(x=p["week"], y=p["actual"], mode="lines+markers", name="Real")
    fig.add_scatter(x=p["week"], y=p["prediction"], mode="lines+markers", name="Predicción")
    fig.update_layout(height=430, yaxis_title="Unidades")
    st.plotly_chart(fig, use_container_width=True)

    with st.expander("¿Qué representa cada modelo?", expanded=False):
        st.markdown(
            "- **Seasonal Naive:** repite la observación de hace 52 semanas.\n"
            "- **Fourier + Ridge:** tendencia + Fourier anual + precio, promoción y calendario, con regularización Ridge.\n"
            "- **SARIMAX estacional:** estacionalidad temporal explícita a 52 semanas + exógenas; **no usa Fourier**.\n"
            "- **Fourier + ARMA (DHR):** regresión armónica con Fourier + exógenas y una estructura ARMA para los errores."
        )

    with st.expander("Detalles técnicos del SKU", expanded=False):
        meta_path = MODELS / f"{sku}_meta.json"
        if meta_path.exists():
            models = load_saved_models(sku, MODELS)
            meta = models["meta"]
            st.write(f"**Periodo estacional:** {meta.get('seasonal_period', PERIOD)} semanas")
            st.write(f"**Ciclos disponibles:** {meta.get('seasonal_cycles', len(series)/PERIOD):.2f}")
            st.write(f"**Ridge λ (alpha):** {meta.get('ridge_alpha')}")
            st.write(f"**DHR order:** {tuple(meta.get('dhr_order', []))}")
            st.write(
                f"**SARIMAX:** {tuple(meta.get('sarimax_order', []))} × "
                f"{tuple(meta.get('sarimax_seasonal_order', []))}"
            )

            coeff_path = ART / "ridge_coefficients.csv"
            if coeff_path.exists():
                coeff = pd.read_csv(coeff_path)
                coeff = coeff[coeff["sku"].eq(sku)].copy()
                if not coeff.empty:
                    st.caption("Coeficientes Ridge sobre predictores estandarizados; son predictivos, no causales.")
                    st.dataframe(coeff[["feature", "standardized_coefficient"]], hide_index=True, use_container_width=True)
        else:
            st.info("Los detalles se generan al ejecutar el notebook de modelamiento.")

elif page == "Forecast":
    series = aggregate_sku(df, sku)
    metrics_path = ART / "model_metrics.csv"
    if metrics_path.exists():
        metrics = pd.read_csv(metrics_path)
        metrics = metrics[metrics["sku"].eq(sku)].sort_values(["WAPE", "MAE"])
        best_model = metrics.iloc[0]["model"] if not metrics.empty else MODEL_RIDGE
    else:
        _, metrics, params = get_evaluation(series, sku)
        best_model = metrics.iloc[0]["model"]

    try:
        models = load_saved_models(sku, MODELS)
    except Exception:
        with st.spinner("Preparando los modelos de este SKU por primera vez…"):
            params = select_hyperparameters(series)
            _, m = rolling_backtest(series, params=params)
            fit_and_save_sku(series, sku, MODELS, m, params=params)
            models = load_saved_models(sku, MODELS)

    st.subheader(f"Forecast para {sku}")
    c1, c2 = st.columns(2)
    horizon = int(c1.number_input("Horizonte (semanas)", 1, 12, 4, 1))
    model_options = ["Mejor según WAPE", *MODEL_NAMES]
    model_choice = c2.selectbox("Modelo", model_options)
    model_name = best_model if model_choice == "Mejor según WAPE" else model_choice

    with st.expander("Escenario futuro", expanded=True):
        price_change = st.slider("Cambio de precio vs. promedio reciente (%)", -20, 20, 0)
        promo_pct = st.slider(
            "Cobertura promocional estimada (%)", 0, 100,
            int(series["promotion_flag"].tail(4).mean() * 100),
        )
        holiday = st.checkbox("Tratar las semanas futuras como periodo especial/feriado", value=False)

    future = future_scenario(
        series, horizon,
        price_change_pct=price_change,
        promotion_share=promo_pct / 100,
        holiday=holiday,
        source_df=df,
    )
    pred = forecast_from_saved(series, future, model_name, models)
    out = future[["week"]].copy()
    out["forecast_units"] = pred

    st.metric(
        f"Pronóstico semana {out['week'].iloc[-1]:%d/%m/%Y}",
        f"{out['forecast_units'].iloc[-1]:,.0f} unidades",
    )
    if model_name == MODEL_NAIVE:
        st.caption("Seasonal Naive no utiliza precio, promoción ni calendario del escenario.")

    hist = series[["week", "units_sold"]].tail(52).copy()
    fig = go.Figure()
    fig.add_scatter(x=hist["week"], y=hist["units_sold"], mode="lines", name="Histórico")
    fig.add_scatter(x=out["week"], y=out["forecast_units"], mode="lines+markers", name="Forecast")
    fig.update_layout(height=450, yaxis_title="Unidades")
    st.plotly_chart(fig, use_container_width=True)
    st.dataframe(out, hide_index=True, use_container_width=True)

st.sidebar.caption("Los modelos guardados se reutilizan entre aperturas de Streamlit.")
