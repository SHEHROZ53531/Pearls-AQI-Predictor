"""
Pearls AQI Multicity Forecast Dashboard.
Live AQI forecast, methodology, and about pages for Islamabad,
Rawalpindi, Lahore, and Faisalabad.
"""

import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import shap
import streamlit.components.v1 as components

from utils import (
    connect_to_mongo,
    load_full_feature_table,
    engineer_features_for_city,
    load_all_bucket_models,
    predict_next_72_hours,
    classify_aqi,
    CITIES,
    CITY_DISPLAY_NAMES,
    MODEL_FEATURE_COLUMNS,
)

st.set_page_config(page_title="Pearls AQI Multicity Forecast", page_icon="🌫️", layout="wide")

CUSTOM_CSS = """
<style>
.stApp, [data-testid="stHeader"], [data-testid="stAppViewContainer"] { background-color: #f8fafc !important; }
.block-container { padding-top: 1.5rem; max-width: 1240px; }

.brand-header { display: flex; align-items: center; gap: 14px; }
.brand-logo {
    width: 46px; height: 46px; border-radius: 12px; background: linear-gradient(135deg, #0f766e 0%, #115e59 100%);
    color: white; display: flex; align-items: center; justify-content: center; font-weight: 800; font-size: 17px;
    box-shadow: 0 4px 12px rgba(15, 118, 110, 0.2); flex-shrink: 0;
}
.brand-title { font-size: 20px; font-weight: 800; color: #0f172a; margin: 0; line-height: 1.2; }
.brand-subtitle { font-size: 13px; color: #64748b; margin: 2px 0 0 0; }

.pill { display: inline-block; padding: 4px 12px; border-radius: 999px; font-size: 11px; font-weight: 700; margin-right: 8px; }
.pill-live { background: #d1fae5; color: #065f46; border: 1px solid #a7f3d0; }
.pill-auto { background: #dbeafe; color: #1e3a8a; }

.hero { background: linear-gradient(135deg, #0f172a 0%, #0f766e 100%); border-radius: 20px; padding: 48px; color: white; }
.hero-eyebrow { font-size: 12px; font-weight: 700; letter-spacing: 0.08em; color: #99f6e4; text-transform: uppercase; }
.hero-title { font-size: 42px; font-weight: 800; line-height: 1.15; margin: 12px 0 16px 0; color: white; }
.hero-desc { font-size: 15px; color: #cbd5e1; max-width: 520px; line-height: 1.6; }

.metric-card {
    background: #ffffff; border: 1px solid #e2e8f0; border-radius: 16px; padding: 20px 22px;
    box-shadow: 0 1px 3px rgba(0,0,0,0.02), 0 4px 6px -2px rgba(0,0,0,0.02); height: 100%;
}
.metric-label { font-size: 11px; font-weight: 700; color: #64748b; text-transform: uppercase; letter-spacing: 0.06em; margin-bottom: 6px; }
.metric-value { font-size: 34px; font-weight: 800; color: #0f172a; letter-spacing: -0.02em; }
.metric-sub { font-size: 12px; color: #64748b; margin-top: 4px; }

.day-card {
    background: #ffffff; border: 1px solid #e2e8f0; border-radius: 16px; padding: 20px 16px;
    text-align: center; box-shadow: 0 1px 3px rgba(0,0,0,0.02);
}
.day-label {
    font-size: 12px; font-weight: 700; color: #0f766e; background: #ccfbf1;
    display: inline-block; padding: 4px 14px; border-radius: 999px; margin-bottom: 12px;
}
.day-aqi { font-size: 32px; font-weight: 800; line-height: 1; margin-bottom: 6px; }
.day-category { font-size: 12px; font-weight: 600; color: #64748b; }

.section-box {
    background: #ffffff; border: 1px solid #e2e8f0; border-radius: 16px;
    padding: 22px; box-shadow: 0 1px 3px rgba(0,0,0,0.02);
}
.section-title { font-size: 18px; font-weight: 800; color: #0f172a; margin-top: 8px; margin-bottom: 2px; }
.section-caption { font-size: 13px; color: #64748b; margin-bottom: 16px; }

.model-bucket-box {
    background: #f8fafc; border: 1px solid #e2e8f0; border-radius: 12px; padding: 14px 16px;
}
.model-bucket-title { font-size: 13px; font-weight: 700; color: #0f766e; text-transform: uppercase; letter-spacing: 0.05em; }

.flow-step { background: white; border: 1px solid #e2e8f0; border-radius: 14px; padding: 18px; height: 100%; }
.flow-step-num { font-size: 12px; font-weight: 700; color: #0f766e; }
.flow-step-title { font-size: 16px; font-weight: 700; color: #0f172a; margin: 6px 0; }
.flow-step-desc { font-size: 13px; color: #64748b; }

.about-card { background: linear-gradient(135deg, #0f172a 0%, #0f766e 100%); border-radius: 20px; padding: 32px; color: white; }
.about-name { font-size: 26px; font-weight: 800; margin: 6px 0; color: #ffffff !important; }
.about-text { color: #e2e8f0 !important; }

.stMarkdown, .stMarkdown p { color: #0f172a; }

div[data-testid="stButton"] > button {
    white-space: nowrap;
    font-size: 13px;
    font-weight: 600;
    padding: 8px 10px;
    border-radius: 10px;
}

div[data-testid="stButton"] > button[kind="primary"] {
    background-color: #0f766e !important;
    border-color: #0f766e !important;
    color: white !important;
}
div[data-testid="stButton"] > button[kind="primary"]:hover {
    background-color: #0d5f58 !important;
    border-color: #0d5f58 !important;
}

div[data-testid="stLinkButton"] > a {
    white-space: nowrap;
    font-size: 13px;
    font-weight: 600;
    border-radius: 10px;
}

.tech-grid { display: grid; grid-template-columns: repeat(3, 1fr); gap: 12px; }
.tech-badge {
    background: white; border: 1px solid #e2e8f0; border-radius: 12px;
    padding: 14px; display: flex; align-items: center; gap: 10px;
}
.tech-icon { font-size: 22px; }
.tech-name { font-weight: 600; color: #0f172a; font-size: 14px; }
</style>
"""
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

if "page" not in st.session_state:
    st.session_state.page = "Home"


@st.cache_resource(show_spinner="Connecting to MongoDB and loading models...")
def get_db_and_models():
    db = connect_to_mongo()
    model_bundles = load_all_bucket_models(db)
    return db, model_bundles


@st.cache_data(ttl=1800, show_spinner="Loading latest data from the database...")
def get_full_table():
    db, _ = get_db_and_models()
    return load_full_feature_table(db)


def get_forecast_for_city(city: str):
    full_df = get_full_table()
    _, model_bundles = get_db_and_models()
    history_df = engineer_features_for_city(full_df, city)
    forecast_df = predict_next_72_hours(history_df, model_bundles)
    latest_actual_aqi = history_df["aqi"].dropna().iloc[-1]
    latest_timestamp = history_df.index.max()
    return history_df, forecast_df, latest_actual_aqi, latest_timestamp, model_bundles


def render_topbar():
    st.markdown(
        """
        <div class="brand-header">
            <div class="brand-logo">AQ</div>
            <div>
                <p class="brand-title">Pearls AQI Multicity Forecast</p>
                <p class="brand-subtitle">Islamabad · Rawalpindi · Lahore · Faisalabad</p>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.markdown("<br>", unsafe_allow_html=True)

    nav_cols = st.columns(5)
    pages = ["Home", "Dashboard", "Trends", "Methodology", "About"]
    for nc, p in zip(nav_cols, pages):
        with nc:
            button_type = "primary" if st.session_state.page == p else "secondary"
            if st.button(p, key=f"nav_{p}", use_container_width=True, type=button_type):
                st.session_state.page = p
                st.rerun()
    st.markdown("<br>", unsafe_allow_html=True)


def render_home_page():
    st.markdown(
        """
        <div class="hero">
            <div class="hero-eyebrow">● Live Multicity System · Production Ready</div>
            <div class="hero-title">Air quality forecasting,<br>explained like an intelligent<br>control room.</div>
            <div class="hero-desc">
                A serverless AQI forecasting platform covering four Pakistani cities,
                combining automated hourly ingestion, daily model retraining, and a
                cloud model registry to deliver live 3-day AQI predictions.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.markdown("<br>", unsafe_allow_html=True)

    col1, col2, col3, col4 = st.columns(4)
    features = [
        ("Hourly", "Feature Ingestion", "Automated collection of weather and pollutant data for 4 cities."),
        ("Daily", "Model Training", "Retraining pipeline finds the most accurate model per forecast day."),
        ("3 Days", "Forecast Horizon", "Serving Day +1, Day +2, and Day +3 AQI forecasts."),
        ("Cloud", "Secure Storage", "MongoDB Atlas stores features, metrics, and registered models."),
    ]
    for col, (big, label, desc) in zip([col1, col2, col3, col4], features):
        with col:
            st.markdown(
                f"""<div class="metric-card">
                    <div style="font-size:24px; font-weight:800; color:#0f766e;">{big}</div>
                    <div style="font-weight:700; color:#0f172a; margin:4px 0;">{label}</div>
                    <div class="metric-sub">{desc}</div>
                </div>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    col1, col2 = st.columns([2, 1])
    with col1:
        if st.button("Explore Dashboard →", use_container_width=True, type="primary"):
            st.session_state.page = "Dashboard"
            st.rerun()
    with col2:
        if st.button("View Methodology", use_container_width=True):
            st.session_state.page = "Methodology"
            st.rerun()


def render_current_metrics(latest_aqi: float, latest_timestamp, city_display: str):
    category, color = classify_aqi(latest_aqi)
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown(
            f"""<div class="metric-card">
                <div class="metric-label">Current Reading · {city_display}</div>
                <div class="metric-value">{latest_aqi:.0f} <span style="font-size:16px; font-weight:500; color:#94a3b8;">AQI</span></div>
                <div class="metric-sub">Updated {latest_timestamp.strftime('%Y-%m-%d %H:%M UTC')}</div>
            </div>""", unsafe_allow_html=True)
    with col2:
        st.markdown(
            f"""<div class="metric-card" style="border-left: 5px solid {color};">
                <div class="metric-label">Air Quality Status</div>
                <div class="metric-value" style="font-size:24px; color:{color};">{category}</div>
                <div class="metric-sub">Live Category Index</div>
            </div>""", unsafe_allow_html=True)
    with col3:
        st.markdown(
            """<div class="metric-card">
                <div class="metric-label">Forecast Horizon</div>
                <div class="metric-value">72 <span style="font-size:16px; font-weight:500; color:#94a3b8;">Hours</span></div>
                <div class="metric-sub">Hourly multi-model resolution</div>
            </div>""", unsafe_allow_html=True)


def render_day_summary(forecast_df: pd.DataFrame):
    st.markdown('<p class="section-title">3-Day Outlook</p>', unsafe_allow_html=True)
    st.markdown('<p class="section-caption">Average predicted AQI for each of the next 3 days</p>', unsafe_allow_html=True)

    day_ranges = {"Day 1": (1, 24), "Day 2": (25, 48), "Day 3": (49, 72)}
    cols = st.columns(3)
    for col, (day_label, (low, high)) in zip(cols, day_ranges.items()):
        day_data = forecast_df[(forecast_df["horizon"] >= low) & (forecast_df["horizon"] <= high)]
        avg_aqi = day_data["predicted_aqi"].mean()
        category, color = classify_aqi(avg_aqi)
        with col:
            st.markdown(
                f"""<div class="day-card">
                    <div class="day-label">{day_label}</div>
                    <div class="day-aqi" style="color:{color};">{avg_aqi:.0f}</div>
                    <div class="day-category">{category}</div>
                </div>""", unsafe_allow_html=True)


def render_hazard_alerts(forecast_df: pd.DataFrame):
    hazardous = forecast_df[forecast_df["predicted_aqi"] > 150]
    if len(hazardous) == 0:
        st.success(" No unhealthy AQI levels expected in the next 72 hours.")
        return
    worst_row = hazardous.loc[hazardous["predicted_aqi"].idxmax()]
    category, _ = classify_aqi(worst_row["predicted_aqi"])
    st.warning(
        f" Unhealthy AQI expected: peak of {worst_row['predicted_aqi']:.0f} ({category}) "
        f"around {worst_row['timestamp'].strftime('%a %b %d, %H:%M UTC')}. "
        f"Consider limiting outdoor activity during this period."
    )


def render_forecast_chart(forecast_df: pd.DataFrame):
    st.markdown('<p class="section-title">Hourly Forecast</p>', unsafe_allow_html=True)
    st.markdown('<p class="section-caption">Predicted continuous AQI trends across the next 72 hours</p>', unsafe_allow_html=True)

    colors = [classify_aqi(v)[1] for v in forecast_df["predicted_aqi"]]
    fig = go.Figure()

    
    fig.add_hrect(y0=0, y1=50, fillcolor="#10b981", opacity=0.04, line_width=0)
    fig.add_hrect(y0=51, y1=100, fillcolor="#f59e0b", opacity=0.04, line_width=0)
    fig.add_hrect(y0=101, y1=150, fillcolor="#f97316", opacity=0.05, line_width=0)
    fig.add_hrect(y0=151, y1=300, fillcolor="#ef4444", opacity=0.06, line_width=0)

    fig.add_trace(go.Scatter(
        x=forecast_df["timestamp"],
        y=forecast_df["predicted_aqi"],
        mode="lines+markers",
        line=dict(color="#0f766e", width=3, shape="spline", smoothing=0.6),
        marker=dict(color=colors, size=7, line=dict(width=1.5, color="#ffffff")),
        name="AQI Forecast",
        fill="tozeroy",
        fillcolor="rgba(15, 118, 110, 0.06)",
        hovertemplate="<b>%{x|%a %b %d, %H:%M UTC}</b><br>Predicted AQI: <b>%{y:.1f}</b><extra></extra>",
    ))

    fig.update_layout(
        plot_bgcolor="#ffffff",
        paper_bgcolor="#ffffff",
        font=dict(color="#0f172a", family="sans-serif", size=12),
        xaxis_title="",
        yaxis_title="Predicted AQI",
        height=390,
        margin=dict(l=10, r=10, t=15, b=10),
        xaxis=dict(
            gridcolor="#f1f5f9",
            showline=True,
            linecolor="#cbd5e1",
            linewidth=1,
            tickfont=dict(color="#64748b", size=11),
        ),
        yaxis=dict(
            gridcolor="#f1f5f9",
            showline=True,
            linecolor="#cbd5e1",
            linewidth=1,
            tickfont=dict(color="#64748b", size=11),
            zeroline=False,
        ),
        hovermode="x",
    )

    st.markdown('<div class="section-box">', unsafe_allow_html=True)
    st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})
    st.markdown('</div>', unsafe_allow_html=True)


def render_model_performance(model_bundles: dict):
    labels = {
        "short": ("Day 1", "1-24h"),
        "medium": ("Day 2", "25-48h"),
        "long": ("Day 3", "49-72h")
    }

    st.markdown(
        """
        <div class="section-box" style="height:100%;">
            <p class="section-title">Champion Models</p>
            <p class="section-caption">Active low-error models selected for each horizon</p>
        """,
        unsafe_allow_html=True
    )

    cols = st.columns(3)
    for col, (bucket_name, bundle) in zip(cols, model_bundles.items()):
        metrics = bundle["metrics"]
        day_title, horizon_sub = labels[bucket_name]
        with col:
            st.markdown(
                f"""
                <div class="model-bucket-box">
                    <div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:8px;">
                        <span class="model-bucket-title">{day_title}</span>
                        <span style="font-size:11px; color:#64748b; font-weight:600;">{horizon_sub}</span>
                    </div>
                    <div style="margin-top:8px;">
                        <div style="display:flex; justify-content:space-between; padding:5px 0; border-bottom:1px solid #e2e8f0;">
                            <span style="font-size:12px; color:#64748b;">R² Score</span>
                            <span style="font-size:13px; font-weight:700; color:#0f172a;">{metrics['r2']:.3f}</span>
                        </div>
                        <div style="display:flex; justify-content:space-between; padding:5px 0; border-bottom:1px solid #e2e8f0;">
                            <span style="font-size:12px; color:#64748b;">RMSE</span>
                            <span style="font-size:13px; font-weight:700; color:#0f172a;">{metrics['rmse']:.2f}</span>
                        </div>
                        <div style="display:flex; justify-content:space-between; padding:5px 0;">
                            <span style="font-size:12px; color:#64748b;">MAE</span>
                            <span style="font-size:13px; font-weight:700; color:#0f172a;">{metrics['mae']:.2f}</span>
                        </div>
                    </div>
                </div>
                """,
                unsafe_allow_html=True
            )
    st.markdown('</div>', unsafe_allow_html=True)


def render_shap_explanation(history_df: pd.DataFrame, model_bundles: dict):
    st.markdown(
        """
        <div class="section-box" style="height:100%;">
            <p class="section-title">Feature Attribution</p>
            <p class="section-caption">Key variables driving near-term Day 1 predictions</p>
        """,
        unsafe_allow_html=True
    )

    bundle = model_bundles["short"]
    if bundle["type"] != "sklearn":
        st.info("SHAP explanation is only available for tree/linear models, not the neural network.")
        st.markdown('</div>', unsafe_allow_html=True)
        return

    recent_rows = history_df[MODEL_FEATURE_COLUMNS].ffill().tail(150).copy()
    recent_rows["horizon"] = 12
    background_scaled = bundle["scaler"].transform(recent_rows)

    latest_row = history_df[MODEL_FEATURE_COLUMNS].ffill().iloc[[-1]].copy()
    latest_row["horizon"] = 12
    X_scaled = bundle["scaler"].transform(latest_row)

    try:
        explainer = shap.Explainer(bundle["model"], background_scaled)
        shap_values = explainer(X_scaled)
        importance_df = pd.DataFrame({
            "feature": list(latest_row.columns), "impact": shap_values.values[0],
        }).sort_values("impact", key=abs, ascending=True).tail(8)

        fig = go.Figure(go.Bar(
            x=importance_df["impact"],
            y=importance_df["feature"],
            orientation="h",
            marker=dict(
                color=["#e11d48" if v > 0 else "#0f766e" for v in importance_df["impact"]],
                line=dict(width=0),
            ),
            hovertemplate="Feature: <b>%{y}</b><br>Impact: <b>%{x:.3f}</b><extra></extra>",
        ))
        fig.update_layout(
            plot_bgcolor="#ffffff",
            paper_bgcolor="#ffffff",
            font=dict(color="#0f172a", family="sans-serif", size=12),
            xaxis_title="SHAP Impact Value",
            height=280,
            margin=dict(l=10, r=10, t=10, b=10),
            xaxis=dict(
                gridcolor="#f1f5f9",
                showline=True,
                linecolor="#cbd5e1",
                linewidth=1,
                tickfont=dict(color="#64748b", size=11),
                zeroline=True,
                zerolinecolor="#94a3b8",
                zerolinewidth=1.5,
            ),
            yaxis=dict(tickfont=dict(color="#334155", size=11)),
        )
        st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})
        st.caption("<span style='color:#e11d48;'>■</span> Increases predicted AQI &nbsp;|&nbsp; <span style='color:#0f766e;'>■</span> Decreases predicted AQI", unsafe_allow_html=True)
    except Exception as e:
        st.info(f"SHAP explanation unavailable right now: {e}")

    st.markdown('</div>', unsafe_allow_html=True)


def render_dashboard_page():
    st.markdown('<p class="brand-title" style="font-size:24px;">Live AQI Forecast</p>', unsafe_allow_html=True)
    st.markdown('<p class="section-caption">Updated on every request from the cloud model registry</p>', unsafe_allow_html=True)

    city_display_selected = st.selectbox("Select city", list(CITY_DISPLAY_NAMES.values()), index=0)
    city = [k for k, v in CITY_DISPLAY_NAMES.items() if v == city_display_selected][0]

    history_df, forecast_df, latest_aqi, latest_timestamp, model_bundles = get_forecast_for_city(city)

    st.markdown("<br>", unsafe_allow_html=True)
    render_current_metrics(latest_aqi, latest_timestamp, city_display_selected)

    st.markdown("<br>", unsafe_allow_html=True)
    render_day_summary(forecast_df)

    st.markdown("<br>", unsafe_allow_html=True)
    render_hazard_alerts(forecast_df)

    st.markdown("<br>", unsafe_allow_html=True)
    render_forecast_chart(forecast_df)

    st.markdown("<br>", unsafe_allow_html=True)
    col1, col2 = st.columns([1, 1])
    with col1:
        render_model_performance(model_bundles)
    with col2:
        render_shap_explanation(history_df, model_bundles)

    st.markdown("<br>", unsafe_allow_html=True)
    with st.expander("Raw forecast data"):
        display_df = forecast_df.copy()
        display_df["timestamp"] = display_df["timestamp"].dt.strftime("%Y-%m-%d %H:%M UTC")
        display_df["predicted_aqi"] = display_df["predicted_aqi"].round(1)
        st.table(display_df)

def render_trends_page():
    st.markdown('<p class="brand-title" style="font-size:24px;">Exploratory Data Analysis</p>', unsafe_allow_html=True)
    st.markdown('<p class="section-caption">Historical trends and patterns across the collected data</p>', unsafe_allow_html=True)

    city_display_selected = st.selectbox("Select city", list(CITY_DISPLAY_NAMES.values()), index=0, key="trends_city")
    city = [k for k, v in CITY_DISPLAY_NAMES.items() if v == city_display_selected][0]

    full_df = get_full_table()
    city_df = full_df[full_df["city"] == city].copy()
    city_df["timestamp"] = pd.to_datetime(city_df["timestamp"], utc=True)
    city_df = city_df.sort_values("timestamp")

    st.markdown("<div style='margin-top:20px;'></div>", unsafe_allow_html=True)
    st.markdown('<p class="section-title">AQI Trend (Last 30 Days)</p>', unsafe_allow_html=True)
    st.markdown('<p class="section-caption">Daily average AQI over the past month</p>', unsafe_allow_html=True)

    cutoff = city_df["timestamp"].max() - pd.Timedelta(days=30)
    recent = city_df[city_df["timestamp"] >= cutoff].copy()
    recent["date"] = recent["timestamp"].dt.date
    daily_avg = recent.groupby("date")["aqi"].mean().reset_index()

    full_date_range = pd.date_range(daily_avg["date"].min(), daily_avg["date"].max(), freq="D").date
    daily_avg = daily_avg.set_index("date").reindex(full_date_range).reset_index()
    daily_avg.columns = ["date", "aqi"]

    fig_trend = go.Figure()
    fig_trend.add_trace(go.Scatter(
        x=daily_avg["date"], y=daily_avg["aqi"],
        mode="lines+markers", line=dict(color="#0f766e", width=2),
        marker=dict(size=6, color="#0f766e"),
        connectgaps=False,
        hovertemplate="<b>%{x}</b><br>Avg AQI: <b>%{y:.1f}</b><extra></extra>",
    ))
    fig_trend.update_layout(
        plot_bgcolor="#ffffff", paper_bgcolor="#ffffff",
        font=dict(color="#0f172a", size=12), height=320,
        margin=dict(l=10, r=10, t=15, b=10),
        xaxis=dict(gridcolor="#f1f5f9", showline=True, linecolor="#cbd5e1", tickfont=dict(color="#64748b")),
        yaxis=dict(gridcolor="#f1f5f9", showline=True, linecolor="#cbd5e1", tickfont=dict(color="#64748b"), title="Avg AQI"),
    )
    st.markdown('<div class="section-box">', unsafe_allow_html=True)
    st.plotly_chart(fig_trend, use_container_width=True, config={"displayModeBar": False})
    st.markdown('</div>', unsafe_allow_html=True)

    st.markdown("<div style='margin-top:20px;'></div>", unsafe_allow_html=True)
    col1, col2 = st.columns(2)

    with col1:
        st.markdown('<p class="section-title">What Drives AQI Here?</p>', unsafe_allow_html=True)
        st.markdown('<p class="section-caption">Correlation between each pollutant/weather variable and AQI</p>', unsafe_allow_html=True)

        corr_cols = ["pm25", "pm10", "o3", "no2", "so2", "co", "temperature", "humidity", "wind_speed"]
        correlations = city_df[corr_cols + ["aqi"]].corr()["aqi"].drop("aqi").sort_values()

        fig_corr = go.Figure(go.Bar(
            x=correlations.values, y=correlations.index, orientation="h",
            marker=dict(color=["#e11d48" if v > 0 else "#0f766e" for v in correlations.values]),
        ))
        fig_corr.update_layout(
            plot_bgcolor="#ffffff", paper_bgcolor="#ffffff",
            font=dict(color="#0f172a", size=12), height=320,
            margin=dict(l=10, r=10, t=15, b=10),
            xaxis=dict(gridcolor="#f1f5f9", showline=True, linecolor="#cbd5e1", title="Correlation with AQI"),
            yaxis=dict(tickfont=dict(color="#334155")),
        )
        st.markdown('<div class="section-box">', unsafe_allow_html=True)
        st.plotly_chart(fig_corr, use_container_width=True, config={"displayModeBar": False})
        st.markdown('</div>', unsafe_allow_html=True)

    with col2:
        st.markdown('<p class="section-title">Daily Pattern</p>', unsafe_allow_html=True)
        st.markdown('<p class="section-caption">Average AQI by hour of day</p>', unsafe_allow_html=True)

        hourly_avg = city_df.groupby("hour")["aqi"].mean().reset_index()

        fig_hourly = go.Figure(go.Bar(
            x=hourly_avg["hour"], y=hourly_avg["aqi"],
            marker=dict(color="#0f766e"),
        ))
        fig_hourly.update_layout(
            plot_bgcolor="#ffffff", paper_bgcolor="#ffffff",
            font=dict(color="#0f172a", size=12), height=320,
            margin=dict(l=10, r=10, t=15, b=10),
            xaxis=dict(gridcolor="#f1f5f9", showline=True, linecolor="#cbd5e1", title="Hour of Day", dtick=2),
            yaxis=dict(
                gridcolor="#f1f5f9", showline=True, linecolor="#cbd5e1", title="Avg AQI",
                range=[hourly_avg["aqi"].min() - 10, hourly_avg["aqi"].max() + 10],
            ),
        )
        st.markdown('<div class="section-box">', unsafe_allow_html=True)
        st.plotly_chart(fig_hourly, use_container_width=True, config={"displayModeBar": False})
        st.markdown('</div>', unsafe_allow_html=True)
def render_methodology_page():
    st.markdown('<p class="brand-title" style="font-size:24px;">How the AQI System Works</p>', unsafe_allow_html=True)
    st.markdown('<p class="section-caption">System architecture, data pipelines, and methodology.</p>', unsafe_allow_html=True)

    steps = [
        ("1", "Hourly Feature Pipeline", "GitHub Actions runs hourly. Fetches weather and pollutant data for 4 cities from Open-Meteo, engineers time-based and lag features, writes to MongoDB Atlas."),
        ("2", "Daily Training Pipeline", "Runs once a day. Pulls the full feature history, builds a multi-horizon training set, and trains 4 candidate models (Ridge, Random Forest, Gradient Boosting, Neural Network) per forecast bucket."),
        ("3", "Champion Selection", "The model with the lowest RMSE on a time-based holdout test set is automatically saved as the active model for each horizon bucket — no hardcoded winner."),
        ("4", "Prediction Service", "The dashboard downloads the latest champion models and the most recent feature data, then generates a live 72-hour forecast on demand."),
    ]
    cols = st.columns(4)
    for col, (num, title, desc) in zip(cols, steps):
        with col:
            st.markdown(
                f"""<div class="flow-step">
                    <div class="flow-step-num">STEP {num}</div>
                    <div class="flow-step-title">{title}</div>
                    <div class="flow-step-desc">{desc}</div>
                </div>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    col1, col2 = st.columns(2)
    with col1:
        st.markdown('<p class="section-title">Model Strategy</p>', unsafe_allow_html=True)
        st.markdown(
            """<div class="metric-card">
                <p style="color:#0f172a;">Three separate models are trained for different forecast horizons
                (Day 1, Day 2, Day 3), since near-term and far-term AQI prediction are very different
                problems. No model is hardcoded as the winner — all four candidate types compete on
                every training run, evaluated with RMSE, MAE, and R². One shared model per horizon
                serves all four cities via a one-hot encoded city feature.</p>
            </div>""", unsafe_allow_html=True)
    with col2:
        st.markdown('<p class="section-title">Feature Engineering</p>', unsafe_allow_html=True)
        st.markdown(
            """<div class="metric-card">
                <p style="color:#0f172a;">Features include pollutant concentrations (PM2.5, PM10, O₃, NO₂, SO₂, CO),
                weather (temperature, humidity, pressure, wind), lagged AQI values (6h, 24h, 48h, 168h back),
                a 24h rolling average, and cyclical time encodings (hour/day/month).</p>
            </div>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown('<p class="section-title">Technologies Used</p>', unsafe_allow_html=True)

    technologies = [
        ("🐍", "Python"),
        ("📊", "Scikit-learn"),
        ("🧠", "TensorFlow"),
        ("🍃", "MongoDB Atlas"),
        ("⚙️", "GitHub Actions"),
        ("🎈", "Streamlit"),
        ("🌤️", "Open-Meteo API"),
        ("🔍", "SHAP"),
        ("🔧", "Git"),
    ]
    badges_html = '<div class="tech-grid">'
    for icon, name in technologies:
        badges_html += f'<div class="tech-badge"><span class="tech-icon">{icon}</span><span class="tech-name">{name}</span></div>'
    badges_html += '</div>'
    st.markdown(badges_html, unsafe_allow_html=True)


def render_about_page():
    st.markdown(
        """<div class="about-card">
            <p style="font-size:12px; font-weight:700; letter-spacing:0.08em; color:#5eead4; text-transform:uppercase; margin-bottom:8px;">Developed By</p>
            <p class="about-name" style="font-size:28px; font-weight:800; margin:0 0 6px 0;">Shahroz Khalid</p>
            <p style="color:#5eead4; font-size:14px; margin-bottom:16px;">Computer Science Undergraduate · Riphah International University, Islamabad</p>
            <p class="about-text" style="max-width:560px; line-height:1.7; font-size:15px;">
                Pearls AQI is a serverless, end-to-end machine learning pipeline forecasting the Air
                Quality Index for Islamabad, Rawalpindi, Lahore, and Faisalabad over the next 3 days.
                Built as part of the 10Pearls Shine Program, covering automated data collection,
                feature engineering, multi-model training and evaluation, and a live interactive dashboard.
            </p>
        </div>""",
        unsafe_allow_html=True,
    )

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown('<p class="section-title">Connect</p>', unsafe_allow_html=True)

    col1, col2, col3 = st.columns(3)
    with col1:
        st.link_button("🔗 GitHub", "https://github.com/shehroz53531/Pearls-AQI-Predictor", use_container_width=True)
    with col2:
        st.link_button("💼 LinkedIn", "https://linkedin.com/in/shahroz-khalid-919262332", use_container_width=True)
    with col3:
        st.link_button("🌐 Portfolio", "https://shehroz53531.github.io/shehroz53531.github.io./", use_container_width=True)

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown('<p class="section-title">Background</p>', unsafe_allow_html=True)

    col1, col2 = st.columns(2)
    with col1:
        st.markdown(
            """<div class="metric-card">
                <div class="metric-label">Recent Experience</div>
                <p style="color:#0f172a; margin-top:10px; line-height:1.7;">
                    Data Science Intern at 10Pearls · Machine Learning Intern at CodeAlpha ·
                    Teaching Assistant for Database Systems at Riphah International University
                </p>
            </div>""", unsafe_allow_html=True)
    with col2:
        st.markdown(
            """<div class="metric-card">
                <div class="metric-label">Focus Areas</div>
                <p style="color:#0f172a; margin-top:10px; line-height:1.7;">
                    Artificial Intelligence · Machine Learning · Deep Learning · Agentic AI · NLP · Data Structures & Algorithms ·
                    Backend Development · System Design . DataBases · Cloud Computing 
                </p>
            </div>""", unsafe_allow_html=True)


SCROLL_TOP_SCRIPT = """
<script>
function scrollAppToTop() {
    try {
        var doc = window.parent.document;
        window.parent.scrollTo({top: 0, left: 0, behavior: 'instant'});
        var candidates = [
            doc.querySelector('[data-testid="stAppViewContainer"]'),
            doc.querySelector('[data-testid="stMain"]'),
            doc.querySelector('section.main'),
            doc.querySelector('.main'),
            doc.scrollingElement,
            doc.documentElement,
            doc.body,
        ];
        candidates.forEach(function (el) {
            if (el && typeof el.scrollTop !== 'undefined') { el.scrollTop = 0; }
        });
    } catch (e) { /* ignore cross-origin or timing errors */ }
}
scrollAppToTop();
setTimeout(scrollAppToTop, 30);
setTimeout(scrollAppToTop, 100);
setTimeout(scrollAppToTop, 250);
setTimeout(scrollAppToTop, 500);
setTimeout(scrollAppToTop, 900);
</script>
"""


def main():
    components.html(SCROLL_TOP_SCRIPT, height=0)

    render_topbar()

    if st.session_state.page == "Home":
        render_home_page()
    elif st.session_state.page == "Dashboard":
        render_dashboard_page()
    elif st.session_state.page == "Trends":
        render_trends_page()    
    elif st.session_state.page == "Methodology":
        render_methodology_page()
    elif st.session_state.page == "About":
        render_about_page()

    components.html(SCROLL_TOP_SCRIPT, height=0)


if __name__ == "__main__":
    main()