"""This module contains utility functions for creating Plotly charts."""
import pandas as pd
import plotly.graph_objects as go


def display_trend_go_chart(df: pd.DataFrame, base_column='base_value', fitted_column='fitted', title_name: str | None = None):
    """Display a clean trend chart using Plotly."""
    df = df.copy()
    df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
    df = df.dropna(subset=['Date', base_column, fitted_column])

    if df.empty:
        return

    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=df['Date'],
        y=df[base_column],
        mode='lines',
        name='Base Value',
        line=dict(color='#14B3EB')
    ))

    fig.add_trace(go.Scatter(
        x=df['Date'],
        y=df[fitted_column],
        mode='lines',
        name='Fitted Value',
        line=dict(color='#EB4C14', dash='dash')
    ))

    fig.update_layout(
        title=dict(
            text=title_name,
            x=0.5,
            y=0.98,
            xanchor='center',
            yanchor='top'
        ) if title_name else None,
        xaxis_title='Date',
        yaxis_title='Value',
        template='plotly_white',
        height=450,
        margin=dict(t=0, b=0, l=0, r=0),
        legend=dict(yanchor="top", y=0.90, xanchor="left", x=0.01)
    )

    return fig


def display_daily_annual_returns_chart(df: pd.DataFrame, annual_column='annual_base_return', date_column='Date', mean_value=None):
    """Display clean daily annual returns chart (no legend, title, or modebar)."""
    df = df.copy()
    df = df.dropna(subset=[date_column, annual_column])

    if df.empty:
        return

    if mean_value is None:
        mean_value = df[annual_column].mean()

    y_min = df[annual_column].min()
    y_max = df[annual_column].max()
    padding = 0.1 * (y_max - y_min)
    y_start = y_min - padding
    y_end = y_max + padding

    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=df[date_column],
        y=df[annual_column],
        mode='lines',
        line=dict(color='#14B3EB'),
        name='Annual Return',
    ))

    fig.add_trace(go.Scatter(
        x=[df[date_column].min(), df[date_column].max()],
        y=[mean_value, mean_value],
        mode='lines',
        line=dict(color='purple', dash='dash'),
        name='Mean return',
    ))

    fig.add_trace(go.Scatter(
        x=[df[date_column].min(), df[date_column].max()],
        y=[0, 0],
        mode='lines',
        line=dict(color='red', dash='dot'),
        name='Zero return',
    ))

    fig.update_layout(
        margin=dict(t=0, b=0, l=0, r=0),
        xaxis_title=None,
        yaxis_title='Annual Return',
        yaxis_tickformat='.1%',
        yaxis=dict(range=[y_start, y_end]),
        template='plotly_white',
        height=250,
        showlegend=False,

    )

    return fig
