"""This module contains utility functions for creating Plotly charts."""
import pandas as pd
import plotly.graph_objects as go


def display_trend_go_chart(df: pd.DataFrame, base_column='base_value'):
    """ Display a trend chart using Plotly. """
    df = df.copy()
    df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
    df = df.dropna(subset=['Date', base_column, 'fitted'])

    if df.empty:
        # st.warning("No valid data to plot.")
        return

    fig = go.Figure()

    # Add base value line
    fig.add_trace(go.Scatter(
        x=df['Date'],
        y=df[base_column],
        mode='lines',
        name='Base Value',
        line=dict(color='#14B3EB')
    ))

    # Add fitted value line
    fig.add_trace(go.Scatter(
        x=df['Date'],
        y=df['fitted'],
        mode='lines',
        name='Fitted Value',
        line=dict(color='#EB4C14')  # dashed line for contrast
    ))

    # Update layout
    fig.update_layout(
        title='Trend Chart',
        xaxis_title='Date',
        yaxis_title='Value',
        template='plotly_white',
        height=450,
        legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01)
    )

    return fig


def display_daily_annual_returns_chart(df: pd.DataFrame, annual_column='annual_base_return', date_column='Date', mean_value=None):
    """
    Display daily annual returns as a clean, compact Plotly chart (no legend or title).
    """
    df = df.copy()
    # df[date_column] = pd.to_datetime(df[date_column], errors='coerce')
    # df = df.dropna(subset=[date_column, annual_column])

    if df.empty:
        return

    if mean_value is None:
        mean_value = df[annual_column].mean()

    # Y-axis padding
    y_min = df[annual_column].min()
    y_max = df[annual_column].max()
    padding = 0.1 * (y_max - y_min)
    y_start = y_min - padding
    y_end = y_max + padding

    fig = go.Figure()

    # Annual return line
    fig.add_trace(go.Scatter(
        x=df[date_column],
        y=df[annual_column],
        mode='lines',
        line=dict(color='#14B3EB'),
        hoverinfo='x+y',
        showlegend=False
    ))

    # Mean line
    fig.add_trace(go.Scatter(
        x=[df[date_column].min(), df[date_column].max()],
        y=[mean_value, mean_value],
        mode='lines',
        line=dict(color='purple', dash='dash'),
        hoverinfo='skip',
        showlegend=False
    ))

    # Zero line
    fig.add_trace(go.Scatter(
        x=[df[date_column].min(), df[date_column].max()],
        y=[0, 0],
        mode='lines',
        line=dict(color='red', dash='dot'),
        hoverinfo='skip',
        showlegend=False
    ))

    fig.update_layout(
        margin=dict(t=10, b=30, l=30, r=10),  # trim top margin
        xaxis_title=None,
        yaxis_title='Annual Return',
        yaxis_tickformat='.1%',
        yaxis=dict(range=[y_start, y_end]),
        template='plotly_white',
        height=250,
    )

    return fig
