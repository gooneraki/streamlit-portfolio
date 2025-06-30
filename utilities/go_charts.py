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


def display_trend_go_chart_2(df: pd.DataFrame, value_column: str, fitted_column: str,
                             secondary_column: str = None,
                             title_name: str | None = None):
    """Display a clean trend chart using Plotly."""
    df = df.copy()

    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=df.index,
        y=df[value_column],
        mode='lines',
        name='Base Value',
        line=dict(color='#14B3EB')
    ))

    fig.add_trace(go.Scatter(
        x=df.index,
        y=df[fitted_column],
        mode='lines',
        name='Fitted Value',
        line=dict(color='#EB4C14', dash='dash')
    ))

    if secondary_column:
        fig.add_trace(go.Scatter(
            x=df.index,
            y=df[secondary_column],
            mode='lines',
            name=secondary_column,
            line=dict(color='#2ECC71', width=1),
            opacity=0.6,
            yaxis='y2'
        ))

    if secondary_column:
        fig.update_layout(
            yaxis2=dict(
                title='Secondary Value',
                overlaying='y',
                side='right',
                showgrid=False
            )
        )

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
        legend=dict(yanchor="top", y=0.90, xanchor="left", x=0.01),
        modebar=dict(remove=['zoom', 'pan', 'select', 'lasso',
                             'zoomIn', 'zoomOut', 'autoScale', 'resetScale']),
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


def display_symbol_metrics_chart(df: pd.DataFrame, symbol: str, metrics: list[str] = None, title: str = None):
    """
    Display a line chart for a selected symbol showing multiple metrics.

    Args:
        df (pd.DataFrame): Multi-index DataFrame with metrics and symbols
        symbol (str): The symbol to display
        metrics (list[str]): List of metrics to display. If None, shows all available metrics
        title (str): Optional title for the chart
    """
    if metrics is None:
        metrics = df.columns.get_level_values('Metric').unique()

    fig = go.Figure()

    colors = ['#14B3EB', '#EB4C14', '#2ECC71', '#F1C40F', '#9B59B6', '#E74C3C']

    for i, metric in enumerate(metrics):
        if metric in df.columns.get_level_values('Metric'):
            data = df[metric][symbol]
            fig.add_trace(go.Scatter(
                x=data.index,
                y=data,
                mode='lines',
                name=metric,
                line=dict(color=colors[i % len(colors)])
            ))

    fig.update_layout(
        title=dict(
            text=f"{symbol} - {title}" if title else symbol,
            x=0.5,
            y=0.98,
            xanchor='center',
            yanchor='top'
        ),
        xaxis_title='Date',
        yaxis_title='Value',
        template='plotly_white',
        height=450,
        margin=dict(t=40, b=0, l=0, r=0),
        legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01),
        showlegend=True,
        hovermode='x unified',
        modebar=dict(remove=['zoom', 'pan', 'select', 'lasso',
                     'zoomIn', 'zoomOut', 'autoScale', 'resetScale']),
        dragmode=False
    )

    return fig


def display_scatter_chart(df: pd.DataFrame, x_column: str, y_column: str, title_name: str | None = None):
    """Display a clean scatter chart using Plotly."""
    df = df.copy()
    df = df.dropna(subset=[x_column, y_column])

    if df.empty:
        return None

    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=df[x_column],
        y=df[y_column],
        mode='markers',
        name='Data Points',
        marker=dict(
            color='#14B3EB',
            size=6,
            opacity=0.7
        )
    ))

    fig.update_layout(
        title=dict(
            text=title_name,
            x=0.5,
            y=0.98,
            xanchor='center',
            yanchor='top'
        ) if title_name else None,
        xaxis_title=x_column,
        yaxis_title=y_column,
        template='plotly_white',
        height=450,
        margin=dict(t=40, b=0, l=0, r=0),
        legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01),
        showlegend=False,
        modebar=dict(remove=['zoom', 'pan', 'select', 'lasso',
                     'zoomIn', 'zoomOut', 'autoScale', 'resetScale']),
        dragmode=False
    )

    return fig


def display_efficient_frontier_chart(
    efficient_frontier_df: pd.DataFrame,
    random_df: pd.DataFrame = None,
    benchmark_return: float = None,
    benchmark_volatility: float = None,
    benchmark_name: str = "Benchmark",
    max_sharpe_return: float = None,
    max_sharpe_volatility: float = None,
    max_sharpe_name: str = "Max Sharpe Portfolio",
    same_risk_return: float = None,
    same_risk_volatility: float = None,
    same_risk_name: str = "Same Risk Portfolio",
    title_name: str | None = None
):
    """Display an efficient frontier chart with optional random portfolios, benchmark point, max Sharpe portfolio point, and same-risk portfolio point for comparison."""
    fig = go.Figure()

    # Add efficient frontier line
    fig.add_trace(go.Scatter(
        x=efficient_frontier_df['Volatility'],
        y=efficient_frontier_df['Returns'],
        mode='lines+markers',
        name='Efficient Frontier',
        line=dict(color='#14B3EB', width=3),
        marker=dict(size=6)
    ))

    # Add random portfolios if provided
    if random_df is not None and not random_df.empty:
        fig.add_trace(go.Scatter(
            x=random_df['Volatility'],
            y=random_df['Returns'],
            mode='markers',
            name='Random Portfolios',
            marker=dict(
                color='#E74C3C',
                size=4,
                opacity=0.6
            )
        ))

    # Add benchmark point if provided
    if benchmark_return is not None and benchmark_volatility is not None:
        fig.add_trace(go.Scatter(
            x=[benchmark_volatility],
            y=[benchmark_return],
            mode='markers',
            name=benchmark_name,
            marker=dict(
                symbol='star',
                size=15,
                color='#2ECC71',
                line=dict(color='#27AE60', width=2)
            )
        ))

    # Add max Sharpe portfolio point if provided
    if max_sharpe_return is not None and max_sharpe_volatility is not None:
        fig.add_trace(go.Scatter(
            x=[max_sharpe_volatility],
            y=[max_sharpe_return],
            mode='markers',
            name=max_sharpe_name,
            marker=dict(
                symbol='diamond',
                size=12,
                color='#9B59B6',
                line=dict(color='#8E44AD', width=2)
            )
        ))

    # Add same risk portfolio point if provided
    if same_risk_return is not None and same_risk_volatility is not None:
        fig.add_trace(go.Scatter(
            x=[same_risk_volatility],
            y=[same_risk_return],
            mode='markers',
            name=same_risk_name,
            marker=dict(
                symbol='triangle-up',
                size=12,
                color='#E67E22',
                line=dict(color='#D35400', width=2)
            )
        ))

    fig.update_layout(
        title=dict(
            text=title_name,
            x=0.5,
            y=0.98,
            xanchor='center',
            yanchor='top'
        ) if title_name else None,
        xaxis_title='Volatility (Risk)',
        yaxis_title='Expected Return',
        template='plotly_white',
        height=450,
        margin=dict(t=40, b=0, l=0, r=0),
        legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01),
        showlegend=True,
        modebar=dict(remove=['zoom', 'pan', 'select', 'lasso',
                     'zoomIn', 'zoomOut', 'autoScale', 'resetScale']),
        dragmode=False
    )

    return fig
