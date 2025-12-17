import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np

def plot_spread_analysis(S_range, spread_values, profit, greeks_dict):
    """
    Plot Spread Value, Profit, and Greeks using Plotly.
    greeks_dict: {'Delta': [...], 'Gamma': [...], ...}
    """
    
    # 1. Main Payoff/Profit Plot
    fig_main = go.Figure()
    fig_main.add_trace(go.Scatter(x=S_range, y=spread_values, mode='lines', name='BSM Value'))
    fig_main.add_trace(go.Scatter(x=S_range, y=profit, mode='lines', name='ProfitAtMaturity', line=dict(dash='dash')))
    
    # Zero line
    fig_main.add_hline(y=0, line_dash="dot", annotation_text="Break-even", annotation_position="bottom right")
    
    fig_main.update_layout(title="Spread Analysis: Value & Profit", xaxis_title="Spot Price", yaxis_title="Value", template="plotly_dark")

    # 2. Greeks Subplots
    fig_greeks = make_subplots(rows=3, cols=2, subplot_titles=["Delta", "Gamma", "Theta", "Vega", "Rho"])
    
    greek_names = ["Delta", "Gamma", "Theta", "Vega", "Rho"]
    positions = [(1, 1), (1, 2), (2, 1), (2, 2), (3, 1)]
    
    for name, pos in zip(greek_names, positions):
        if name in greeks_dict:
            fig_greeks.add_trace(
                go.Scatter(x=S_range, y=greeks_dict[name], name=name),
                row=pos[0], col=pos[1]
            )
            
    fig_greeks.update_layout(height=800, title="Greeks Analysis", template="plotly_dark", showlegend=False)
    
    return fig_main, fig_greeks

def plot_simulation_results(pnl_percent, process_name):
    """
    Plot histogram of PnL from Monte Carlo.
    """
    fig = go.Figure()
    fig.add_trace(go.Histogram(
        x=pnl_percent, 
        nbinsx=50, 
        name='PnL Distribution',
        marker_color='#1f77b4',
        opacity=0.75
    ))
    
    mean_val = np.mean(pnl_percent)
    fig.add_vline(x=mean_val, line_dash="dash", line_color="red", annotation_text=f"Mean: {mean_val:.2f}%")
    fig.add_vline(x=0, line_color="white", annotation_text="0%")
    
    fig.update_layout(
        title=f"{process_name} - PnL Distribution",
        xaxis_title="PnL (%)",
        yaxis_title="Frequency",
        template="plotly_dark"
    )
    return fig

def plot_efficient_frontier(ef_risks, ef_rets, current_risk=None, current_ret=None):
    """
    Plot Efficient Frontier.
    """
    fig = go.Figure()
    
    # Frontier
    fig.add_trace(go.Scatter(
        x=ef_risks, y=ef_rets, 
        mode='lines', 
        name='Efficient Frontier',
        line=dict(color='cyan', width=3)
    ))
    
    # Current Portfolio Marker
    if current_risk is not None and current_ret is not None:
        fig.add_trace(go.Scatter(
            x=[current_risk], y=[current_ret],
            mode='markers',
            marker=dict(size=12, color='red', symbol='star'),
            name='Optimal Portfolio'
        ))
        
    fig.update_layout(
        title="Efficient Frontier",
        xaxis_title="Volatility (Std Dev)",
        yaxis_title="Expected Return",
        template="plotly_dark"
    )
    return fig
