import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np

def plot_spread_analysis(S_range, spread_values, profit, greeks_dict, view_mode="Profit"):
    """
    Plot Spread Value, Profit, and Greeks using Plotly.
    """
    
    # Determine labels based on view mode
    if view_mode == "Profit":
        immediate_label = "Profit Now (mark-to-market PnL)"
        maturity_label = "Profit at Maturity"
        y_axis_label = "Profit ($)"
        title = "Spread Profit"
    else:  # Payoff
        immediate_label = "Payoff Now (mark-to-market)"
        maturity_label = "Payoff at Maturity"
        y_axis_label = "Payoff ($)"
        title = "Spread Payoff"

    fig_main = go.Figure()
    fig_main.add_trace(go.Scatter(x=S_range, y=spread_values, mode='lines', name=immediate_label, line=dict(color='#1f77b4', width=2)))
    fig_main.add_trace(go.Scatter(x=S_range, y=profit, mode='lines', name=maturity_label, line=dict(dash='dash', color='#ff7f0e')))
    
    
    breakeven_points = []
    for i in range(len(profit) - 1):
        if (profit[i] <= 0 and profit[i+1] > 0) or (profit[i] >= 0 and profit[i+1] < 0):
            if profit[i+1] != profit[i]:
                t = -profit[i] / (profit[i+1] - profit[i])
                breakeven_spot = S_range[i] + t * (S_range[i+1] - S_range[i])
                breakeven_points.append(breakeven_spot)
    

    if view_mode == "Profit":
        fig_main.add_hline(y=0, line_dash="dash", line_color="red", line_width=2, 
                           annotation_text="Break-even", annotation_position="top right")
        

        if breakeven_points:
            for idx, be_spot in enumerate(breakeven_points):
                fig_main.add_vline(x=be_spot, line_dash="dot", line_color="rgba(255, 0, 0, 0.3)", 
                                  annotation_text=f"breakeven: ${be_spot:.2f}", 
                                  annotation_position="top")
    else:
  
        fig_main.add_hline(y=0, line_dash="dot", line_color="gray", line_width=1)
    
    fig_main.update_layout(title=title, xaxis_title="Spot Price", yaxis_title=y_axis_label, template="plotly_dark")


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

def plot_simulation_results(pnl_data, process_name, mean_label=None):
    """
    Plot histogram of PnL from Monte Carlo.
    """
    fig = go.Figure()
    fig.add_trace(go.Histogram(
        x=pnl_data, 
        nbinsx=50, 
        name='PnL Distribution',
        marker_color='#1f77b4',
        opacity=0.75
    ))
    
    mean_val = np.mean(pnl_data)
    if mean_label is None:
        mean_label = f"Mean: {mean_val:.2f}"
    
    fig.add_vline(x=mean_val, line_dash="dash", line_color="red", annotation_text=mean_label)
    fig.add_vline(x=0, line_color="white", annotation_text="0")
    
    fig.update_layout(
        title=f"{process_name} - PnL Distribution",
        xaxis_title="PnL ($)",
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

def plot_heston_calibration(market_df, model_df, title="Heston Calibration: Market vs Model IV"):
    """
    Plot Market Implied Volatility vs Model Implied Volatility.
    market_df: DataFrame with ['Strike', 'IV']
    model_df: DataFrame with ['Strike', 'IV']
    """
    fig = go.Figure()
    
    # Market Data
    fig.add_trace(go.Scatter(
        x=market_df['Strike'], 
        y=market_df['IV'],
        mode='markers',
        name='Market IV',
        marker=dict(color='#1f77b4', size=8, symbol='circle-open'),
    ))
    
    # Model Data
    fig.add_trace(go.Scatter(
        x=model_df['Strike'], 
        y=model_df['IV'],
        mode='lines',
        name='Heston Model IV',
        line=dict(color='#ff7f0e', width=2),
    ))
    
    fig.update_layout(
        title=title,
        xaxis_title="Strike Price",
        yaxis_title="Implied Volatility",
        template="plotly_dark",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )
    
    return fig
