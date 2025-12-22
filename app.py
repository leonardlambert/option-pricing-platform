#imports
import streamlit as st
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import plotly.graph_objects as go

#local imports
from src.dashboard_utils import initialize_session_state, add_strategy_to_book, reset_book, delete_strategy, interpolate_volatility
from pricing.pricing import black_scholes_price, compute_greeks, implied_volatility, calculate_pnl_attribution
from pricing.FFT_pricer import fft_pricer
from pricing.characteristic_functions import phi_bsm, phi_vg, phi_merton
from src.visualizer import plot_spread_analysis, plot_simulation_results, plot_efficient_frontier
from src.simulation import simulate_gbm_paths, simulate_vg_paths, simulate_mjd_paths
from src.market_data import get_option_aggregates, get_option_previous_close, get_stock_history_vol, get_underlying_history_range, validate_api_key, get_available_dates, get_all_preloaded_options
from pricing.finite_differences import compute_greeks_fd

#config
st.set_page_config(page_title="Option Pricing & Risk Management", layout="wide", page_icon="📈", initial_sidebar_state="collapsed")
initialize_session_state()

#style
st.markdown("""
    <style>
        /* Serious Font Import (Inter) */
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600&display=swap');
        
        html, body, [class*="css"]  {
            font-family: 'Inter', 'Segoe UI', 'Roboto', sans-serif;
        }

        .block-container {
            padding-top: 3rem; /* Moderate spacing */
            padding-bottom: 0rem;
        }
        h1 {
            font-family: 'Inter', 'Segoe UI', sans-serif;
            font-weight: 600;
            font-size: 1.5rem !important; /* Smaller, serious title */
            margin-top: 0rem !important;
            margin-bottom: 0rem !important;
            padding-bottom: 0rem !important;
        }
        
        /* Increase space between title and tabs slightly */
        .stTabs {
            margin-top: 1rem;
        }
    </style>
""", unsafe_allow_html=True)

#sidebar
with st.sidebar:
    st.subheader("Data Mode Settings")
    
    #data mode selector
    current_mode = st.session_state.get("data_mode", "Live API")
    new_mode = st.radio(
        "Data Source",
        ["Live API", "Preloaded Dataset"],
        index=0 if current_mode == "Live API" else 1,
        help="Switch between market data and offline preloaded dataset."
    )
    
    if new_mode != current_mode:
        st.session_state["data_mode"] = new_mode
        #clear data on switch
        for k in ["center_data", "smile_data", "surface_data", "custom_data_table", "market_mode", "underlying_S", "underlying_HV"]:
            if k in st.session_state: del st.session_state[k]
        st.rerun()
    
    st.divider()
    
    #live api key section
    if st.session_state["data_mode"] == "Live API":
        st.subheader("Massive / Polygon API key")
        user_key = st.text_input("Enter API Key", type="password", help="Enter your Massive API key here to fetch live market data.")
        
        if st.button("Validate & Apply Key"):
            if not user_key:
                st.error("Please enter a key.")
            else:
                with st.spinner("Validating..."):
                    is_valid, msg = validate_api_key(user_key)  
                    if is_valid:
                        st.session_state["user_api_key"] = user_key
                        st.success("API Key validated and applied!")
                        st.rerun()
                    else:
                        st.error(msg)
        
        if "user_api_key" in st.session_state:
            st.info("Currently using: **Applied API Key**")
            if st.button("Clear Applied Key"):
                del st.session_state["user_api_key"]
                st.rerun()
        else:
            st.warning("⚠️ No API Key Applied. Live data fetch will fail.")
            st.caption("Please enter a valid Massive API key to use Live mode.")
    else:
        st.info("Currently using: **Preloaded Dataset**")
        st.caption("No API key required in this mode.")

#apply light theme css
st.markdown("""
    <style>
        .stApp {
            background-color: white;
            color: black;
        }
        [data-testid="stHeader"] {
            background-color: white;
        }
        [data-testid="stSidebar"] {
            background-color: #f8f9fa;
        }
        .stMarkdown, .stText, h1, h2, h3, h4, h5, h6, label, .stMetric {
            color: black !important;
        }
        /* Metric labels specifically */
        [data-testid="stMetricLabel"] {
            color: #262730 !important;
        }
        .stButton>button {
            color: black;
            border-color: #ccc;
        }
    </style>
""", unsafe_allow_html=True)

st.title("Option Pricing & Scenario Analysis Platform")

#sections list
tabs = st.tabs(["Option Pricing & Greeks", "Strategy Builder", "Strategy PnL Distribution", "Strategy Stress Testing", "Volatility Smile", "Volatility Surface"])

#TAB 1 : single option pricing
with tabs[0]:
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        S = st.number_input("Spot Price (S)", value=100.0)
        K = st.number_input("Strike Price (K)", value=100.0)
    with col2:
        T = st.number_input("Time to Maturity (T)", value=1.0)
        r = st.number_input("Risk-Free Rate (r)", value=0.05)
    with col3:
        sigma = st.number_input("Volatility (σ)", value=0.2)
        option_type = st.selectbox("Type", ["C", "P"])
    with col4:
        model = st.selectbox("Pricing Model", ["Black-Scholes-Merton", "Variance Gamma", "Merton"])
        
        #inof about the models
        model_info = {
            "Black-Scholes-Merton": "**Stochastic Family:** GBM | **Pricing Logic:** Closed Form",
            "Variance Gamma": "**Stochastic Family:** Lévy Process | **Pricing Logic:** Fast Fourier Transform",
            "Merton": "**Stochastic Family:** Jump Diffusion | **Pricing Logic:** Fast Fourier Transform"
        }
        st.caption(f"{model_info[model]}")

    if st.button("Calculate Price"):
        st.divider()
        c1, c2 = st.columns(2)
        
        #pricing part
        price = 0.0
        model_params = {}
        use_fd_greeks = False
        
        if model == "Black-Scholes-Merton":
            price = black_scholes_price(S, K, T, r, sigma, option_type)
        elif model == "Variance Gamma":
            #hardcoded VGparams for now
            theta, nu = -0.1, 0.2 
            price = fft_pricer(K, S, T, r, phi_vg, args=(sigma, theta, nu), call=(option_type=="C"))
            st.caption(f"Used VG Parameters: θ = {theta}, ν = {nu}")
            model_params = {'theta': theta, 'nu': nu}
            use_fd_greeks = True
        elif model == "Merton":
            #hardcoded merton params for now
            lamb, mu_j, sigma_j = 0.1, -0.05, 0.2
            price = fft_pricer(K, S, T, r, phi_merton, args=(sigma, lamb, mu_j, sigma_j), call=(option_type=="C"))
            st.caption(f"Used Merton Parameters: λ = {lamb}, μ_j = {mu_j}, σ_j = {sigma_j}")
            model_params = {'lamb': lamb, 'mu_j': mu_j, 'sigma_j': sigma_j}
            use_fd_greeks = True
            
        #compute greeks
        if use_fd_greeks:
            #finite difference greeks for exotic price processes
            model_name = "VG" if "Variance Gamma" in model else "Merton"
            greeks = compute_greeks_fd(S, K, T, r, sigma, option_type, model_name, model_params)
            delta, gamma, theta_g, vega, rho = greeks['delta'], greeks['gamma'], greeks['theta'], greeks['vega'], greeks['rho']
        else:
            #for BSM use closed form greeks
            delta, gamma, theta_g, vega, rho = compute_greeks(S, K, T, r, sigma, option_type)
        
        with c1:
            st.metric("Option Price", f"${price:.4f}")
            
        with c2:
            #FD disclaimer
            if use_fd_greeks:
                st.caption("Greeks computed using finite differences", help="Finite difference approximation is used for models without analytical Greek formulas / Exotic Price Processes. Results are numerical approximations.")
            
            g_col1, g_col2 = st.columns(2)
            g_col1.write(f"**Delta**: {delta:.4f}")
            g_col1.write(f"**Gamma**: {gamma:.4f}")
            g_col1.write(f"**Theta**: {theta_g:.4f}")
            g_col2.write(f"**Vega**: {vega:.4f}")
            g_col2.write(f"**Rho**: {rho:.4f}")

#TAB 2 : strategy builder
with tabs[1]:
    
    col_setup, col_plot = st.columns([1, 2])
    
    with col_setup:
        st.subheader("Market Parameters")
        
        #fixed to BSM for now
        pricing_model = "Black-Scholes"
        
        S0_spread = st.number_input("Spot Price (S)", value=100.0, key="s_spread")
        T_spread = st.number_input("Time to Maturity (T)", value=1.0, key="t_spread")
        r_spread = st.number_input("Risk-Free Rate (r)", value=0.05, key="r_spread")
        sigma_spread = st.number_input("Volatility (σ)", value=0.2, key="v_spread")
        
        st.subheader("Strategy Legs")
        num_legs = st.number_input("Legs Count", 1, 6, 2)
        legs = []
        for i in range(num_legs):
            c_type, c_strike, c_pos = st.columns(3)
            l_type = c_type.selectbox("Type", ["C", "P"], key=f"l_type_{i}")
            l_strike = c_strike.number_input("Strike", 50.0, 200.0, 100.0, key=f"l_strike_{i}")
            l_pos = c_pos.selectbox("Position", [1, -1], format_func=lambda x: "Long" if x==1 else "Short", key=f"l_pos_{i}")
            legs.append({"type": l_type, "strike": l_strike, "position": l_pos})
            
        st.divider()
        strat_name = st.text_input("Strategy Name", value="My Strategy", placeholder="e.g. Iron Condor")
        if st.button("💾 Save Strategy to Book"):

            save_legs = []
            for leg in legs:
                p = black_scholes_price(S0_spread, leg["strike"], T_spread, r_spread, sigma_spread, leg["type"])
                leg_with_premium = leg.copy()
                leg_with_premium["premium"] = round(p, 4)
                save_legs.append(leg_with_premium)
            
            add_strategy_to_book(S0_spread, T_spread, r_spread, sigma_spread, save_legs, name=strat_name)
            st.success(f"Strategy '{strat_name}' saved with leg premiums!")

    with col_plot:
        
        S_range = np.linspace(S0_spread * 0.5, S0_spread * 1.5, 50) #linspace for FFT computation
        spread_vals = np.zeros_like(S_range)
        payoff_vals = np.zeros_like(S_range)
        
        #aggregate greeks (only for BSM)
        greeks_agg = {k: np.zeros_like(S_range) for k in ["Delta", "Gamma", "Theta", "Vega", "Rho"]}
        net_premium = 0.0
        
        #progress bar if needed
        progress_text = "Pricing Spread..."
        my_bar = st.progress(0, text=progress_text)
        
        total_steps = len(legs) * len(S_range) + len(legs) # approx
        step_ctr = 0
        
        for leg in legs:
            
            leg_price = black_scholes_price(S0_spread, leg["strike"], T_spread, r_spread, sigma_spread, leg["type"])
            net_premium += leg["position"] * leg_price
            
            for idx, s_i in enumerate(S_range):

                step_ctr += 1
                if step_ctr % 10 == 0:
                    my_bar.progress(min(step_ctr / total_steps, 1.0), text=progress_text)

                p_i = black_scholes_price(s_i, leg["strike"], T_spread, r_spread, sigma_spread, leg["type"])
                d, g, th, v, rh = compute_greeks(s_i, leg["strike"], T_spread, r_spread, sigma_spread, leg["type"])
                greeks_agg["Delta"][idx] += leg["position"] * d
                greeks_agg["Gamma"][idx] += leg["position"] * g
                greeks_agg["Theta"][idx] += leg["position"] * th
                greeks_agg["Vega"][idx] += leg["position"] * v
                greeks_agg["Rho"][idx] += leg["position"] * rh
                
                spread_vals[idx] += leg["position"] * p_i
                
                intrinsic = max(0, s_i - leg["strike"]) if leg["type"] == "C" else max(0, leg["strike"] - s_i)
                payoff_vals[idx] += leg["position"] * intrinsic

        my_bar.empty()
        
        # Adjust spread values to show profit (value - premium)
        immediate_profit_vals = spread_vals - net_premium
        profit_at_maturity_vals = payoff_vals - net_premium
        
        # Get view mode from session state or default to Profit
        if "view_mode" not in st.session_state:
            st.session_state.view_mode = "Profit"
        
        # Prepare data based on view mode
        if st.session_state.view_mode == "Profit":
            immediate_vals = immediate_profit_vals
            maturity_vals = profit_at_maturity_vals
        else:  # Payoff
            immediate_vals = spread_vals
            maturity_vals = payoff_vals
        
        fig1, fig2 = plot_spread_analysis(S_range, immediate_vals, maturity_vals, greeks_agg, st.session_state.view_mode)
        
        st.plotly_chart(fig1, key="spread_plot_1", width='stretch')
        
        # Toggle for Profit vs Payoff view (below the graph)
        view_mode = st.radio("Display Mode", ["Profit", "Payoff"], index=0 if st.session_state.view_mode == "Profit" else 1, horizontal=True, 
                            help="Profit: adjusted for premium paid/received | Payoff: intrinsic value only", key="view_mode_radio")
        
        # Update session state if changed
        if view_mode != st.session_state.view_mode:
            st.session_state.view_mode = view_mode
            st.rerun()
        
        with st.expander("View Greeks"):
            st.plotly_chart(fig2, key="spread_plot_2", width='stretch')
            
        # Display net premium with DEBIT/CREDIT label
        premium_label = "(DEBIT)" if net_premium > 0 else "(CREDIT)" if net_premium < 0 else ""
        st.info(f"Net Premium (Black-Scholes): ${abs(net_premium):.2f} {premium_label}")

#TAB 3 : strategy PnL distribution
with tabs[2]:
    
    if not st.session_state.book:
        st.warning("Book is empty. Add strategies in 'Strategy Builder' tab first.")
    else:
        
        if "selected_strategies" not in st.session_state:
            st.session_state.selected_strategies = [True] * len(st.session_state.book)
        
        if len(st.session_state.selected_strategies) != len(st.session_state.book):
            st.session_state.selected_strategies = [True] * len(st.session_state.book)

        st.info(f"Total Strategies: {len(st.session_state.book)}")

        entries_to_delete = []
        
        for idx, strat in enumerate(st.session_state.book):
            name = strat.get("name", "Untitled")
            timestamp = strat.get("timestamp", "N/A")
            params = f"S0={strat['S0']}, T={strat['T']}, r={strat['r']}, σ={strat['sigma']}"
            
            with st.expander(f"📍 **{name}** | 🕒 {timestamp}"):
                c_sel, c_details, c_action = st.columns([0.5, 3.5, 1])
                with c_sel:
                    current_sel = st.checkbox("Select Strategy", value=st.session_state.selected_strategies[idx], key=f"pnl_sel_{idx}_{strat['id']}", label_visibility="collapsed")
                    if current_sel != st.session_state.selected_strategies[idx]:
                        st.session_state.selected_strategies[idx] = current_sel
                        if "pnl_results" in st.session_state:
                            del st.session_state["pnl_results"]
                        st.rerun()
                with c_details:
                    st.write(f"**Parameters**: {params}")
                    
                    leg_data = []
                    for leg in strat["legs"]:
                        pos_str = "Long (+1)" if leg["position"] == 1 else "Short (-1)"
                        type_str = leg["type"].title()
                        strike = leg["strike"]
                        prem = leg.get("premium", 0.0)
                        leg_data.append({"Position": pos_str, "Type": type_str, "Strike": strike, "Premium": prem})
                    st.table(pd.DataFrame(leg_data))
                
                with c_action:
                    if st.button("Delete Strategy", key=f"del_{idx}"):
                        entries_to_delete.append(idx)
        
        if entries_to_delete:
            for i in sorted(entries_to_delete, reverse=True):
                delete_strategy(i)
                
                if "selected_strategies" in st.session_state and len(st.session_state.selected_strategies) > i:
                    st.session_state.selected_strategies.pop(i)
            if "pnl_results" in st.session_state:
                del st.session_state["pnl_results"]
            st.rerun()

        if st.button("Clear Entire Book"):
            reset_book()
            if "selected_strategies" in st.session_state:
                st.session_state.selected_strategies = []
            if "pnl_results" in st.session_state:
                del st.session_state["pnl_results"]
            st.rerun()
            
        st.divider()
        #simulation part !!! needs at least one strategy selected
        if st.session_state.book:
            selected_indices = [i for i, sel in enumerate(st.session_state.selected_strategies) if sel]
            
            if not selected_indices:
                st.warning("Please select at least one strategy to run simulation.")
            else:
                # Lock parameters to defaults
                steps = 50
                T_sim = 1.0
                process = "GBM"
                
                n_paths = st.number_input("Paths", 1000, 50000, 5000, key="pnl_n_paths")
                
                # Clear results if paths changed or if selected strategies changed
                if "last_n_paths" not in st.session_state:
                    st.session_state.last_n_paths = n_paths
                if "last_selected_indices" not in st.session_state:
                    st.session_state.last_selected_indices = selected_indices.copy()
                
                # Check if anything changed
                paths_changed = st.session_state.last_n_paths != n_paths
                selection_changed = st.session_state.last_selected_indices != selected_indices
                
                if paths_changed or selection_changed:
                    st.session_state.last_n_paths = n_paths
                    st.session_state.last_selected_indices = selected_indices.copy()
                    if "pnl_results" in st.session_state:
                        del st.session_state["pnl_results"]
                        st.info("⚠️ Parameters changed. Click 'Run Monte Carlo' to generate new results.")
                    
                if st.button("Run Monte Carlo", key="run_mc_pnl"):
                    
                    first_strat = st.session_state.book[selected_indices[0]]
                    S0_sim = first_strat["S0"]
                    r_sim = first_strat["r"]
                    sigma_sim = first_strat["sigma"]
                    
                    with st.spinner("Simulating..."):
                        if process == "GBM":
                            paths = simulate_gbm_paths(S0_sim, r_sim, sigma_sim, T_sim, steps, n_paths)
                        elif process == "Variance Gamma":
                            paths = simulate_vg_paths(S0_sim, r_sim, sigma_sim, theta=-0.1, nu=0.2, T=T_sim, steps=steps, n_paths=n_paths)
                        else:
                            paths = simulate_mjd_paths(S0_sim, r_sim, sigma_sim, lamb=0.1, mu_j=-0.05, sigma_j=0.2, T=T_sim, steps=steps, n_paths=n_paths)
                        
                        S_T = paths[:, -1]
                        pnl = np.zeros(n_paths)
                        initial_book_value = 0.0
                        
                        # Calculate initial book value (premium paid/received at t=0)
                        # Positive for long positions (you pay), negative for short positions (you receive)
                        for idx in selected_indices:
                            strat = st.session_state.book[idx]
                            for leg in strat["legs"]:
                                #use saved premium if available, else fallback
                                p = leg.get("premium", black_scholes_price(strat["S0"], leg["strike"], strat["T"], strat["r"], strat["sigma"], leg["type"]))
                                initial_book_value += leg["position"] * p
                        
                        # Calculate PnL for each simulated path
                        for i in range(n_paths):
                            spot = S_T[i]
                            val_t = 0.0
                            for idx in selected_indices:
                                strat = st.session_state.book[idx]
                                remaining_t = max(0, strat["T"] - T_sim)
                                for leg in strat["legs"]:
                                    if remaining_t == 0:
                                        # At expiration: use intrinsic value
                                        val = max(0, spot - leg["strike"]) if leg["type"] == "C" else max(0, leg["strike"] - spot)
                                    else:
                                        # Before expiration: use Black-Scholes value
                                        val = black_scholes_price(spot, leg["strike"], remaining_t, strat["r"], strat["sigma"], leg["type"])
                                    val_t += leg["position"] * val
                            # PnL = Current value - Initial premium paid/received
                            pnl[i] = val_t - initial_book_value
                        
                        # Calculate percentage returns
                        # Use absolute value of initial book value to avoid distorted percentages
                        # when combining debit and credit spreads
                        abs_initial_value = abs(initial_book_value)
                        
                        if abs_initial_value > 0.01:  # Only calculate % if meaningful initial investment
                            pnl_percent = (pnl / abs_initial_value) * 100
                        else:
                            # If initial book value is near zero (e.g., balanced debit/credit spreads),
                            # percentage returns don't make sense - just use dollar PnL
                            pnl_percent = pnl  # Will display as dollar amounts
                        
                        st.session_state["pnl_results"] = {
                            "pnl_percent": pnl_percent,
                            "process": process,
                            "pnl": pnl,
                            "initial_value": initial_book_value,
                            "abs_initial_value": abs_initial_value,
                            "use_percentage": abs_initial_value > 0.01
                        }
                        st.success(f"✅ Simulation complete! ({n_paths:,} paths)")
                        st.rerun()

                if "pnl_results" in st.session_state:
                    results = st.session_state["pnl_results"]
                    use_pct = results.get("use_percentage", True)
                    abs_init = results.get("abs_initial_value", 1.0)
                    init_val = results.get("initial_value", 0.0)
                    
                    # Display initial investment info
                    if init_val > 0:
                        st.info(f"📊 Initial Investment: ${abs_init:.2f} (Debit - you paid)")
                    elif init_val < 0:
                        st.info(f"📊 Initial Investment: ${abs_init:.2f} (Credit - you received)")
                    else:
                        st.info(f"📊 Initial Investment: ~$0 (Balanced position)")
                    
                    st.plotly_chart(plot_simulation_results(results["pnl_percent"], results["process"]))
                    
                    # Calculate metrics
                    mean_pnl = np.mean(results['pnl'])
                    var_95 = np.percentile(results['pnl'], 5)
                    cvar_95 = np.mean(results['pnl'][results['pnl'] <= var_95])
                    
                    c_metrics1, c_metrics2, c_metrics3 = st.columns(3)
                    with c_metrics1:
                        if use_pct:
                            st.metric("Mean PnL", f"${mean_pnl:.2f}", f"{(mean_pnl/abs_init)*100:.2f}%")
                        else:
                            st.metric("Mean PnL", f"${mean_pnl:.2f}")
                    with c_metrics2:
                        if use_pct:
                            st.metric("VaR (95%)", f"${var_95:.2f}", f"{(var_95/abs_init)*100:.2f}%")
                        else:
                            st.metric("VaR (95%)", f"${var_95:.2f}")
                    with c_metrics3:
                        if use_pct:
                            st.metric("CVaR (95%)", f"${cvar_95:.2f}", f"{(cvar_95/abs_init)*100:.2f}%")
                        else:
                            st.metric("CVaR (95%)", f"${cvar_95:.2f}")

#TAB 4: strategy stress testing
with tabs[3]:
    if not st.session_state.book:
        st.warning("Book is empty. Add strategies in 'Strategy Builder' tab first.")
    else:
        
        if "selected_strategies" not in st.session_state:
            st.session_state.selected_strategies = [True] * len(st.session_state.book)
        
        if len(st.session_state.selected_strategies) != len(st.session_state.book):
            st.session_state.selected_strategies = [True] * len(st.session_state.book)

        st.info(f"Total Strategies: {len(st.session_state.book)}")

        for idx, strat in enumerate(st.session_state.book):
            name = strat.get("name", "Untitled")
            with st.expander(f"📍 **{name}**"):
                c_sel_st, c_details_st = st.columns([0.5, 4.5])
                with c_sel_st:
                    current_sel_st = st.checkbox("Select Strategy", value=st.session_state.selected_strategies[idx], key=f"stress_sel_{idx}_{strat['id']}", label_visibility="collapsed")
                    if current_sel_st != st.session_state.selected_strategies[idx]:
                        st.session_state.selected_strategies[idx] = current_sel_st
                        if "pnl_results" in st.session_state:
                            del st.session_state["pnl_results"]
                        if "stress_results" in st.session_state:
                            del st.session_state["stress_results"]
                        st.rerun()
                with c_details_st:
                    # Leg Details
                    leg_data = []
                    for leg in strat["legs"]:
                        pos_str = "Long (+1)" if leg["position"] == 1 else "Short (-1)"
                        type_str = leg["type"].title()
                        strike = leg["strike"]
                        prem = leg.get("premium", 0.0)
                        leg_data.append({"Position": pos_str, "Type": type_str, "Strike": strike, "Premium": prem})
                    st.table(pd.DataFrame(leg_data))

        st.divider()
        
        c_p1, c_p2 = st.columns([1, 2])
        with c_p1:
            st.subheader("Scenario Parameters")
            
            #quick scenarios
            st.caption("Quick Scenarios")
            qs_cols = st.columns(5)
            if qs_cols[0].button("Spot Shock"):
                st.session_state.stress_spot = -10.0
                st.rerun()
            if qs_cols[1].button("Vol Shock"):
                st.session_state.stress_vol = 10.0
                st.rerun()
            if qs_cols[2].button("Week Decay"):
                st.session_state.stress_decay = 7
                st.rerun()
            if qs_cols[3].button("Equity Crash"):
                st.session_state.stress_spot = -10.0
                st.session_state.stress_vol = 10.0
                st.rerun()
            if qs_cols[4].button("Equity Rally"):
                st.session_state.stress_spot = 10.0
                st.session_state.stress_vol = -5.0
                st.rerun()
            
            st.divider()

            #Initialize keys if missing
            if "stress_spot" not in st.session_state: st.session_state.stress_spot = 0.0
            if "stress_vol" not in st.session_state: st.session_state.stress_vol = 0.0
            if "stress_decay" not in st.session_state: st.session_state.stress_decay = 0

            spot_change_pct = st.slider("Underlying Price Change (%)", -20.0, 20.0, key="stress_spot", step=1.0)
            vol_change_pts = st.slider("Volatility Change (pts)", -20.0, 20.0, key="stress_vol", step=1.0)
            time_decay_days = st.slider("Time Decay (Days)", 0, 30, key="stress_decay", step=1)
            
            spot_change = spot_change_pct / 100.0
            vol_change = vol_change_pts / 100.0
            time_decay = time_decay_days

        with c_p2:
            st.subheader("Scenario Results")
            selected_indices = [i for i, sel in enumerate(st.session_state.selected_strategies) if sel]
            
            if not selected_indices:
                st.warning("Please select at least one strategy to run stress test.")
            else:
                #variation computation
                price_before = 0.0
                price_after = 0.0
                days_to_years = time_decay / 365.0
                
                for idx in selected_indices:
                    strat = st.session_state.book[idx]
                    for leg in strat["legs"]:
                        
                        price_before += leg["position"] * leg.get("premium", 0.0)
                        
                        new_spot = strat["S0"] * (1 + spot_change)
                        new_vol = max(0.01, strat["sigma"] + vol_change)
                        new_t = max(1e-4, strat["T"] - days_to_years)
                        
                        p_after = black_scholes_price(new_spot, leg["strike"], new_t, strat["r"], new_vol, leg["type"])
                        price_after += leg["position"] * p_after

                variation = price_after - price_before
                st.metric("Spread Price Variation", f"${variation:.2f}", help="Price After Stress - Price Before Stress")
                
                c_m1, c_m2 = st.columns(2)
                with c_m1:
                    st.write(f"**Old Price**: ${price_before:.2f}")
                with c_m2:
                    st.write(f"**New Price**: ${price_after:.2f}")

                #PNL attribution
                st.divider()
                st.subheader("PnL Attribution (Using Taylor Approximation)")
                
                attr_totals = {"Delta": 0.0, "Gamma": 0.0, "Vega": 0.0, "Theta": 0.0}
                
                for idx in selected_indices:
                    strat = st.session_state.book[idx]
                    dS = strat["S0"] * spot_change
                    dVol = vol_change
                    dT = days_to_years
                    
                    for leg in strat["legs"]:
                        attr = calculate_pnl_attribution(
                            strat["S0"], leg["strike"], strat["T"], strat["r"], strat["sigma"], 
                            leg["type"], dS, dVol, dT
                        )
                        for k in attr_totals:
                            attr_totals[k] += leg["position"] * attr[k]
                
                pnl_explained = sum(attr_totals.values())
                residual = variation - pnl_explained
                
                c_attr1, c_attr2, c_attr3, c_attr4 = st.columns(4)
                c_attr1.metric("Delta", f"${attr_totals['Delta']:.2f}")
                c_attr2.metric("Gamma", f"${attr_totals['Gamma']:.2f}")
                c_attr3.metric("Vega", f"${attr_totals['Vega']:.2f}")
                c_attr4.metric("Theta", f"${attr_totals['Theta']:.2f}")
                
                st.metric("PnL Residual", f"${residual:.2f}", help="Total Variation - Sum of explained Greeks")
            


#TAB 5: volatility smile
with tabs[4]:
    
    analysis_date = None

    col_input, col_view = st.columns([1, 2])
    
    with col_input:

        st.subheader("Market Parameters")
        if st.session_state.get("data_mode") == "Preloaded Dataset":
            ticker = st.selectbox("Ticker", ["AAPL", "NVDA", "SPY"], key="smile_ticker").upper()

            from src.market_data import get_available_dates
            
            temp_op_type = st.session_state.get("smile_op_type", "C")
            available_dates = get_available_dates(ticker, temp_op_type)
            
            if available_dates:
                analysis_date = st.date_input(
                    "Analysis Date",
                    value=available_dates[-1],
                    min_value=available_dates[0],
                    max_value=available_dates[-1],
                    key="smile_analysis_date"
                )
            else:
                analysis_date = st.date_input("Analysis Date", disabled=True, key="smile_analysis_date_disabled")
                st.error(f"No concurrent data for {ticker} {temp_op_type} in preloaded CSVs.")

            c_exp1, c_exp2 = st.columns(2)
            #lcok expiration at 2026/02/20
            c_exp1.text_input("Expiration", value="2026/02/20", disabled=True, key="smile_exp_disp")
            exp_date = datetime(2026, 2, 20).date() 
            op_type = c_exp2.selectbox("Type", ["C", "P"], key="smile_op_type")
            
            #locked central strike based on ticker
            strike_map = {"NVDA": 170.0, "AAPL": 270.0, "SPY": 670.0}
            default_strike = strike_map.get(ticker, 100.0)
            st.number_input("Central Strike", value=default_strike, disabled=True, step=1.0, key=f"smile_strike_disp_{ticker}")
            strike = default_strike

            st.divider()
            if st.button("Update Data & Generate Smile", key="smile_full_update_preloaded"):
                if analysis_date:
                    ref_date_str = analysis_date.strftime("%Y-%m-%d")
                    with st.spinner(f"Updating data for {ref_date_str}..."):
                        
                        S_real, hv_real, err_s = get_stock_history_vol(ticker, ref_date_str)
                        df_opt_bar, err_o = get_option_aggregates(ticker, exp_date, op_type, strike, ref_date_str, ref_date_str)
                        
                        if not df_opt_bar.empty and S_real:
                            row_i = df_opt_bar.iloc[0]
                            st.session_state["underlying_S"] = S_real
                            st.session_state["underlying_HV"] = hv_real
                            st.session_state["market_mode"] = "Previous Day"
                                                      
                            time_to_exp = (pd.to_datetime(exp_date).date() - analysis_date).days / 365.0
                            st.session_state["center_data"] = {
                                "Strike": strike, "Date": row_i["Date"], "Open": row_i["Open"], "High": row_i["High"],
                                "Low": row_i["Low"], "Close": row_i["Close"], "Volume": row_i["Volume"],
                                "IV": row_i.get("Implied Volatility"), "IV_Error": None, "TimeToExp": time_to_exp
                            }

                            from src.market_data import load_preloaded_options
                            df_all = load_preloaded_options()
                            mask = (df_all['ticker'] == ticker) & \
                                   (df_all['type'] == ('Call' if op_type == 'C' else 'Put')) & \
                                   (df_all['Date'].dt.date == analysis_date)
                            df_day = df_all[mask].sort_values("Strike")
                            
                            if not df_day.empty:
                                smile_rows = []
                                for _, row_s in df_day.iterrows():
                                    smile_rows.append({
                                        "Strike": row_s["Strike"], 
                                        "IV": row_s.get("Implied Volatility"), 
                                        "Price": row_s["Close"], 
                                        "Moneyness": np.log(row_s["Strike"] / S_real)
                                    })
                                st.session_state["smile_data"] = pd.DataFrame(smile_rows)
                                st.success(f"Data and Smile updated for {ref_date_str}!")
                            else:
                                st.session_state.pop("smile_data", None)
                                st.warning("Metrics updated, but no strikes found for smile.")
                        else:
                            st.warning(f"No concurrent data found for {ticker} on {ref_date_str}")
                else:
                    st.error("Please select a valid analysis date.")

            if not available_dates:
                st.error(f"No concurrent data for {ticker} {temp_op_type} in preloaded CSVs.")


        else:
            ticker = st.text_input("Ticker", value="AAPL", key="smile_ticker").upper()
            
            c_exp1, c_exp2 = st.columns(2)
            exp_date = c_exp1.date_input("Expiration", value=datetime(2026, 2, 20).date(), key="smile_exp")
            op_type = c_exp2.selectbox("Type", ["C", "P"], key="smile_op_type")
            
            strike = st.number_input("Central Strike", value=275.0, step=1.0, key="smile_strike")

        st.divider()
        
        if st.session_state.get("data_mode") == "Live API":
            st.subheader("Data Fetching")
            if st.button("Fetch Data", key="smile_fetch", disabled=("user_api_key" not in st.session_state)):
                
                if "surface_data" in st.session_state: del st.session_state["surface_data"]
                if "custom_data_table" in st.session_state: del st.session_state["custom_data_table"]
                if "smile_data" in st.session_state: del st.session_state["smile_data"]
                
                with st.spinner("Fetching data from Massive..."):
                    center_df, err_msg, contract_symbol = get_option_previous_close(ticker, exp_date, op_type, strike)
    
                    if not center_df.empty:
                        ref_date_obj = center_df.iloc[0]["Date"]
                        ref_date_str = ref_date_obj.strftime("%Y-%m-%d")
                        row = center_df.iloc[0]
                        
                        st.session_state["center_data"] = {
                            "Strike": strike,
                            "Date": row["Date"],
                            "Open": row["Open"],
                            "High": row["High"],
                            "Low": row["Low"],
                            "Close": row["Close"],
                            "Volume": row["Volume"],
                            "IV": None,
                            "IV_Error": "Waiting for Underlying",
                            "TimeToExp": 0.0
                        }
                        st.session_state["market_mode"] = "Previous Day"

                        S_real, hv_real, err_s = get_stock_history_vol(ticker, ref_date_str)
                        
                        if S_real:
                            st.session_state["underlying_S"] = S_real
                            st.session_state["underlying_HV"] = hv_real
                            
                            #calc center IV or use precomputed
                            if "Implied Volatility" in row:
                                iv_c = row["Implied Volatility"]
                                iv_err_c = None
                            else:
                                time_to_exp = (pd.to_datetime(exp_date) - row["Date"]).days / 365.0
                                if time_to_exp < 0.001: time_to_exp = 0.001
                                iv_c, iv_err_c = implied_volatility(row["Close"], S_real, strike, time_to_exp, 0.05, op_type)
                                
                            st.session_state["center_data"]["IV"] = iv_c
                            st.session_state["center_data"]["IV_Error"] = iv_err_c
                            st.session_state["center_data"]["TimeToExp"] = (pd.to_datetime(exp_date) - row["Date"]).days / 365.0
                            
                            st.success("Option Data Loaded!")
                        else:
                            st.warning(f"Option Data Loaded, but Underlying failed: {err_s}")
                            st.session_state.pop("underlying_S", None)
                            st.session_state["center_data"]["IV_Error"] = f"Underlying Missing: {err_s}"
                    else:
                        st.error(f"Center Strike Option Data Error: {err_msg}")
    with col_input:
        if st.session_state.get("data_mode") == "Live API" and st.session_state.get("market_mode") == "Previous Day" and "center_data" in st.session_state:
            st.divider()
            if st.button("Generate Volatility Smile", key="smile_gen"):
                with st.spinner("Fetching neighbor strikes via API..."):
                    center = st.session_state["center_data"]
                    S_real = st.session_state["underlying_S"]
                    ticker = st.session_state.get("smile_ticker", "AAPL")
                    exp_date = st.session_state.get("smile_exp")
                    op_type = st.session_state.get("smile_op_type", "C")
                    
                    #relative strikes: +/- 15% of spot for DEEP OTM / ITM and +/- 5% of spot for SLIGHT OTM / ITM, rounded up to multiple of 5 for data availability purposes
                    rel_offsets = [0.85, 0.95, 1.05, 1.15]
                    strikes_to_fetch = sorted(list(set([int(np.ceil((S_real * r) / 5.0) * 5) for r in rel_offsets])))

                    if center["Strike"] in strikes_to_fetch:
                        strikes_to_fetch.remove(center["Strike"])
                    
                    smile_rows = []
                    
                    def calc_moneyness(S, K):
                        return np.log(K / S)

                    smile_rows.append({
                        "Strike": center["Strike"], 
                        "IV": center["IV"], 
                        "Price": center["Close"], 
                        "Moneyness": calc_moneyness(S_real, center["Strike"])
                    })
                    
                    for k_i in strikes_to_fetch:
                        df_i, err_i, _ = get_option_previous_close(ticker, exp_date, op_type, k_i)
                        
                        if not df_i.empty:
                            row_i = df_i.iloc[0]
                            iv_i, iv_err_i = implied_volatility(
                                row_i["Close"], S_real, k_i, center["TimeToExp"], 0.05, op_type
                            )
                            smile_rows.append({
                                "Strike": k_i, "IV": iv_i, "Price": row_i["Close"], "Moneyness": calc_moneyness(S_real, k_i)
                            })
                    
                    st.session_state["smile_data"] = pd.DataFrame(smile_rows).sort_values("Strike")
                    st.success("Smile Generated!")

    with col_view:
        if st.session_state.get("market_mode") == "Previous Day" and "center_data" in st.session_state:
            center = st.session_state["center_data"]
            
            #OHLCV data
            st.markdown("### 1. Option OHLCV Data")
            texp = center.get("TimeToExp", 0.0)
            date_str = center['Date'].strftime('%Y-%m-%d')
            st.caption(f"Date: {date_str} | Time to Exp: {texp:.4f}y")
            
            ohlc_cols = st.columns(5)
            ohlc_cols[0].metric("Open", f"${center['Open']:.2f}")
            ohlc_cols[1].metric("High", f"${center['High']:.2f}")
            ohlc_cols[2].metric("Low", f"${center['Low']:.2f}")
            ohlc_cols[3].metric("Close", f"${center['Close']:.2f}")
            ohlc_cols[4].metric("Volume", f"{center['Volume']:,}")
            
            #Underlying data
            st.markdown("### 2. Underlying Asset")
            if "underlying_S" in st.session_state:
                udata = st.columns(2)
                udata[0].metric("Underlying Price", f"${st.session_state.get('underlying_S', 0):.2f}")

                curr_hv = st.session_state.get('underlying_HV', 0)
                if st.session_state.get("data_mode") == "Preloaded Dataset":

                    t_curr = st.session_state.get("smile_ticker", "AAPL")
                    if f"constant_hv_{t_curr}" not in st.session_state:
                        from src.market_data import get_stock_history_vol
                        _, hv_const, _ = get_stock_history_vol(t_curr, "2025-12-17") # Dataset max date
                        st.session_state[f"constant_hv_{t_curr}"] = hv_const
                    
                    curr_hv = st.session_state.get(f"constant_hv_{t_curr}", curr_hv)

                udata[1].metric(
                    "Historical Volatility (1Y)", 
                    f"{curr_hv*100:.2f}%",
                    help="The value displayed here is the historical volatility of the underlying asset for the last year - It is hypothesized to be constant - This is a model simplification for ease of use"
                )
            else:
                 st.warning("Underlying Data Not Available")
            
            #IV part
            st.markdown("### 3. Implied Volatility")
            if center.get('IV_Error'):
                st.error(f"IV Calculation Failed: {center['IV_Error']}")
            else:
                st.metric("Implied Volatility", f"{center['IV']:.2%}" if center['IV'] else "N/A")

            if "smile_data" in st.session_state:
                df_smile = st.session_state["smile_data"]
                
                #filter invalid IVs
                df_smile = df_smile.replace([np.inf, -np.inf], np.nan).dropna(subset=["IV"])
                
                if not df_smile.empty:
                    st.subheader("Volatility Smile")
                    
                    use_moneyness = st.checkbox("change x axis to log moneyness", value=False)
                    
                    x_col = "Moneyness" if use_moneyness else "Strike"
                    x_label = "Log Moneyness ( log of (K/S) )" if use_moneyness else "Strike"
                    
                    x_vals = df_smile[x_col].values
                    y_vals = df_smile["IV"].values

                    S_spot = st.session_state.get('underlying_S', center["Strike"])
                    target_x = S_spot if not use_moneyness else 0.0

                    min_k = df_smile["Strike"].min()
                    max_k = df_smile["Strike"].max()
                    if S_spot < min_k or S_spot > max_k:
                         st.warning("Spot price not within smile bounds, data may be wrong")

                    iv_at_spot = interpolate_volatility(x_vals, y_vals, target_x)
                    
                    if iv_at_spot:
                        st.caption(f"**Interpolated IV at Spot:** {iv_at_spot:.2%}")

                    #curve smoothing
                    if len(x_vals) >= 4:
                         from scipy.interpolate import make_interp_spline
                         try:
                             X_Y_Spline = make_interp_spline(x_vals, y_vals)
                             X_smooth = np.linspace(x_vals.min(), x_vals.max(), 50)
                             Y_smooth = X_Y_Spline(X_smooth)
                         except:
                             X_smooth, Y_smooth = x_vals, y_vals
                    else:
                        X_smooth, Y_smooth = x_vals, y_vals
                        
                    fig = go.Figure()
                    fig.add_trace(go.Scatter(x=X_smooth, y=Y_smooth, mode='lines', name='Smile', line=dict(shape='spline', color='#00CC96')))
                    fig.add_trace(go.Scatter(x=x_vals, y=y_vals, mode='markers', name='Data', marker=dict(size=8, color='#EF553B')))
                    
                    #spot vertical line for visual purposes
                    fig.add_vline(
                        x=target_x, 
                        line_width=2, 
                        line_dash="dot", 
                        line_color="red", 
                        annotation_text="Underlying", 
                        annotation_position="top right"
                    )
                    
                    fig.update_layout(title=f"IV vs {x_label}", xaxis_title=x_label, yaxis_title="Implied Volatility", template="plotly_dark")
                    st.plotly_chart(fig, width="stretch")
            else:
                if st.session_state.get("data_mode") == "Live API":
                    st.info("Click 'Generate Volatility Smile' to fetch neighbor strikes.")
        else:
             if st.session_state.get("data_mode") == "Live API":
                st.info("Select parameters and click Fetch Data.")

#TAB 6: volatility surface
with tabs[5]:
    
    col_s_input, col_s_view = st.columns([1, 2])
    
    with col_s_input:
        
        st.subheader("Market Parameters")
        if st.session_state.get("data_mode") == "Preloaded Dataset":
            ticker = st.selectbox("Ticker", ["AAPL", "NVDA", "SPY"], key="surf_ticker").upper()

            c_d1, c_d2 = st.columns(2)
            end_date_default = datetime(2025, 12, 17).date()
            start_date_default = end_date_default - timedelta(days=15)
            
            surf_end_date = c_d2.date_input("End Date", value=end_date_default, min_value=datetime(2024, 12, 18).date(), max_value=datetime(2025, 12, 17).date(), key="surf_analysis_end")
            surf_start_date = c_d1.date_input("Start Date", value=start_date_default, min_value=datetime(2024, 12, 18).date(), max_value=surf_end_date - timedelta(days=1), key="surf_analysis_start")

            c_exp1, c_exp2 = st.columns(2)
            #exp date locked at 2026/02/20 to match preloaded dataset
            c_exp1.text_input("Expiration", value="2026/02/20", disabled=True, key="surf_exp_disp")
            exp_date = datetime(2026, 2, 20).date()
            op_type = c_exp2.selectbox("Type", ["C", "P"], key="surf_op_type")
            
            #strike locked based on ticker
            strike_map = {"NVDA": 170.0, "AAPL": 270.0, "SPY": 670.0}
            default_strike = strike_map.get(ticker, 100.0)
            st.number_input("Central Strike", value=default_strike, disabled=True, step=1.0, key=f"surf_strike_disp_{ticker}")
            strike = default_strike
        else:
            ticker = st.text_input("Ticker", value="AAPL", key="surf_ticker").upper()
            
            c_exp1, c_exp2 = st.columns(2)
            exp_date = c_exp1.date_input("Expiration", value=datetime(2026, 2, 20).date(), key="surf_exp")
            op_type = c_exp2.selectbox("Type", ["C", "P"], key="surf_op_type")
            
            strike = st.number_input("Central Strike", value=275.0, step=1.0, key="surf_strike")
        
        if st.session_state.get("data_mode") == "Live API":
            st.divider()
            st.subheader("Data Fetching")
            
            c_d1, c_d2 = st.columns(2)
            today = datetime.now().date()
            max_end_date = today - timedelta(days=1)
            
            end_date = c_d2.date_input("End Date", value=max_end_date, max_value=max_end_date, key="surf_end_date")
            start_date = c_d1.date_input("Start Date", value=end_date - timedelta(days=10), max_value=end_date - timedelta(days=1), key="surf_start_date")
            
            if start_date >= end_date:
                st.error("Start Date must be before End Date.")
            
            is_key_missing = "user_api_key" not in st.session_state
            if st.button("Fetch Data", key="surf_fetch", disabled=(start_date >= end_date or is_key_missing)):

                if "center_data" in st.session_state: del st.session_state["center_data"]
                if "smile_data" in st.session_state: del st.session_state["smile_data"]
                if "surface_data" in st.session_state: del st.session_state["surface_data"]
                
                with st.spinner("Fetching data from Massive (Multi-Strike)..."):

                    u_data, u_err = get_underlying_history_range(
                        ticker, 
                        start_date.strftime("%Y-%m-%d"), 
                        end_date.strftime("%Y-%m-%d")
                    )
                    
                    if u_data:
                        latest_date = max(u_data.keys())
                        S_spot = u_data[latest_date]
                        
                        #fetch neighbors with the same logic as smile
                        rel_offsets = [0.85, 0.95, 1.05, 1.15]
                        neighbors = [int(np.ceil((S_spot * r) / 5.0) * 5) for r in rel_offsets]
                        strikes_to_fetch = sorted(list(set([strike] + neighbors)))
                        
                        all_surface_data = []
                        
                        progress_bar = st.progress(0, text="Fetching Strikes...")
                        
                        for idx, k_curr in enumerate(strikes_to_fetch):
                             progress_bar.progress((idx + 1) / len(strikes_to_fetch), text=f"Fetching Strike {k_curr}...")
                             
                             df_data, error_msg = get_option_aggregates(
                                ticker, 
                                exp_date, 
                                op_type, 
                                k_curr, 
                                start_date.strftime("%Y-%m-%d"), 
                                end_date.strftime("%Y-%m-%d"),
                                limit=200 # increased limit for longer ranges
                            )
                             
                             if not df_data.empty:
                                 for _, row in df_data.iterrows():
                                     d_date = row["Date"].date()
                                     if d_date in u_data:
                                         S_t = u_data[d_date]
                                         
                                         if "Implied Volatility" in row:
                                             iv_t = row["Implied Volatility"]
                                         else:
                                             T_t = (pd.to_datetime(exp_date).date() - d_date).days / 365.0
                                             if T_t < 0.001: T_t = 0.001
                                             iv_t, _ = implied_volatility(row["Close"], S_t, k_curr, T_t, 0.05, op_type)
                                         
                                         if iv_t is not None:
                                             all_surface_data.append({
                                                 "Date": row["Date"],
                                                 "Underlying": S_t,
                                                 "OptionPrice": row["Close"],
                                                 "Strike": k_curr,
                                                 "Moneyness": np.log(k_curr / S_t), 
                                                 "IV": iv_t
                                             })
                        
                        progress_bar.empty()
                        
                        if all_surface_data:
                            st.session_state["custom_data_table"] = pd.DataFrame(all_surface_data) # Stores full dataset
                            st.session_state["market_mode"] = "Custom Timeframe"
                            st.success(f"Loaded {len(all_surface_data)} data points across {len(strikes_to_fetch)} strikes!")
                        else:
                            st.warning("No valid overlapping data found for these strikes.")
                    else:
                        st.error(f"Underlying fetch error: {u_err}")
        else:
            
            st.divider()
            if st.button("Update Data & Generate Surface", key="surf_gen_preloaded"):
                if "center_data" in st.session_state: del st.session_state["center_data"]
                if "smile_data" in st.session_state: del st.session_state["smile_data"]
                if "surface_data" in st.session_state: del st.session_state["surface_data"]
                if "custom_data_table" in st.session_state: del st.session_state["custom_data_table"]

                with st.spinner("Processing offline dataset..."):
                    u_data, u_err = get_underlying_history_range(
                        ticker, 
                        surf_start_date.strftime("%Y-%m-%d"), 
                        surf_end_date.strftime("%Y-%m-%d")
                    )
                    
                    if u_data:
                        df_all_opt, err_msg = get_all_preloaded_options(
                            ticker, exp_date, op_type, 
                            surf_start_date.strftime("%Y-%m-%d"), 
                            surf_end_date.strftime("%Y-%m-%d")
                        )
                        
                        if not df_all_opt.empty:
                            all_surface_data = []
                            for _, row in df_all_opt.iterrows():
                                d_date = row["Date"].date()
                                if d_date in u_data:
                                    S_t = u_data[d_date]
                                    k_curr = row["Strike"]
                                    
                                    if "Implied Volatility" in row:
                                        iv_t = row["Implied Volatility"]
                                    else:
                                        T_t = (pd.to_datetime(exp_date).date() - d_date).days / 365.0
                                        if T_t < 0.001: T_t = 0.001
                                        iv_t, _ = implied_volatility(row["Close"], S_t, k_curr, T_t, 0.05, op_type)
                                    
                                    if iv_t is not None:
                                        all_surface_data.append({
                                            "Date": row["Date"], "Underlying": S_t, "OptionPrice": row["Close"],
                                            "Strike": k_curr, "Moneyness": np.log(k_curr / S_t), "IV": iv_t
                                        })
                        
                        if all_surface_data:
                            st.session_state["custom_data_table"] = pd.DataFrame(all_surface_data)
                            st.session_state["surface_data"] = st.session_state["custom_data_table"]
                            st.session_state["market_mode"] = "Custom Timeframe"
                            st.success(f"Surface generated with {len(all_surface_data)} points!")
                        else:
                            st.error("No overlap data found in range.")
                    else:
                        st.error(f"Underlying range error: {u_err}")

    with col_s_input:
        #generate the surface
        if st.session_state.get("data_mode") == "Live API" and st.session_state.get("market_mode") == "Custom Timeframe" and "custom_data_table" in st.session_state:
             st.divider()
             if st.button("Generate Volatility Surface", key="surf_gen"):
                 st.session_state["surface_data"] = st.session_state["custom_data_table"]
                 
    with col_s_view:
        if st.session_state.get("market_mode") == "Custom Timeframe":
            
            if "custom_data_table" in st.session_state:
                df_all = st.session_state["custom_data_table"]
                
                center_val = strike
                df_table = df_all[df_all["Strike"] == center_val].sort_values("Date")
                
                if df_table.empty:
                    st.info(f"No data for central strike {center_val}, showing all.")
                    df_table = df_all.head(50)
                
                d_min = df_all["Date"].min().strftime("%Y-%m-%d")
                d_max = df_all["Date"].max().strftime("%Y-%m-%d")
                
                st.subheader(f"Historical Data ({d_min} to {d_max})")
                st.caption(f"Showing data for Center Strike: {center_val}")
                st.dataframe(
                    df_table[["Date", "Underlying", "OptionPrice", "Moneyness", "IV"]].style.format({
                        "Underlying": "{:.2f}",
                        "OptionPrice": "{:.2f}",
                        "Moneyness": "{:.4f}",
                        "IV": "{:.2%}"
                    }), 
                    width="stretch",
                    hide_index=True
                )
            
            if "surface_data" in st.session_state:
                df_surf = st.session_state["surface_data"]
                
                st.divider()
                st.subheader("Implied Volatility Surface")
                st.info(f"Surface generated using {len(df_surf)} data points.")
                
                try:
                    df_clean = df_surf.copy()
                    df_clean['Date'] = pd.to_datetime(df_clean['Date'])
                    df_clean['TimeToExp'] = (pd.to_datetime(exp_date) - df_clean['Date']).dt.days / 365.0
                    
                    #sanity check
                    df_clean = df_clean[(df_clean['IV'] > 0) & (df_clean['TimeToExp'] > 0)].copy()
                    
                    # Filter for validity and drop NaNs
                    df_clean = df_clean.replace([np.inf, -np.inf], np.nan).dropna(subset=["IV", "Moneyness", "TimeToExp"])
                    
                    if len(df_clean) < 4:
                        st.warning("Insufficient data points to form a surface. Need at least 4 valid points.")
                    else:
                        from scipy.interpolate import griddata
                        
                        min_mon, max_mon = df_clean["Moneyness"].min(), df_clean["Moneyness"].max()
                        min_tte, max_tte = df_clean["TimeToExp"].min(), df_clean["TimeToExp"].max()
                        
                        if max_mon == min_mon or max_tte == min_tte:
                            st.info("Forming a 3D surface requires a spread across both Strikes and Time. Plotting as 2D fallback.")

                            fig = go.Figure(data=[go.Scatter(
                                x=df_clean["Moneyness"], y=df_clean["IV"], mode='markers', 
                                marker=dict(size=8, color=df_clean["TimeToExp"], colorscale='Viridis', showscale=True)
                            )])
                            fig.update_layout(title="IV vs Moneyness (Single Slice Fallback)", xaxis_title="Log Moneyness", yaxis_title="IV", template="plotly_dark")
                            st.plotly_chart(fig, width="stretch")
                        else:
                            grid_x_1d = np.linspace(min_mon, max_mon, 30)
                            grid_y_1d = np.linspace(min_tte, max_tte, 30)
                            grid_x, grid_y = np.meshgrid(grid_x_1d, grid_y_1d)
                            
                            points = df_clean[["Moneyness", "TimeToExp"]].values
                            values = df_clean["IV"].values
                            
                            grid_z = griddata(points, values, (grid_x, grid_y), method='linear')
                            
                            grid_z_nearest = griddata(points, values, (grid_x, grid_y), method='nearest')
                            if np.isnan(grid_z).any():
                                if np.isnan(grid_z).all():
                                    grid_z = grid_z_nearest
                                else:
                                    mask = np.isnan(grid_z)
                                    grid_z[mask] = grid_z_nearest[mask]
                            
                            if np.isnan(grid_z).all():
                                 st.error("Surface interpolation failed (All NaNs). Likely collinear data.")
                            else:
                                #plot the surface
                                fig = go.Figure(data=[go.Surface(
                                    z=grid_z,
                                    x=grid_x_1d,
                                    y=grid_y_1d,
                                    colorscale='Viridis',
                                    opacity=0.9,
                                    contours_z=dict(show=True, usecolormap=True, highlightcolor="limegreen", project_z=True)
                                )])
                                
                                fig.update_layout(
                                    title=f"IV Surface (Moneyness vs TTE)",
                                    scene=dict(
                                        xaxis=dict(title='Log Moneyness ( log(K/S) )'),
                                        yaxis=dict(title='Time to Expiration (Years)'),
                                        zaxis=dict(title='Implied Volatility'),
                                        camera=dict(eye=dict(x=1.5, y=-1.5, z=0.5)),
                                        aspectratio=dict(x=1, y=1, z=1)
                                    ),
                                    template="plotly_dark",
                                    margin=dict(l=0, r=0, b=0, t=40),
                                    height=700
                                )
                                st.plotly_chart(fig, width="stretch")

                except Exception as e:
                    st.error(f"Error creating surface plot: {e}")
                    import traceback
                    st.text(traceback.format_exc())
                    st.write("Raw Data Summary:", df_surf.describe())
        else:
             st.info("Select parameters and click Fetch Data.")
