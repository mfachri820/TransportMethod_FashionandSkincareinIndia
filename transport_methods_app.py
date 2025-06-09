import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

st.set_page_config(
    page_title="Transport Method Solver",
    layout="wide",
    initial_sidebar_state="expanded"
)

class TransportMethodSolver:
    def __init__(self):
        self.cost_matrix = None
        self.supply = None
        self.demand = None
        self.allocation = None
        self.total_cost = 0
        self.suppliers = []
        self.destinations = []
        
    def load_data(self):
        try:
            files_to_check = [
                'data/matriks_biaya_final.csv',
                'data/vektor_pasokan_final.csv', 
                'data/vektor_permintaan_final.csv'
            ]
            
            cost_df = pd.read_csv('data/matriks_biaya_final.csv', index_col=0)
            self.cost_matrix = cost_df.values.astype(float)
            
            supply_df = pd.read_csv('data/vektor_pasokan_final.csv', index_col=0)
            self.supply = supply_df['Supply'].values.astype(float)
            
            demand_df = pd.read_csv('data/vektor_permintaan_final.csv', index_col=0)
            self.demand = demand_df['Demand'].values.astype(float)
            
            self.suppliers = cost_df.index.tolist()
            self.destinations = cost_df.columns.tolist()
            
            if len(self.suppliers) != len(self.supply):
                raise ValueError("Mismatch between suppliers and supply vector dimensions")
            if len(self.destinations) != len(self.demand):
                raise ValueError("Mismatch between destinations and demand vector dimensions")
            if self.cost_matrix.shape != (len(self.suppliers), len(self.destinations)):
                raise ValueError("Cost matrix dimensions don't match suppliers/destinations")
            
            return True
            
        except Exception as e:
            st.error(f"Error loading data: {e}")
            return False
    
    def northwest_corner_method(self):
        m, n = len(self.supply), len(self.demand)
        supply_copy = self.supply.copy()
        demand_copy = self.demand.copy()
        allocation = np.zeros((m, n))
        
        steps = []
        i, j = 0, 0
        
        while i < m and j < n:
            allocated = min(supply_copy[i], demand_copy[j])
            allocation[i, j] = allocated
            
            steps.append({
                'Step': len(steps) + 1,
                'Supplier': self.suppliers[i],
                'Destination': self.destinations[j],
                'Allocation': allocated,
                'Unit Cost': self.cost_matrix[i, j],
                'Cost': allocated * self.cost_matrix[i, j],
                'Remaining Supply': supply_copy[i] - allocated,
                'Remaining Demand': demand_copy[j] - allocated
            })
            
            supply_copy[i] -= allocated
            demand_copy[j] -= allocated
            
            if supply_copy[i] == 0:
                i += 1
            if demand_copy[j] == 0:
                j += 1
        
        self.allocation = allocation
        self.total_cost = np.sum(allocation * self.cost_matrix)
        return steps
    
    def least_cost_method(self):
        m, n = len(self.supply), len(self.demand)
        supply_copy = self.supply.copy()
        demand_copy = self.demand.copy()
        allocation = np.zeros((m, n))
        
        steps = []
        cost_matrix_copy = self.cost_matrix.copy()
        
        while np.sum(supply_copy) > 0 and np.sum(demand_copy) > 0:
            min_cost = float('inf')
            min_i, min_j = -1, -1
            
            for i in range(m):
                for j in range(n):
                    if supply_copy[i] > 0 and demand_copy[j] > 0 and cost_matrix_copy[i, j] < min_cost:
                        min_cost = cost_matrix_copy[i, j]
                        min_i, min_j = i, j
            
            if min_i == -1:
                break
                
            allocated = min(supply_copy[min_i], demand_copy[min_j])
            allocation[min_i, min_j] = allocated
            
            steps.append({
                'Step': len(steps) + 1,
                'Supplier': self.suppliers[min_i],
                'Destination': self.destinations[min_j],
                'Unit Cost': min_cost,
                'Allocation': allocated,
                'Cost': allocated * min_cost,
                'Remaining Supply': supply_copy[min_i] - allocated,
                'Remaining Demand': demand_copy[min_j] - allocated
            })
            
            supply_copy[min_i] -= allocated
            demand_copy[min_j] -= allocated
            
            if supply_copy[min_i] == 0:
                cost_matrix_copy[min_i, :] = float('inf')
            if demand_copy[min_j] == 0:
                cost_matrix_copy[:, min_j] = float('inf')
        
        self.allocation = allocation
        self.total_cost = np.sum(allocation * self.cost_matrix)
        return steps
    
    def vogels_approximation_method(self):
        m, n = len(self.supply), len(self.demand)
        supply_copy = self.supply.copy()
        demand_copy = self.demand.copy()
        allocation = np.zeros((m, n))
        
        steps = []
        cost_matrix_copy = self.cost_matrix.copy()
        
        while np.sum(supply_copy) > 0 and np.sum(demand_copy) > 0:  
            row_penalties = []
            col_penalties = []
            
            for i in range(m):
                if supply_copy[i] > 0:
                    row_costs = [cost_matrix_copy[i, j] for j in range(n) if demand_copy[j] > 0]
                    if len(row_costs) >= 2:
                        row_costs.sort()
                        penalty = row_costs[1] - row_costs[0]
                    else:
                        penalty = 0
                else:
                    penalty = -1
                row_penalties.append(penalty)
            
            for j in range(n):
                if demand_copy[j] > 0:
                    col_costs = [cost_matrix_copy[i, j] for i in range(m) if supply_copy[i] > 0]
                    if len(col_costs) >= 2:
                        col_costs.sort()
                        penalty = col_costs[1] - col_costs[0]
                    else:
                        penalty = 0
                else:
                    penalty = -1
                col_penalties.append(penalty)
            
            max_row_penalty = max([p for p in row_penalties if p >= 0]) if any(p >= 0 for p in row_penalties) else -1
            max_col_penalty = max([p for p in col_penalties if p >= 0]) if any(p >= 0 for p in col_penalties) else -1
            
            if max_row_penalty >= max_col_penalty:
                selected_row = row_penalties.index(max_row_penalty)
                min_cost = float('inf')
                selected_col = -1
                for j in range(n):
                    if demand_copy[j] > 0 and cost_matrix_copy[selected_row, j] < min_cost:
                        min_cost = cost_matrix_copy[selected_row, j]
                        selected_col = j
            else:
                selected_col = col_penalties.index(max_col_penalty)
                min_cost = float('inf')
                selected_row = -1
                for i in range(m):
                    if supply_copy[i] > 0 and cost_matrix_copy[i, selected_col] < min_cost:
                        min_cost = cost_matrix_copy[i, selected_col]
                        selected_row = i
            
            if selected_row == -1 or selected_col == -1:
                break
            
            allocated = min(supply_copy[selected_row], demand_copy[selected_col])
            allocation[selected_row, selected_col] = allocated
            
            steps.append({
                'Step': len(steps) + 1,
                'Supplier': self.suppliers[selected_row],
                'Destination': self.destinations[selected_col],
                'Unit Cost': min_cost,
                'Allocation': allocated,
                'Cost': allocated * min_cost,
                'Row Penalty': max_row_penalty,
                'Col Penalty': max_col_penalty,
                'Remaining Supply': supply_copy[selected_row] - allocated,
                'Remaining Demand': demand_copy[selected_col] - allocated
            })
            
            supply_copy[selected_row] -= allocated
            demand_copy[selected_col] -= allocated
            
            if supply_copy[selected_row] == 0:
                cost_matrix_copy[selected_row, :] = float('inf')
            if demand_copy[selected_col] == 0:
                cost_matrix_copy[:, selected_col] = float('inf')
        
        self.allocation = allocation
        self.total_cost = np.sum(allocation * self.cost_matrix)
        return steps

def create_allocation_heatmap(allocation, suppliers, destinations, title):
    try:
        fig = go.Figure(data=go.Heatmap(
            z=allocation,
            x=destinations,
            y=suppliers,
            colorscale='RdYlBu_r',
            text=allocation.astype(int),
            texttemplate="%{text}",
            textfont={"size": 10, "color": "black"},
            hoverongaps=False,
            showscale=True,
            colorbar=dict(
                title="Allocation Units",
                title_side="right"
            )
        ))
        
        fig.update_layout(
            title={
                'text': title,
                'x': 0.5,
                'xanchor': 'center',
                'font': {'size': 14, 'color': 'darkblue'}
            },
            xaxis_title="Destinations",
            yaxis_title="Suppliers",
            height=400,
            font=dict(size=10),
            plot_bgcolor='white'
        )
        
        return fig
    except Exception as e:
        st.error(f"Error creating heatmap: {e}")
        fig = go.Figure()
        fig.add_annotation(
            text=f"Chart Error: {str(e)}",
            xref="paper", yref="paper",
            x=0.5, y=0.5, xanchor='center', yanchor='middle',
            showarrow=False,
            font=dict(size=16, color="red")
        )
        return fig

def create_cost_comparison_chart(results):
    try:
        methods = list(results.keys())
        costs = [results[method]['total_cost'] for method in methods]
        
        colors = ['#2E86AB', '#A23B72', '#F18F01']
        
        fig = go.Figure(data=[
            go.Bar(
                x=methods, 
                y=costs, 
                text=[f"₹{cost:,.0f}" for cost in costs],
                textposition='auto',
                marker_color=colors[:len(methods)],
                textfont=dict(size=12, color='white')
            )
        ])
        
        fig.update_layout(
            title={
                'text': "Total Transportation Cost Comparison",
                'x': 0.5,
                'xanchor': 'center',
                'font': {'size': 16, 'color': 'darkblue'}
            },
            xaxis_title="Transportation Methods",
            yaxis_title="Total Cost",
            height=400,
            font=dict(size=12),
            plot_bgcolor='white',
            showlegend=False
        )
        
        return fig
    except Exception as e:
        st.error(f"Error creating comparison chart: {e}")
        return go.Figure()

def create_allocation_summary_chart(allocation, suppliers, destinations, costs):
    try:
        fig = make_subplots(
            rows=1, cols=2,
            subplot_titles=('Supply Utilization', 'Demand Fulfillment'),
            specs=[[{"type": "bar"}, {"type": "bar"}]]
        )
        
        supply_used = np.sum(allocation, axis=1)
        fig.add_trace(
            go.Bar(x=suppliers, y=supply_used, name="Supply Used", marker_color='lightblue'),
            row=1, col=1
        )
        
        demand_met = np.sum(allocation, axis=0)
        fig.add_trace(
            go.Bar(x=destinations, y=demand_met, name="Demand Met", marker_color='lightcoral'),
            row=1, col=2
        )
        
        fig.update_layout(height=400, showlegend=False)
        return fig
    except Exception as e:
        st.error(f"Error creating summary chart: {e}")
        return go.Figure()

def main():
    st.title("Transportation Method Solver")
    solver = TransportMethodSolver()
    
    if not solver.load_data():
        st.stop()
    
    with st.sidebar:
        st.header("Data Overview")
        
        try:
            total_supply = np.sum(solver.supply)
            total_demand = np.sum(solver.demand)
            
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Total Supply", f"{total_supply:,.0f}")
            with col2:
                st.metric("Total Demand", f"{total_demand:,.0f}")
            
            if total_supply == total_demand:
                st.success("Balanced Problem")
            else:
                st.warning("Unbalanced Problem")
                
            st.subheader("Cost Matrix")
            cost_df = pd.DataFrame(solver.cost_matrix, 
                                  index=solver.suppliers, 
                                  columns=solver.destinations)
            st.dataframe(cost_df.round(2), use_container_width=True)
            
            col1, col2 = st.columns(2)
            with col1:
                st.subheader("Supply")
                supply_df = pd.DataFrame({
                    'Supplier': solver.suppliers,
                    'Supply': solver.supply.astype(int)
                })
                st.dataframe(supply_df, use_container_width=True, hide_index=True)
            
            with col2:
                st.subheader("Demand") 
                demand_df = pd.DataFrame({
                    'Destination': solver.destinations,
                    'Demand': solver.demand.astype(int)
                })
                st.dataframe(demand_df, use_container_width=True, hide_index=True)
        except Exception as e:
            st.error(f"Error displaying data overview: {e}")
    
    st.header("Select Transportation Method")
    
    method_options = {
        "Northwest Corner Method": "ncm",
        "Least Cost Method": "lcm", 
        "Vogel's Approximation Method": "vam"
    }
    
    method_descriptions = {
        "Northwest Corner Method": "Simple method starting from top-left corner",
        "Least Cost Method": "Greedy approach selecting minimum cost cells first",
        "Vogel's Approximation Method": "Advanced method using penalty calculations"
    }
    
    selected_methods = st.multiselect(
        "Choose methods to compare:",
        options=list(method_options.keys()),
        default=list(method_options.keys()),
        help="Select one or more methods to solve and compare"
    )
    
    if selected_methods:
        st.write("**Selected Methods:**")
        for method in selected_methods:
            st.write(f"• **{method}**: {method_descriptions[method]}")
    
    if st.button("Solve Transportation Problem", type="primary", use_container_width=True):
        if not selected_methods:
            st.warning("Please select at least one method.")
            st.stop()
            
        results = {}
        
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        try:
            for i, method_name in enumerate(selected_methods):
                method_code = method_options[method_name]
                
                status_text.text(f"Solving using {method_name}...")
                progress_bar.progress((i + 0.5) / len(selected_methods))
                
                try:
                    if method_code == "ncm":
                        steps = solver.northwest_corner_method()
                    elif method_code == "lcm":
                        steps = solver.least_cost_method()
                    elif method_code == "vam":
                        steps = solver.vogels_approximation_method()
                    
                    results[method_name] = {
                        'allocation': solver.allocation.copy(),
                        'total_cost': solver.total_cost,
                        'steps': steps
                    }
                except Exception as e:
                    st.error(f"Error solving with {method_name}: {e}")
                    continue
                
                progress_bar.progress((i + 1) / len(selected_methods))
            
            status_text.text("All methods completed!")
            progress_bar.empty()
            status_text.empty()
        except Exception as e:
            st.error(f"Error during solving process: {e}")
            progress_bar.empty()
            status_text.empty()
            st.stop()
        
        if not results:
            st.error("No methods completed successfully. Please check your data.")
            st.stop()
        
        for method_name in results.keys():
            st.subheader(f"{method_name} Results")
            
            try:
                allocation = results[method_name]['allocation']
                total_cost = results[method_name]['total_cost']
                steps = results[method_name]['steps']
                
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Total Cost", f"{total_cost:,.2f}")
                with col2:
                    st.metric("Solution Steps", len(steps))
                with col3:
                    non_zero_allocations = np.count_nonzero(allocation)
                    st.metric("Active Routes", non_zero_allocations)
                
                tab1, tab2, tab3 = st.tabs(["Allocation Matrix", "Visualization", "Solution Steps"])
                
                with tab1:
                    try:
                        allocation_df = pd.DataFrame(
                            allocation.astype(int), 
                            index=solver.suppliers, 
                            columns=solver.destinations
                        )
                        st.dataframe(allocation_df, use_container_width=True)
                    except Exception as e:
                        st.error(f"Error displaying allocation matrix: {e}")
                
                with tab2:
                    fig_heatmap = create_allocation_heatmap(
                        allocation, 
                        solver.suppliers, 
                        solver.destinations,
                        f"{method_name} - Allocation Heatmap"
                    )
                    st.plotly_chart(fig_heatmap, use_container_width=True, key=f"heatmap_{method_name}")
                    
                    fig_summary = create_allocation_summary_chart(
                        allocation, solver.suppliers, solver.destinations, solver.cost_matrix
                    )
                    st.plotly_chart(fig_summary, use_container_width=True, key=f"summary_{method_name}")
                
                with tab3:
                    try:
                        steps_df = pd.DataFrame(steps)
                        st.dataframe(steps_df, use_container_width=True, hide_index=True)
                    except Exception as e:
                        st.error(f"Error displaying solution steps: {e}")
                
            except Exception as e:
                st.error(f"Error displaying results for {method_name}: {e}")
            
            st.markdown("---")
        
        if len(results) > 1:
            st.header("Methods Comparison")
            
            try:
                fig_comparison = create_cost_comparison_chart(results)
                st.plotly_chart(fig_comparison, use_container_width=True, key="cost_comparison")
                
                best_method = min(results.keys(), key=lambda x: results[x]['total_cost'])
                best_cost = results[best_method]['total_cost']
                worst_method = max(results.keys(), key=lambda x: results[x]['total_cost'])
                worst_cost = results[worst_method]['total_cost']
                
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.success(f"**Best**: {best_method}")
                    st.write(f"Cost: {best_cost:,.2f}")
                with col2:
                    if len(results) > 2:
                        savings = worst_cost - best_cost
                        savings_pct = (savings / worst_cost) * 100 if worst_cost > 0 else 0
                        st.info(f"**Savings**: {savings:,.2f}")
                        st.write(f"({savings_pct:.1f}% improvement)")
                with col3:
                    st.error(f"**Highest**: {worst_method}")
                    st.write(f"Cost: {worst_cost:,.2f}")
                
                st.subheader("Detailed Comparison")
                comparison_data = []
                for method, result in results.items():
                    comparison_data.append({
                        'Method': method,
                        'Total Cost': f"{result['total_cost']:,.2f}",
                        'Solution Steps': len(result['steps']),
                        'Active Routes': np.count_nonzero(result['allocation']),
                        'Cost Difference': f"{result['total_cost'] - best_cost:,.2f}",
                        'Status': 'Best' if method == best_method else ('Highest' if method == worst_method else 'Good')
                    })
                
                comparison_df = pd.DataFrame(comparison_data)
                st.dataframe(comparison_df, use_container_width=True, hide_index=True)
            except Exception as e:
                st.error(f"Error in comparison section: {e}")

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        st.error(f"Application error: {e}")