import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

st.set_page_config(
    page_title="Transport Method Solver with Stepping Stone",
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
        self.stepping_stone_iterations = []
        
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

    def get_basic_variables(self, allocation):
        """Get positions of basic variables (non-zero allocations)"""
        basic_vars = []
        m, n = allocation.shape
        for i in range(m):
            for j in range(n):
                if allocation[i, j] > 0:
                    basic_vars.append((i, j))
        return basic_vars

    def find_loop(self, basic_vars, start_pos):
        """Find a closed loop for stepping stone method"""
        def can_form_rectangle(pos1, pos2, pos3, pos4):
            return ((pos1[0] == pos2[0] and pos3[0] == pos4[0] and pos1[1] == pos4[1] and pos2[1] == pos3[1]) or
                   (pos1[1] == pos2[1] and pos3[1] == pos4[1] and pos1[0] == pos4[0] and pos2[0] == pos3[0]))

        # Try to find a simple rectangular loop
        for i, pos1 in enumerate(basic_vars):
            for j, pos2 in enumerate(basic_vars[i+1:], i+1):
                for k, pos3 in enumerate(basic_vars[j+1:], j+1):
                    # Check if we can form a rectangle with start_pos
                    if can_form_rectangle(start_pos, pos1, pos2, pos3):
                        return [start_pos, pos1, pos2, pos3]
        
        # Fallback: return a simple loop if rectangle not found
        if len(basic_vars) >= 3:
            return [start_pos] + basic_vars[:3]
        return None

    def calculate_opportunity_cost(self, allocation, i, j):
        """Calculate opportunity cost for non-basic variable (i,j)"""
        basic_vars = self.get_basic_variables(allocation)
        
        # Find a loop including position (i,j)
        loop = self.find_loop(basic_vars, (i, j))
        
        if not loop:
            return 0  # Can't form loop, assume no improvement
        
        # Calculate opportunity cost along the loop
        opportunity_cost = self.cost_matrix[i, j]
        
        # Alternate signs: -, +, -, +, ...
        for idx, (r, c) in enumerate(loop[1:], 1):
            if idx % 2 == 1:  # Odd positions: subtract
                opportunity_cost -= self.cost_matrix[r, c]
            else:  # Even positions: add
                opportunity_cost += self.cost_matrix[r, c]
        
        return opportunity_cost

    def stepping_stone_optimization(self, initial_allocation, max_iterations=10):
        """Optimize allocation using Stepping Stone method"""
        self.stepping_stone_iterations = []
        allocation = initial_allocation.copy()
        current_cost = np.sum(allocation * self.cost_matrix)
        
        for iteration in range(max_iterations):
            m, n = allocation.shape
            basic_vars = self.get_basic_variables(allocation)
            
            # Calculate opportunity costs for all non-basic variables
            opportunity_costs = np.zeros((m, n))
            min_opportunity_cost = 0
            entering_var = None
            
            for i in range(m):
                for j in range(n):
                    if allocation[i, j] == 0:  # Non-basic variable
                        opp_cost = self.calculate_opportunity_cost(allocation, i, j)
                        opportunity_costs[i, j] = opp_cost
                        
                        if opp_cost < min_opportunity_cost:
                            min_opportunity_cost = opp_cost
                            entering_var = (i, j)
            
            # Store iteration details
            iteration_info = {
                'iteration': iteration + 1,
                'current_cost': current_cost,
                'opportunity_costs': opportunity_costs.copy(),
                'min_opportunity_cost': min_opportunity_cost,
                'entering_variable': entering_var,
                'allocation': allocation.copy()
            }
            
            # If all opportunity costs are non-negative, optimal solution found
            if min_opportunity_cost >= 0:
                iteration_info['status'] = 'Optimal'
                iteration_info['improvement'] = 0
                self.stepping_stone_iterations.append(iteration_info)
                break
            
            # Find loop for entering variable and perform allocation shift
            if entering_var:
                basic_vars = self.get_basic_variables(allocation)
                loop = self.find_loop(basic_vars, entering_var)
                
                if loop and len(loop) >= 4:
                    # Find minimum allocation in positions to be decreased
                    min_allocation = float('inf')
                    for idx, (r, c) in enumerate(loop[1:], 1):
                        if idx % 2 == 1:  # Positions to decrease
                            min_allocation = min(min_allocation, allocation[r, c])
                    
                    if min_allocation > 0 and min_allocation != float('inf'):
                        # Shift allocations along the loop
                        allocation[entering_var[0], entering_var[1]] += min_allocation
                        
                        for idx, (r, c) in enumerate(loop[1:], 1):
                            if idx % 2 == 1:  # Decrease
                                allocation[r, c] -= min_allocation
                            else:  # Increase
                                allocation[r, c] += min_allocation
                        
                        new_cost = np.sum(allocation * self.cost_matrix)
                        improvement = current_cost - new_cost
                        
                        iteration_info['status'] = 'Improved'
                        iteration_info['improvement'] = improvement
                        iteration_info['new_cost'] = new_cost
                        iteration_info['loop'] = loop
                        iteration_info['shift_amount'] = min_allocation
                        
                        current_cost = new_cost
                    else:
                        iteration_info['status'] = 'No improvement possible'
                        iteration_info['improvement'] = 0
                        self.stepping_stone_iterations.append(iteration_info)
                        break
                else:
                    iteration_info['status'] = 'Loop not found'
                    iteration_info['improvement'] = 0
                    self.stepping_stone_iterations.append(iteration_info)
                    break
            else:
                iteration_info['status'] = 'No entering variable'
                iteration_info['improvement'] = 0
                self.stepping_stone_iterations.append(iteration_info)
                break
            
            self.stepping_stone_iterations.append(iteration_info)
        
        self.allocation = allocation
        self.total_cost = current_cost
        return allocation, current_cost

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

def create_opportunity_cost_heatmap(opportunity_costs, suppliers, destinations, title):
    """Create heatmap for opportunity costs"""
    try:
        fig = go.Figure(data=go.Heatmap(
            z=opportunity_costs,
            x=destinations,
            y=suppliers,
            colorscale='RdYlGn_r',
            text=np.round(opportunity_costs, 2),
            texttemplate="%{text}",
            textfont={"size": 10, "color": "white"},
            hoverongaps=False,
            showscale=True,
            colorbar=dict(
                title="Opportunity Cost",
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
        st.error(f"Error creating opportunity cost heatmap: {e}")
        return go.Figure()

def create_stepping_stone_convergence_chart(iterations):
    """Create chart showing cost improvement through iterations"""
    try:
        iteration_nums = [iter_info['iteration'] for iter_info in iterations]
        costs = [iter_info['current_cost'] for iter_info in iterations]
        
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=iteration_nums,
            y=costs,
            mode='lines+markers',
            name='Transportation Cost',
            line=dict(width=3),
            marker=dict(size=8)
        ))
        
        fig.update_layout(
            title={
                'text': "Stepping Stone Method - Cost Convergence",
                'x': 0.5,
                'xanchor': 'center',
                'font': {'size': 16, 'color': 'darkblue'}
            },
            xaxis_title="Iteration",
            yaxis_title="Total Cost",
            height=400,
            font=dict(size=12),
            plot_bgcolor='white',
            showlegend=True
        )
        
        return fig
    except Exception as e:
        st.error(f"Error creating convergence chart: {e}")
        return go.Figure()

def create_cost_comparison_chart(results):
    try:
        methods = list(results.keys())
        costs = [results[method]['total_cost'] for method in methods]
        
        colors = ['#2E86AB', '#A23B72', '#F18F01', '#8E44AD']
        
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
    st.title("Transportation Method Solver with Stepping Stone Optimization")
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
        "Choose initial methods:",
        options=list(method_options.keys()),
        default=list(method_options.keys()),
        help="Select one or more initial methods to solve and then optimize with Stepping Stone"
    )
    
    apply_stepping_stone = st.checkbox(
        "Apply Stepping Stone Optimization", 
        value=True,
        help="Apply Stepping Stone method to optimize the initial solutions"
    )
    
    if selected_methods:
        st.write("**Selected Methods:**")
        for method in selected_methods:
            st.write(f"• **{method}**: {method_descriptions[method]}")
        
        if apply_stepping_stone:
            st.info("🔄 Stepping Stone optimization will be applied to improve the initial solutions")
    
    if st.button("Solve Transportation Problem", type="primary", use_container_width=True):
        if not selected_methods:
            st.warning("Please select at least one method.")
            st.stop()
            
        results = {}
        
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        try:
            total_methods = len(selected_methods) * (2 if apply_stepping_stone else 1)
            current_step = 0
            
            for method_name in selected_methods:
                method_code = method_options[method_name]
                
                # Solve initial method
                status_text.text(f"Solving using {method_name}...")
                progress_bar.progress(current_step / total_methods)
                
                try:
                    if method_code == "ncm":
                        steps = solver.northwest_corner_method()
                    elif method_code == "lcm":
                        steps = solver.least_cost_method()
                    elif method_code == "vam":
                        steps = solver.vogels_approximation_method()
                    
                    initial_allocation = solver.allocation.copy()
                    initial_cost = solver.total_cost
                    
                    results[method_name] = {
                        'allocation': initial_allocation,
                        'total_cost': initial_cost,
                        'steps': steps,
                        'method_type': 'initial'
                    }
                    
                    current_step += 1
                    
                    # Apply Stepping Stone if selected
                    if apply_stepping_stone:
                        status_text.text(f"Optimizing {method_name} with Stepping Stone...")
                        progress_bar.progress(current_step / total_methods)
                        
                        optimized_allocation, optimized_cost = solver.stepping_stone_optimization(initial_allocation)
                        
                        optimized_method_name = f"{method_name} + Stepping Stone"
                        results[optimized_method_name] = {
                            'allocation': optimized_allocation,
                            'total_cost': optimized_cost,
                            'steps': steps,  # Keep original steps
                            'method_type': 'optimized',
                            'initial_cost': initial_cost,
                            'improvement': initial_cost - optimized_cost,
                            'stepping_stone_iterations': solver.stepping_stone_iterations.copy()
                        }
                        
                        current_step += 1
                    
                except Exception as e:
                    st.error(f"Error solving with {method_name}: {e}")
                    continue
            
            status_text.text("All methods completed!")
            progress_bar.progress(1.0)
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
        
        # Display results for each method
        for method_name in results.keys():
            st.subheader(f"{method_name} Results")
            
            try:
                result = results[method_name]
                allocation = result['allocation']
                total_cost = result['total_cost']
                steps = result['steps']
                
                # Metrics row
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("Total Cost", f"{total_cost:,.2f}")
                with col2:
                    st.metric("Solution Steps", len(steps))
                with col3:
                    non_zero_allocations = np.count_nonzero(allocation)
                    st.metric("Active Routes", non_zero_allocations)
                with col4:
                    if result['method_type'] == 'optimized':
                        improvement = result['improvement']
                        st.metric("Cost Improvement", f"{improvement:,.2f}", delta=f"-{improvement:,.2f}")
                
                # Create tabs based on method type
                if result['method_type'] == 'optimized':
                    tab1, tab2, tab3, tab4, tab5 = st.tabs([
                        "Allocation Matrix", "Visualization", "Solution Steps", 
                        "Stepping Stone Details", "Optimization Progress"
                    ])
                else:
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
                
                # Additional tabs for optimized methods
                if result['method_type'] == 'optimized':
                    with tab4:
                        st.subheader("Stepping Stone Iteration Details")
                        
                        if 'stepping_stone_iterations' in result:
                            iterations = result['stepping_stone_iterations']
                            
                            if iterations:
                                # Summary of iterations
                                col1, col2, col3 = st.columns(3)
                                with col1:
                                    st.metric("Total Iterations", len(iterations))
                                with col2:
                                    initial_cost = result['initial_cost']
                                    final_cost = iterations[-1]['current_cost']
                                    total_improvement = initial_cost - final_cost
                                    st.metric("Total Improvement", f"{total_improvement:,.2f}")
                                with col3:
                                    final_status = iterations[-1]['status']
                                    st.metric("Final Status", final_status)
                                
                                # Iteration details
                                st.subheader("Iteration by Iteration Analysis")
                                
                                for i, iteration in enumerate(iterations):
                                    with st.expander(f"Iteration {iteration['iteration']} - {iteration['status']}"):
                                        col1, col2 = st.columns(2)
                                        
                                        with col1:
                                            st.write(f"**Current Cost:** {iteration['current_cost']:,.2f}")
                                            st.write(f"**Min Opportunity Cost:** {iteration['min_opportunity_cost']:.2f}")
                                            if 'improvement' in iteration:
                                                st.write(f"**Improvement:** {iteration['improvement']:,.2f}")
                                            
                                            if iteration['entering_variable']:
                                                supplier_idx, dest_idx = iteration['entering_variable']
                                                supplier_name = solver.suppliers[supplier_idx]
                                                dest_name = solver.destinations[dest_idx]
                                                st.write(f"**Entering Variable:** {supplier_name} → {dest_name}")
                                        
                                        with col2:
                                            # Show opportunity costs heatmap for this iteration
                                            if 'opportunity_costs' in iteration:
                                                fig_opp = create_opportunity_cost_heatmap(
                                                    iteration['opportunity_costs'],
                                                    solver.suppliers,
                                                    solver.destinations,
                                                    f"Opportunity Costs - Iteration {iteration['iteration']}"
                                                )
                                                st.plotly_chart(fig_opp, use_container_width=True, key=f"opp_cost_{method_name}_{i}")
                                        
                                        # Show allocation for this iteration
                                        if 'allocation' in iteration:
                                            st.write("**Allocation Matrix:**")
                                            iter_allocation_df = pd.DataFrame(
                                                iteration['allocation'].astype(int),
                                                index=solver.suppliers,
                                                columns=solver.destinations
                                            )
                                            st.dataframe(iter_allocation_df, use_container_width=True)
                            else:
                                st.info("No stepping stone iterations recorded.")
                        else:
                            st.info("Stepping stone details not available.")
                    
                    with tab5:
                        st.subheader("Optimization Progress")
                        
                        if 'stepping_stone_iterations' in result and result['stepping_stone_iterations']:
                            # Cost convergence chart
                            fig_convergence = create_stepping_stone_convergence_chart(result['stepping_stone_iterations'])
                            st.plotly_chart(fig_convergence, use_container_width=True, key=f"convergence_{method_name}")
                            
                            # Improvement summary table
                            st.subheader("Iteration Summary")
                            iteration_summary = []
                            for iteration in result['stepping_stone_iterations']:
                                iteration_summary.append({
                                    'Iteration': iteration['iteration'],
                                    'Cost': f"{iteration['current_cost']:,.2f}",
                                    'Min Opportunity Cost': f"{iteration['min_opportunity_cost']:.2f}",
                                    'Improvement': f"{iteration.get('improvement', 0):,.2f}",
                                    'Status': iteration['status']
                                })
                            
                            summary_df = pd.DataFrame(iteration_summary)
                            st.dataframe(summary_df, use_container_width=True, hide_index=True)
                        else:
                            st.info("No optimization progress data available.")
                
            except Exception as e:
                st.error(f"Error displaying results for {method_name}: {e}")
            
            st.markdown("---")
        
        # Methods comparison section
        if len(results) > 1:
            st.header("Methods Comparison")
            
            try:
                fig_comparison = create_cost_comparison_chart(results)
                st.plotly_chart(fig_comparison, use_container_width=True, key="cost_comparison")
                
                # Find best and worst methods
                best_method = min(results.keys(), key=lambda x: results[x]['total_cost'])
                best_cost = results[best_method]['total_cost']
                worst_method = max(results.keys(), key=lambda x: results[x]['total_cost'])
                worst_cost = results[worst_method]['total_cost']
                
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.success(f"**Best Method**: {best_method}")
                    st.write(f"Cost: {best_cost:,.2f}")
                with col2:
                    if len(results) > 2:
                        savings = worst_cost - best_cost
                        savings_pct = (savings / worst_cost) * 100 if worst_cost > 0 else 0
                        st.info(f"**Maximum Savings**: {savings:,.2f}")
                        st.write(f"({savings_pct:.1f}% improvement)")
                with col3:
                    st.error(f"**Highest Cost**: {worst_method}")
                    st.write(f"Cost: {worst_cost:,.2f}")
                
                # Detailed comparison table
                st.subheader("Detailed Comparison")
                comparison_data = []
                for method, result in results.items():
                    comparison_data.append({
                        'Method': method,
                        'Method Type': result['method_type'].title(),
                        'Total Cost': f"{result['total_cost']:,.2f}",
                        'Solution Steps': len(result['steps']),
                        'Active Routes': np.count_nonzero(result['allocation']),
                        'Cost vs Best': f"{result['total_cost'] - best_cost:,.2f}",
                        'Improvement from Initial': f"{result.get('improvement', 0):,.2f}" if result['method_type'] == 'optimized' else "N/A",
                        'Status': 'Optimal' if method == best_method else ('Suboptimal' if method == worst_method else 'Good')
                    })
                
                comparison_df = pd.DataFrame(comparison_data)
                st.dataframe(comparison_df, use_container_width=True, hide_index=True)
                
                # Stepping Stone effectiveness analysis
                st.subheader("Stepping Stone Optimization Effectiveness")
                
                initial_methods = [k for k, v in results.items() if v['method_type'] == 'initial']
                optimized_methods = [k for k, v in results.items() if v['method_type'] == 'optimized']
                
                if initial_methods and optimized_methods:
                    effectiveness_data = []
                    
                    for initial_method in initial_methods:
                        optimized_method = f"{initial_method} + Stepping Stone"
                        if optimized_method in results:
                            initial_cost = results[initial_method]['total_cost']
                            optimized_cost = results[optimized_method]['total_cost']
                            improvement = initial_cost - optimized_cost
                            improvement_pct = (improvement / initial_cost) * 100 if initial_cost > 0 else 0
                            
                            effectiveness_data.append({
                                'Initial Method': initial_method,
                                'Initial Cost': f"{initial_cost:,.2f}",
                                'Optimized Cost': f"{optimized_cost:,.2f}",
                                'Absolute Improvement': f"{improvement:,.2f}",
                                'Percentage Improvement': f"{improvement_pct:.2f}%",
                                'Optimization Status': 'Improved' if improvement > 0 else ('No Change' if improvement == 0 else 'Degraded')
                            })
                    
                    if effectiveness_data:
                        effectiveness_df = pd.DataFrame(effectiveness_data)
                        st.dataframe(effectiveness_df, use_container_width=True, hide_index=True)
                        
                        # Summary statistics
                        total_methods_optimized = len(effectiveness_data)
                        methods_improved = sum(1 for row in effectiveness_data if 'Improved' in row['Optimization Status'])
                        avg_improvement = np.mean([
                            float(row['Absolute Improvement'].replace(',', '')) 
                            for row in effectiveness_data
                        ])
                        
                        st.write("**Stepping Stone Summary:**")
                        col1, col2, col3 = st.columns(3)
                        with col1:
                            st.metric("Methods Optimized", total_methods_optimized)
                        with col2:
                            st.metric("Methods Improved", f"{methods_improved}/{total_methods_optimized}")
                        with col3:
                            st.metric("Average Improvement", f"{avg_improvement:,.2f}")
                
            except Exception as e:
                st.error(f"Error in comparison section: {e}")
                
if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        st.error(f"Application error: {e}")