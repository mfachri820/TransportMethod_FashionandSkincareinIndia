import streamlit as st
import pandas as pd
import numpy as np

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
        self.stepping_stone_iterations = []
        
    def load_data(self):
        try:
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

        for i, pos1 in enumerate(basic_vars):
            for j, pos2 in enumerate(basic_vars[i+1:], i+1):
                for k, pos3 in enumerate(basic_vars[j+1:], j+1):
                    if can_form_rectangle(start_pos, pos1, pos2, pos3):
                        return [start_pos, pos1, pos2, pos3]
        
        if len(basic_vars) >= 3:
            return [start_pos] + basic_vars[:3]
        return None

    def calculate_opportunity_cost(self, allocation, i, j):
        """Calculate opportunity cost for non-basic variable (i,j)"""
        basic_vars = self.get_basic_variables(allocation)
        
        loop = self.find_loop(basic_vars, (i, j))
        
        if not loop:
            return 0  
        
        opportunity_cost = self.cost_matrix[i, j]    

        for idx, (r, c) in enumerate(loop[1:], 1):
            if idx % 2 == 1:  
                opportunity_cost -= self.cost_matrix[r, c]
            else:  
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
            
            min_opportunity_cost = 0
            entering_var = None
            
            for i in range(m):
                for j in range(n):
                    if allocation[i, j] == 0: 
                        opp_cost = self.calculate_opportunity_cost(allocation, i, j)
                        
                        if opp_cost < min_opportunity_cost:
                            min_opportunity_cost = opp_cost
                            entering_var = (i, j)
            
            if min_opportunity_cost >= 0:
                break
            
            if entering_var:
                basic_vars = self.get_basic_variables(allocation)
                loop = self.find_loop(basic_vars, entering_var)
                
                if loop and len(loop) >= 4:
                    min_allocation = float('inf')
                    for idx, (r, c) in enumerate(loop[1:], 1):
                        if idx % 2 == 1:
                            min_allocation = min(min_allocation, allocation[r, c])
                    
                    if min_allocation > 0 and min_allocation != float('inf'):
                        allocation[entering_var[0], entering_var[1]] += min_allocation
                        
                        for idx, (r, c) in enumerate(loop[1:], 1):
                            if idx % 2 == 1:  
                                allocation[r, c] -= min_allocation
                            else:  
                                allocation[r, c] += min_allocation
                        
                        current_cost = np.sum(allocation * self.cost_matrix)
                    else:
                        break
                else:
                    break
            else:
                break
        
        self.allocation = allocation
        self.total_cost = current_cost
        return allocation, current_cost

    def modi_optimization(self, initial_allocation, max_iterations=10):
        """Optimize allocation using MODI (Modified Distribution) method"""
        allocation = initial_allocation.copy()
        current_cost = np.sum(allocation * self.cost_matrix)
        
        for iteration in range(max_iterations):
            m, n = allocation.shape
            basic_vars = self.get_basic_variables(allocation)
            
            ui = [None] * m
            vj = [None] * n
            ui[0] = 0  
            
            changed = True
            while changed:
                changed = False
                for i, j in basic_vars:
                    if ui[i] is not None and vj[j] is None:
                        vj[j] = self.cost_matrix[i, j] - ui[i]
                        changed = True
                    elif vj[j] is not None and ui[i] is None:
                        ui[i] = self.cost_matrix[i, j] - vj[j]
                        changed = True
            
            min_opportunity_cost = 0
            entering_var = None
            
            for i in range(m):
                for j in range(n):
                    if allocation[i, j] == 0:  # Non-basic variable
                        if ui[i] is not None and vj[j] is not None:
                            opp_cost = self.cost_matrix[i, j] - ui[i] - vj[j]
                            if opp_cost < min_opportunity_cost:
                                min_opportunity_cost = opp_cost
                                entering_var = (i, j)
            
            if min_opportunity_cost >= 0:
                break
            
            if entering_var:
                loop = self.find_loop(basic_vars, entering_var)
                if loop and len(loop) >= 4:
                    min_allocation = float('inf')
                    for idx, (r, c) in enumerate(loop[1:], 1):
                        if idx % 2 == 1: 
                            min_allocation = min(min_allocation, allocation[r, c])
                    
                    if min_allocation > 0 and min_allocation != float('inf'):
                        allocation[entering_var[0], entering_var[1]] += min_allocation
                        for idx, (r, c) in enumerate(loop[1:], 1):
                            if idx % 2 == 1:
                                allocation[r, c] -= min_allocation
                            else:
                                allocation[r, c] += min_allocation
                        
                        current_cost = np.sum(allocation * self.cost_matrix)
                    else:
                        break
                else:
                    break
            else:
                break
        
        self.allocation = allocation
        self.total_cost = current_cost
        return allocation, current_cost

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
    
    selected_methods = st.multiselect(
        "Choose initial methods:",
        options=list(method_options.keys()),
        default=list(method_options.keys()),
        help="Select one or more initial methods to solve"
    )
    
    optimization_method = st.selectbox(
        "Choose optimization method:",
        options=["None", "Stepping Stone", "MODI", "Both"],
        index=1,
        help="Select optimization method to apply"
    )
    
    apply_stepping_stone = optimization_method in ["Stepping Stone", "Both"]
    apply_modi = optimization_method in ["MODI", "Both"]
    
    if st.button("Solve Transportation Problem", type="primary", use_container_width=True):
        if not selected_methods:
            st.warning("Please select at least one method.")
            st.stop()
            
        results = {}
        
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        try:
            multiplier = 1
            if apply_stepping_stone: multiplier += 1
            if apply_modi: multiplier += 1
            total_methods = len(selected_methods) * multiplier
            current_step = 0
            
            for method_name in selected_methods:
                method_code = method_options[method_name]
                
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
                    
                    if apply_stepping_stone:
                        status_text.text(f"Optimizing {method_name} with Stepping Stone...")
                        progress_bar.progress(current_step / total_methods)
                        
                        optimized_allocation, optimized_cost = solver.stepping_stone_optimization(initial_allocation)
                        
                        optimized_method_name = f"{method_name} + Stepping Stone"
                        results[optimized_method_name] = {
                            'allocation': optimized_allocation,
                            'total_cost': optimized_cost,
                            'steps': steps,
                            'method_type': 'optimized',
                            'initial_cost': initial_cost,
                            'improvement': initial_cost - optimized_cost
                        }
                        
                        current_step += 1
                    
                    if apply_modi:
                        status_text.text(f"Optimizing {method_name} with MODI...")
                        progress_bar.progress(current_step / total_methods)
                        
                        modi_allocation, modi_cost = solver.modi_optimization(initial_allocation)
                        
                        modi_method_name = f"{method_name} + MODI"
                        results[modi_method_name] = {
                            'allocation': modi_allocation,
                            'total_cost': modi_cost,
                            'steps': steps,
                            'method_type': 'optimized',
                            'initial_cost': initial_cost,
                            'improvement': initial_cost - modi_cost
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
        
        for method_name in results.keys():
            st.subheader(f"{method_name} Results")
            
            try:
                result = results[method_name]
                allocation = result['allocation']
                total_cost = result['total_cost']
                steps = result['steps']
                
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
                        st.metric("Cost Improvement", f"{improvement:,.2f}")
                
                st.subheader("Allocation Matrix")
                allocation_df = pd.DataFrame(
                    allocation.astype(int), 
                    index=solver.suppliers, 
                    columns=solver.destinations
                )
                st.dataframe(allocation_df, use_container_width=True)
                
                st.subheader("Solution Steps")
                steps_df = pd.DataFrame(steps)
                st.dataframe(steps_df, use_container_width=True, hide_index=True)
                
            except Exception as e:
                st.error(f"Error displaying results for {method_name}: {e}")
            
            st.markdown("---")
        
        if len(results) > 1:
            st.header("Methods Comparison")
            
            try:
                best_method = min(results.keys(), key=lambda x: results[x]['total_cost'])
                best_cost = results[best_method]['total_cost']
                
                st.success(f"**Best Method**: {best_method} with cost: {best_cost:,.2f}")
                
                comparison_data = []
                for method, result in results.items():
                    comparison_data.append({
                        'Method': method,
                        'Method Type': result['method_type'].title(),
                        'Total Cost': f"{result['total_cost']:,.2f}",
                        'Solution Steps': len(result['steps']),
                        'Active Routes': np.count_nonzero(result['allocation']),
                        'Cost vs Best': f"{result['total_cost'] - best_cost:,.2f}",
                        'Improvement from Initial': f"{result.get('improvement', 0):,.2f}" if result['method_type'] == 'optimized' else "N/A"
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