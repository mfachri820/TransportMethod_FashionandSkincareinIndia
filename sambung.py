import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

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
            cost_df = pd.read_csv('data/matriks_biaya_final.csv', index_col=0)
            self.cost_matrix = cost_df.values.astype(float)

            supply_df = pd.read_csv('data/vektor_pasokan_final.csv', index_col=0)
            self.supply = supply_df['Supply'].values.astype(float)

            demand_df = pd.read_csv('data/vektor_permintaan_final.csv', index_col=0)
            self.demand = demand_df['Demand'].values.astype(float)

            self.suppliers = cost_df.index.tolist()
            self.destinations = cost_df.columns.tolist()

            self.balance_problem()

            return True
        except Exception as e:
            st.error(f"Error loading data: {e}")
            return False

    def balance_problem(self):
        total_supply = np.sum(self.supply)
        total_demand = np.sum(self.demand)

        if total_supply > total_demand:
            dummy_col = np.zeros((self.cost_matrix.shape[0], 1))
            self.cost_matrix = np.hstack((self.cost_matrix, dummy_col))
            self.demand = np.append(self.demand, total_supply - total_demand)
            self.destinations.append("Dummy")
        elif total_demand > total_supply:
            dummy_row = np.zeros((1, self.cost_matrix.shape[1]))
            self.cost_matrix = np.vstack((self.cost_matrix, dummy_row))
            self.supply = np.append(self.supply, total_demand - total_supply)
            self.suppliers.append("Dummy")

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

def main():
    st.title("Optimasi Transportasi - Least Cost Method")

    solver = TransportMethodSolver()

    if not solver.load_data():
        st.stop()

    with st.sidebar:
        st.header("Data Ringkasan")
        total_supply = np.sum(solver.supply)
        total_demand = np.sum(solver.demand)

        st.metric("Total Supply", f"{total_supply:,.0f}")
        st.metric("Total Demand", f"{total_demand:,.0f}")
        st.write("Model telah otomatis diseimbangkan jika supply ≠ demand.")

        st.subheader("Matriks Biaya")
        st.dataframe(pd.DataFrame(solver.cost_matrix, index=solver.suppliers, columns=solver.destinations))

        st.subheader("Supply")
        st.dataframe(pd.DataFrame({'Supplier': solver.suppliers, 'Supply': solver.supply.astype(int)}), hide_index=True)

        st.subheader("Demand")
        st.dataframe(pd.DataFrame({'Destination': solver.destinations, 'Demand': solver.demand.astype(int)}), hide_index=True)

    if st.button("Jalankan Least Cost Method", type="primary", use_container_width=True):
        steps = solver.least_cost_method()

        st.success(f"Total Biaya Transportasi: Rp{solver.total_cost:,.2f}")
        st.subheader("Matriks Alokasi")
        allocation_df = pd.DataFrame(
            solver.allocation.astype(int),
            index=solver.suppliers,
            columns=solver.destinations
        )
        st.dataframe(allocation_df, use_container_width=True)

        st.subheader("Heatmap Alokasi")
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.heatmap(allocation_df, annot=True, fmt="g", cmap="YlGnBu", ax=ax)
        ax.set_xlabel("Tujuan")
        ax.set_ylabel("Sumber")
        st.pyplot(fig)

        st.subheader("Langkah Distribusi")
        steps_df = pd.DataFrame(steps)
        st.dataframe(steps_df, use_container_width=True, hide_index=True)

        st.download_button(
            "Download Hasil Alokasi (CSV)",
            allocation_df.to_csv().encode('utf-8'),
            file_name="hasil_lcm_allocation.csv",
            mime="text/csv"
        )

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        st.error(f"Application error: {e}")
