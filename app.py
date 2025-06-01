import streamlit as st
import pandas as pd
import numpy as np

st.set_page_config(page_title="Transportation Problem Solver", layout="wide")
st.title("🚚 Transportation Problem Solver")

st.markdown("""
This app supports solving the classical transportation problem using:
- **Northwest Corner Method (NCM)**
- **Least Cost Method (LCM)**
- **Vogel's Approximation Method (VAM)**

Each solution will be shown step-by-step.
""")

# File upload
supply_file = st.file_uploader("Upload Supply CSV", type="csv")
demand_file = st.file_uploader("Upload Demand CSV", type="csv")
cost_file = st.file_uploader("Upload Cost Matrix CSV", type="csv")

if supply_file and demand_file and cost_file:
    supply = pd.read_csv(supply_file)
    demand = pd.read_csv(demand_file)
    cost_matrix = pd.read_csv(cost_file, index_col=0)

    supply_values = supply['Supply'].tolist()
    demand_values = demand['Demand'].tolist()
    cost = cost_matrix.values.tolist()

    m, n = len(supply_values), len(demand_values)

    # Balance the problem if needed
    total_supply = sum(supply_values)
    total_demand = sum(demand_values)

    if total_supply > total_demand:
        st.warning("Demand is less than supply. Adding dummy demand.")
        for row in cost:
            row.append(0)
        demand_values.append(total_supply - total_demand)
        cost_matrix['Dummy'] = 0

    elif total_demand > total_supply:
        st.warning("Supply is less than demand. Adding dummy supply.")
        cost.append([0] * len(demand_values))
        supply_values.append(total_demand - total_supply)
        cost_matrix.loc['Dummy'] = [0] * len(demand_values)

    st.subheader("Balanced Cost Matrix")
    st.dataframe(pd.DataFrame(cost, index=cost_matrix.index, columns=cost_matrix.columns))

    method = st.selectbox("Select Method", ["Northwest Corner", "Least Cost", "Vogel's Approximation"])

    steps = []

    def add_step(desc, alloc):
        steps.append((desc, [row[:] for row in alloc]))

    def least_cost_method(supply, demand, cost):
        m, n = len(cost), len(cost[0])
        alloc = [[0]*n for _ in range(m)]
        supply = supply[:]
        demand = demand[:]
        available = [[True]*n for _ in range(m)]

        while True:
            min_cost = float('inf')
            x = y = -1
            for i in range(m):
                if supply[i] == 0:
                    continue
                for j in range(n):
                    if demand[j] == 0 or not available[i][j]:
                        continue
                    if cost[i][j] < min_cost:
                        min_cost = cost[i][j]
                        x, y = i, j
            if x == -1 or y == -1:
                break

            qty = min(supply[x], demand[y])
            alloc[x][y] = qty
            add_step(f"Allocate {qty} units to cell ({x}, {y}) with cost {cost[x][y]}", alloc)
            supply[x] -= qty
            demand[y] -= qty

            if supply[x] == 0:
                for j in range(n):
                    available[x][j] = False
            if demand[y] == 0:
                for i in range(m):
                    available[i][y] = False

        return alloc

    def vogel_method(supply, demand, cost):
        m, n = len(cost), len(cost[0])
        alloc = [[0]*n for _ in range(m)]
        supply = supply[:]
        demand = demand[:]
        active_rows = list(range(m))
        active_cols = list(range(n))

        while active_rows and active_cols:
            penalties = []
            for i in active_rows:
                row_costs = [cost[i][j] for j in active_cols]
                sorted_costs = sorted(row_costs)
                penalty = sorted_costs[1] - sorted_costs[0] if len(sorted_costs) > 1 else sorted_costs[0]
                penalties.append((penalty, i, 'row'))
            for j in active_cols:
                col_costs = [cost[i][j] for i in active_rows]
                sorted_costs = sorted(col_costs)
                penalty = sorted_costs[1] - sorted_costs[0] if len(sorted_costs) > 1 else sorted_costs[0]
                penalties.append((penalty, j, 'col'))

            penalties.sort(reverse=True)
            _, idx, typ = penalties[0]

            if typ == 'row':
                i = idx
                j = min(active_cols, key=lambda c: cost[i][c])
            else:
                j = idx
                i = min(active_rows, key=lambda r: cost[r][j])

            qty = min(supply[i], demand[j])
            alloc[i][j] = qty
            add_step(f"Allocate {qty} units to cell ({i}, {j}) with cost {cost[i][j]}", alloc)
            supply[i] -= qty
            demand[j] -= qty

            if supply[i] == 0:
                active_rows.remove(i)
            if demand[j] == 0:
                active_cols.remove(j)

        return alloc

    def northwest_corner(supply, demand, cost):
        m, n = len(supply), len(demand)
        alloc = [[0]*n for _ in range(m)]
        i = j = 0
        while i < m and j < n:
            qty = min(supply[i], demand[j])
            alloc[i][j] = qty
            add_step(f"Allocate {qty} units to cell ({i}, {j})", alloc)
            supply[i] -= qty
            demand[j] -= qty
            if supply[i] == 0:
                i += 1
            else:
                j += 1
        return alloc

    method_funcs = {
        "Northwest Corner": northwest_corner,
        "Least Cost": least_cost_method,
        "Vogel's Approximation": vogel_method
    }

    alloc = method_funcs[method](supply_values[:], demand_values[:], cost)

    # Compute total cost
    total_cost = sum(
        alloc[i][j] * cost[i][j] for i in range(len(alloc)) for j in range(len(alloc[0]))
    )

    st.subheader("Step-by-Step Allocation")
    for i, (desc, matrix) in enumerate(steps):
        st.markdown(f"**Step {i+1}: {desc}**")
        st.dataframe(pd.DataFrame(matrix, index=cost_matrix.index, columns=cost_matrix.columns))

    st.subheader("Final Allocation Table")
    alloc_df = pd.DataFrame(alloc, index=cost_matrix.index, columns=cost_matrix.columns)
    st.dataframe(alloc_df)

    st.success(f"💰 Total Transportation Cost: {total_cost:.2f}")

else:
    st.info("Please upload all three files (Supply, Demand, Cost Matrix) to proceed.")
