🚚 Transportation Problem Solver (Streamlit App)

This project is a simple web application built using Streamlit to solve classical Transportation Problems in Operations Research. It allows users to upload custom data and solve transportation cost minimization using three methods:

Northwest Corner Method (NCM)

Least Cost Method (LCM)

Vogel's Approximation Method (VAM)

📂 File Upload Requirements

To run the app, prepare and upload the following three CSV files:

supply_vector.csv – contains columns:

Product type (optional)

Supply

demand_vector.csv – contains columns:

Location (optional)

Demand

cost_matrix.csv – a matrix with:

Rows as supply origins (product types)

Columns as demand destinations (locations)

Each cell containing the cost per unit to ship from source to destination

🧠 Features

Balances supply and demand automatically with dummy rows/columns

Provides step-by-step explanation of each method's allocation

Shows the final allocation table and total cost

Clean, readable layout using Streamlit

▶️ How to Run

Make sure you have Python 3.7+ installed. Then:

pip install streamlit pandas numpy
streamlit run app.py

Make sure your CSV files are formatted correctly and placed in the correct upload section in the app.

🛠 Future Improvements

Export step-by-step allocations to CSV

Add graphical visualizations for each step

Include degeneracy handling for tie-breaking

📄 License

This project is created for academic and research use.

