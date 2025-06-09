import pandas as pd
import numpy as np
import os

input_file_path = 'data/supply_chain_data.csv'
output_cost_matrix_file = 'data/matriks_biaya_final.csv'
output_supply_vector_file = 'data/vektor_pasokan_final.csv'
output_demand_vector_file = 'data/vektor_permintaan_final.csv'


def validate_and_clean_data(df):
    
    critical_columns = ['Product type', 'Location', 'Production volumes', 
                       'Number of products sold', 'Shipping costs']

    df = df.dropna(subset=critical_columns)
    
    string_columns = ['Product type', 'Location', 'Supplier name']
    for col in string_columns:
        if col in df.columns:
            df[col] = df[col].astype(str).str.strip().str.title()
    
    numeric_columns = ['Production volumes', 'Number of products sold', 'Shipping costs']
    for col in numeric_columns:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    
    df = df.dropna(subset=numeric_columns)
    
    return df


def aggregate_supply_demand(df):
    supply_agg = df.groupby('Product type').agg({
        'Production volumes': 'sum',
        'Supplier name': 'first'  
    }).reset_index()
    
    demand_agg = df.groupby('Location').agg({
        'Number of products sold': 'sum'
    }).reset_index()
    
    return supply_agg, demand_agg


def create_cost_matrix(df, suppliers, destinations):
    cost_pivot = df.pivot_table(
        index='Product type',
        columns='Location', 
        values='Shipping costs',
        aggfunc='mean'
    )
    
    cost_matrix_df = cost_pivot.reindex(index=suppliers, columns=destinations)
    cost_matrix = cost_matrix_df.values
    
    return cost_matrix, cost_matrix_df


def balance_supply_demand(supply_vector, demand_vector, suppliers, destinations, cost_matrix):
    
    total_supply = np.sum(supply_vector)
    total_demand = np.sum(demand_vector)
    
    print(f"\nBalancing Analysis:")
    print(f"Total Supply: {total_supply:,.0f}")
    print(f"Total Demand: {total_demand:,.0f}")
    print(f"Difference: {abs(total_supply - total_demand):,.0f}")
    
    if total_supply > total_demand:
        diff = total_supply - total_demand
        demand_vector = np.append(demand_vector, diff)
        destinations.append('Dummy_Destination')
        
        dummy_costs = np.zeros((len(suppliers), 1))
        cost_matrix = np.hstack([cost_matrix, dummy_costs])
        
    elif total_demand > total_supply:
        diff = total_demand - total_supply
        supply_vector = np.append(supply_vector, diff)
        suppliers.append('Dummy_Supplier')
        
        dummy_costs = np.zeros((1, len(destinations)))
        cost_matrix = np.vstack([cost_matrix, dummy_costs])
    
    return supply_vector, demand_vector, suppliers, destinations, cost_matrix


def save_processed_data(cost_matrix, supply_vector, demand_vector, suppliers, destinations):
    
    try:
        cost_df = pd.DataFrame(cost_matrix, index=suppliers, columns=destinations)
        supply_df = pd.DataFrame({'Supply': supply_vector}, index=suppliers)
        demand_df = pd.DataFrame({'Demand': demand_vector}, index=destinations)
        
        cost_df.index.name = 'Supplier'
        supply_df.index.name = 'Supplier' 
        demand_df.index.name = 'Destination'
        
        cost_df.to_csv(output_cost_matrix_file)
        supply_df.to_csv(output_supply_vector_file)
        demand_df.to_csv(output_demand_vector_file)
        
        return True
        
    except Exception as e:
        return False


def process_and_save_data(input_path, output_cost_matrix, output_supply_vector, output_demand_vector):

    try:
        df = pd.read_csv(input_path)
    except Exception as e:
        return False
    
    df = validate_and_clean_data(df)

    supply_agg, demand_agg = aggregate_supply_demand(df)
    
    suppliers = supply_agg['Product type'].tolist()
    destinations = demand_agg['Location'].tolist()
    
    supply_vector = supply_agg['Production volumes'].values.astype(float)
    demand_vector = demand_agg['Number of products sold'].values.astype(float)
    
    cost_matrix, cost_matrix_df = create_cost_matrix(df, suppliers, destinations)
    
    supply_vector, demand_vector, suppliers, destinations, cost_matrix = balance_supply_demand(
        supply_vector, demand_vector, suppliers, destinations, cost_matrix
    )

    success = save_processed_data(cost_matrix, supply_vector, demand_vector, suppliers, destinations)
    
    if success:
        print("Data berhasil diproses dan disimpan!")
        return True
    else:
        print("Gagal menyimpan data!")
        return False

if __name__ == "__main__":
    success = process_and_save_data(
        input_file_path,
        output_cost_matrix_file,
        output_supply_vector_file,
        output_demand_vector_file
    )