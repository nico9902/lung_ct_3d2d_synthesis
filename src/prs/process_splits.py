import pandas as pd
import numpy as np

def process_splits(input_csv, output_prefix="dataset", params=None):
    """
    Reads input CSV, filters Indeterminate, creates splits, saves to new CSVs.
    params: list of dicts with 'col_name' and 'output_suffix'
    """
    df = pd.read_csv(input_csv)
    
    # Set random seed for reproducibility
    np.random.seed(42)
    
    for param in params:
        col_name = param['col_name']
        suffix = param['suffix']
        
        print(f"Processing for label column: {col_name}")
        
        # 1. Filter Indeterminate
        # Assuming Indeterminate is represented as 'Indeterminate' string
        df_filtered = df[df[col_name] != 'Indeterminate'].copy()
        print(f"  Original size: {len(df)}, Filtered size: {len(df_filtered)}")
        
        # 2. Assign Splits (Train: 70%, Test: 20%, Val: 10%)
        from sklearn.model_selection import train_test_split
        
        # Split into training+val (80%) and test (20%)
        # Note: we use stratify to maintain label distribution
        train_val, test = train_test_split(
            df_filtered, 
            test_size=0.2, 
            random_state=42, 
            stratify=df_filtered[col_name]
        )
        
        # Split training+val into training (70% total) and val (10% total)
        # 10% / 80% = 0.125
        train, val = train_test_split(
            train_val, 
            test_size=0.125, 
            random_state=42, 
            stratify=train_val[col_name]
        )
        
        # Assign split column
        df_filtered.loc[train.index, 'split'] = 'train'
        df_filtered.loc[val.index, 'split'] = 'val'
        df_filtered.loc[test.index, 'split'] = 'test'
        
        # 3. Standardize label column
        # Rename the specific column (col_name) to 'target' so Dataset class doesn't need to change
        df_filtered.rename(columns={col_name: 'target'}, inplace=True)
        
        # Drop the other label column to avoid confusion
        other_col = 'label_nodule_mean' if col_name == 'label' else 'label'
        if other_col in df_filtered.columns:
            df_filtered.drop(columns=[other_col], inplace=True)
        
        output_file = f"{output_prefix}_{suffix}.csv"
        df_filtered.to_csv(output_file, index=False)
        print(f"  Saved to {output_file} with label column renamed to 'target' and '{other_col}' removed.")
        print(f"  Split counts:\n{df_filtered['split'].value_counts()}")

if __name__ == "__main__":
    params = [
        {'col_name': 'label', 'suffix': 'original'},
        {'col_name': 'label_nodule_mean', 'suffix': 'nodule_mean'}
    ]
    process_splits("patient_level_labels.csv", params=params)
