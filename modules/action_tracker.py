import pandas as pd

def initialize_tracker():
    columns = ["Action", "Owner", "Due Date", "Status", "Estimated Savings (₹ lakh)", "Completed On"]
    return pd.DataFrame(columns=columns)

def add_action(df, action, owner, due, status, savings, completed):
    new_row = {
        "Action": action,
        "Owner": owner,
        "Due Date": due,
        "Status": status,
        "Estimated Savings (₹ lakh)": savings,
        "Completed On": completed
    }
    return pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)

def update_action(df, row_idx, **kwargs):
    for key, val in kwargs.items():
        if key in df.columns:
            df.at[row_idx, key] = val
    return df

def completed_savings(df):
    done = df[df['Status'].str.lower() == 'done']
    return done["Estimated Savings (₹ lakh)"].sum()
