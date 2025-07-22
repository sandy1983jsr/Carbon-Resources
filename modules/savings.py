import pandas as pd

def update_savings_tracker(action_tracker_df):
    tracker = action_tracker_df.copy()
    tracker['Date'] = pd.to_datetime(tracker['Completed On'], errors='coerce')
    tracker = tracker.dropna(subset=['Date'])
    tracker = tracker.sort_values('Date')
    tracker['Cumulative Savings'] = tracker["Estimated Savings (₹ lakh)"].cumsum()
    return tracker[["Date", "Cumulative Savings"]]
