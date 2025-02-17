from flask import Flask, request, jsonify
import requests
import pandas as pd
import joblib
import time
from sklearn.ensemble import RandomForestClassifier

# Load the model
model_filename = "random_forest_model.pkl"
model = joblib.load(model_filename)

CATEGORIES = [
    "greedy", "math", "implementation", "constructive algorithms", "brute force", "dp", 
    "data structures", "sortings", "binary search", "strings", "number theory", 
    "dfs and similar", "graphs", "two pointers", "trees", "bitmasks", "combinatorics", 
    "shortest paths", "dsu", "games", "hashing", "divide and conquer", "interactive", 
    "geometry", "*special", "probabilities", "flows", "ternary search", 
    "string suffix structures", "expression parsing", "Other"
]

API_URL = "https://codeforces.com/api/"

# Initialize Flask app
app = Flask(__name__)

# Helper functions (same as in your `predict.py`)
def fetch_user_rating(handle):
    url = f"{API_URL}user.rating?handle={handle}"
    response = requests.get(url)
    if response.status_code == 200:
        return response.json().get("result", [])
    else:
        print(f"Error fetching rating history for {handle}: {response.status_code}")
        return []

def fetch_user_status(handle):
    url = f"{API_URL}user.status?handle={handle}"
    response = requests.get(url)
    if response.status_code == 200:
        return response.json().get("result", [])
    else:
        print(f"Error fetching status for {handle}: {response.status_code}")
        return []

# ... [same other functions for calculating metrics] ...

def get_user_data(handle):
    rating_data = fetch_user_rating(handle)
    status_data = fetch_user_status(handle)
    
    contests_participated = len(rating_data)
    solved_counts, category_counts, unique_solved_problems = count_solved_problems_by_rating_and_category(status_data)
    unique_problems_solved = len(unique_solved_problems)
    avg_problem_rating = sum(
        problem.get("rating", 0)
        for entry in status_data
        if entry.get("verdict") == "OK"
        and (entry.get("problem", {}).get("contestId"), entry.get("problem", {}).get("index")) in unique_solved_problems
        and (problem := entry.get("problem", {})).get("rating") is not None
    ) / unique_problems_solved if unique_problems_solved > 0 else 0
    
    rating_progression = [r.get("newRating") for r in rating_data]
    best_rating = max(rating_progression, default=0)

    best_rank, avg_rank_50, avg_rank = calculate_performance_metrics(rating_data)
    engagement = calculate_engagement_metrics(rating_data)

    avg_submissions_per_day, acceptance_ratio = calculate_submission_metrics(status_data)

    user_data = {
        "User ID": handle,
        "Best Rating": best_rating,
        "Contests Participated": contests_participated,
        "Problems Solved": unique_problems_solved,
        "Average Problem Rating": avg_problem_rating,
        "Best Rank": best_rank,
        "Average Rank best 50": avg_rank_50,
        "Average Rank ": avg_rank,
        "Engagement (Contests per Month)": engagement,
        "Average Submissions per Day": avg_submissions_per_day,
        "Acceptance Ratio": acceptance_ratio,
    }

    user_data.update(solved_counts)
    user_data.update(category_counts)
    
    return user_data

# Endpoint for prediction
@app.route('/predict', methods=['POST'])
def predict_job_status():
    data = request.get_json()
    handle = data.get('handle', '').strip()

    if not handle:
        return jsonify({'error': 'Username is required'}), 400

    user_data = get_user_data(handle)
    
    # Prepare the data for prediction
    user_df = pd.DataFrame([user_data])
    
    missing_columns = [col for col in model.feature_names_in_ if col not in user_df.columns]
    if missing_columns:
        print(f"Warning: Missing columns in input data - {missing_columns}")
    
    user_df = user_df.reindex(columns=model.feature_names_in_, fill_value=0)
    
    # Make the prediction
    prediction = model.predict(user_df)
    return jsonify({'prediction': prediction[0]})

# Main entry point for running the app
if __name__ == '__main__':
    app.run(debug=True)
