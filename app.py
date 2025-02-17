from flask import Flask, render_template, request
import requests
import pandas as pd
import joblib

# Initialize the Flask app
app = Flask(__name__)

# Load the trained model
model_filename = "random_forest_model.pkl"
model = joblib.load(model_filename)

# List of categories (same as in your predict.py)
CATEGORIES = [
    "greedy", "math", "implementation", "constructive algorithms", "brute force", "dp", 
    "data structures", "sortings", "binary search", "strings", "number theory", 
    "dfs and similar", "graphs", "two pointers", "trees", "bitmasks", "combinatorics", 
    "shortest paths", "dsu", "games", "hashing", "divide and conquer", "interactive", 
    "geometry", "*special", "probabilities", "flows", "ternary search", 
    "string suffix structures", "expression parsing", "Other"
]

API_URL = "https://codeforces.com/api/"

# Fetch user data from Codeforces API
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

# Data collection functions (same as in your `predict.py`)
def count_solved_problems_by_rating_and_category(status_data):
    solved_counts = {
        "solved_800_1100": 0, "solved_1200_1400": 0, "solved_1500_1800": 0,
        "solved_1900_2100": 0, "solved_2100_plus": 0
    }
    category_counts = {category: 0 for category in CATEGORIES}
    
    # Track unique, correct submissions
    unique_solved_problems = set()
    
    for entry in status_data:
        if entry.get("verdict") != "OK":
            continue
        
        problem = entry.get("problem", {})
        problem_id = (problem.get("contestId"), problem.get("index"))
        
        if problem_id in unique_solved_problems:
            continue
        unique_solved_problems.add(problem_id)
        
        problem_rating = problem.get("rating", 0)
        problem_tags = problem.get("tags", [])
        
        if 800 <= problem_rating <= 1100:
            solved_counts["solved_800_1100"] += 1
        elif 1200 <= problem_rating <= 1400:
            solved_counts["solved_1200_1400"] += 1
        elif 1500 <= problem_rating <= 1800:
            solved_counts["solved_1500_1800"] += 1
        elif 1900 <= problem_rating <= 2100:
            solved_counts["solved_1900_2100"] += 1
        elif problem_rating > 2100:
            solved_counts["solved_2100_plus"] += 1
        
        for tag in problem_tags:
            if tag in category_counts:
                category_counts[tag] += 1
            else:
                category_counts["Other"] += 1

    return solved_counts, category_counts, unique_solved_problems

def get_user_data(handle):
    print(f"Fetching data for user: {handle}")
    
    rating_data = fetch_user_rating(handle)
    status_data = fetch_user_status(handle)
    
    contests_participated = len(rating_data)
    solved_counts, category_counts, unique_solved_problems = count_solved_problems_by_rating_and_category(status_data)
    unique_problems_solved = len(unique_solved_problems)
    avg_problem_rating = (
        sum(
            problem.get("rating", 0)
            for entry in status_data
            if entry.get("verdict") == "OK"
            and (entry.get("problem", {}).get("contestId"), entry.get("problem", {}).get("index")) in unique_solved_problems
            and (problem := entry.get("problem", {})).get("rating") is not None
        ) / unique_problems_solved
        if unique_problems_solved > 0
        else 0
    )
    
    user_data = {
        "User ID": handle,
        "Contests Participated": contests_participated,
        "Problems Solved": unique_problems_solved,
        "Average Problem Rating": avg_problem_rating,
    }
    
    user_data.update(solved_counts)
    user_data.update(category_counts)
    
    return user_data

def predict_job_status(handle):
    user_data = get_user_data(handle)
    
    user_df = pd.DataFrame([user_data])
    
    missing_columns = [col for col in model.feature_names_in_ if col not in user_df.columns]
    if missing_columns:
        print(f"Warning: Missing columns in input data - {missing_columns}")
    
    user_df = user_df.reindex(columns=model.feature_names_in_, fill_value=0)
    
    prediction = model.predict(user_df)
    return prediction[0]

# Flask routes
@app.route('/')
def home():
    return render_template('index.html')  # Render the form

@app.route('/predict', methods=['POST'])
def predict():
    username = request.form.get('username')  # Get username from the form
    if username:
        result = predict_job_status(username)
        return render_template('index.html', result=result)  # Show the result
    else:
        return render_template('index.html', result="Please provide a valid username.")  # Handle missing input

if __name__ == "__main__":
    app.run(debug=True)
