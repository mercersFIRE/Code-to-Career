from flask import Flask, render_template, request
import requests
import pandas as pd
import joblib

app = Flask(__name__)

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

# Fetch rating and status for user
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

# Calculate metrics
def count_solved_problems_by_rating_and_category(status_data):
    solved_counts = {
        "solved_800_1100": 0, "solved_1200_1400": 0, "solved_1500_1800": 0,
        "solved_1900_2100": 0, "solved_2100_plus": 0
    }
    category_counts = {category: 0 for category in CATEGORIES}
    
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

def calculate_performance_metrics(rating_data):
    ranks = [r.get("rank", None) for r in rating_data if r.get("rank") is not None]
    
    if ranks:
        best_rank = min(ranks)
        avg_rank = sum(ranks) / len(ranks)
        avg_rank_50 = sum(ranks[:50]) / 50 if len(ranks) >= 50 else avg_rank
    else:
        best_rank, avg_rank_50, avg_rank = None, None, None
    
    return best_rank, avg_rank_50, avg_rank

def calculate_engagement_metrics(rating_data):
    if rating_data:
        first_contest = rating_data[0]["ratingUpdateTimeSeconds"]
        last_contest = rating_data[-1]["ratingUpdateTimeSeconds"]
        engagement = len(rating_data) / ((last_contest - first_contest+1) / (60 * 60 * 24 * 30))
    else:
        engagement = 0
    return engagement

def calculate_submission_metrics(status_data):
    if not status_data:
        return 0, 0, 0.0
    
    first_submission = status_data[-1]["creationTimeSeconds"]
    last_submission = status_data[0]["creationTimeSeconds"]
    days_active = (last_submission - first_submission+1) / (60 * 60 * 24)
    



    total_submissions = len(status_data)
    accepted_submissions = sum(1 for entry in status_data if entry.get("verdict") == "OK")
    avg_submissions_per_day = total_submissions / days_active if days_active > 0 else 0
    acceptance_ratio = accepted_submissions / total_submissions if total_submissions > 0 else 0

    return avg_submissions_per_day, acceptance_ratio

def get_user_data(handle):
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
        )/ unique_problems_solved
        if unique_problems_solved > 0
        else 0
    )
    rating_progression = [r.get("newRating") for r in rating_data]
    best_rating = 0
    for rating in rating_progression :
        best_rating=max(best_rating,rating)
        
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

# Predict job status
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
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    username = request.form.get('username')
    
    if username:
        result = predict_job_status(username)
        return render_template('index.html', result=result)
    else:
        return render_template('index.html', error="Please provide a Codeforces username")

if __name__ == "__main__":
    app.run(debug=True)
