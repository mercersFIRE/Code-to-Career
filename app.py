from flask import Flask, request, jsonify
import requests
import pandas as pd
import joblib

app = Flask(__name__)

# Load trained model
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

# Function to fetch Codeforces user data
def fetch_user_rating(handle):
    url = f"{API_URL}user.rating?handle={handle}"
    response = requests.get(url)
    return response.json().get("result", []) if response.status_code == 200 else []

def fetch_user_status(handle):
    url = f"{API_URL}user.status?handle={handle}"
    response = requests.get(url)
    return response.json().get("result", []) if response.status_code == 200 else []

def process_user_data(handle):
    rating_data = fetch_user_rating(handle)
    status_data = fetch_user_status(handle)
    
    # Compute features
    contests_participated = len(rating_data)
    best_rating = max((r.get("newRating", 0) for r in rating_data), default=0)
    
    # Rank Metrics
    ranks = [r.get("rank") for r in rating_data if r.get("rank") is not None]
    best_rank = min(ranks, default=0)
    avg_rank = sum(ranks) / len(ranks) if ranks else 0
    avg_rank_50 = sum(ranks[:50]) / 50 if len(ranks) >= 50 else avg_rank

    # Engagement Metrics
    engagement = (contests_participated / ((rating_data[-1]["ratingUpdateTimeSeconds"] - rating_data[0]["ratingUpdateTimeSeconds"]) / (60 * 60 * 24 * 30))) if rating_data else 0

    # Submission Metrics
    total_submissions = len(status_data)
    accepted_submissions = sum(1 for entry in status_data if entry.get("verdict") == "OK")
    acceptance_ratio = accepted_submissions / total_submissions if total_submissions > 0 else 0

    # Problem solving statistics
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

    unique_problems_solved = len(unique_solved_problems)
    avg_problem_rating = (sum(problem.get("rating", 0) for entry in status_data if entry.get("verdict") == "OK" and (problem := entry.get("problem", {})).get("rating") is not None) / unique_problems_solved) if unique_problems_solved > 0 else 0

    # Create DataFrame
    user_data = {
        "Best Rating": best_rating,
        "Contests Participated": contests_participated,
        "Problems Solved": unique_problems_solved,
        "Average Problem Rating": avg_problem_rating,
        "Best Rank": best_rank,
        "Average Rank best 50": avg_rank_50,
        "Average Rank ": avg_rank,
        "Engagement (Contests per Month)": engagement,
        "Acceptance Ratio": acceptance_ratio,
    }
    user_data.update(solved_counts)
    user_data.update(category_counts)

    return user_data

@app.route('/predict', methods=['GET'])
def predict():
    username = request.args.get("username")

    if not username:
        return jsonify({"error": "Please provide a Codeforces username"}), 400

    user_data = process_user_data(username)

    if not user_data:
        return jsonify({"error": "User not found or invalid data"}), 404

    user_df = pd.DataFrame([user_data])
    user_df = user_df.reindex(columns=model.feature_names_in_, fill_value=0)

    prediction = model.predict(user_df)
    return jsonify({"username": username, "predicted_job_status": prediction[0]})

if __name__ == '__main__':
    app.run(debug=True)
