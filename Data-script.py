import requests
import pandas as pd
import time
import os

API_URL = "https://codeforces.com/api/"

CATEGORIES = [
    "greedy", "math", "implementation", "constructive algorithms", "brute force", "dp", 
    "data structures", "sortings", "binary search", "strings", "number theory", 
    "dfs and similar", "graphs", "two pointers", "trees", "bitmasks", "combinatorics", 
    "shortest paths", "dsu", "games", "hashing", "divide and conquer", "interactive", 
    "geometry", "*special", "probabilities", "flows", "ternary search", 
    "string suffix structures", "expression parsing", "Other"
]

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

def calculate_performance_metrics(rating_data):
    number_of = 0
    tota = 0 
    ranks = [r.get("rank", None) for r in rating_data if r.get("rank") is not None]
    
    
    if ranks:
        best_rank = min(ranks)
        number_of=min(50,len(ranks))
        ranks.reverse()
        avg_rank=0 
        
        cccc=0
        for z in ranks:
            avg_rank+=z
            cccc+=1
            if cccc>=number_of:
                break
        avg_rank/=cccc        
        ranks.sort()
        
        cnt = 0
        for z in ranks:
            tota+=z
            cnt+=1
            if cnt >= number_of: 
                break
        avg_rank_50=tota/number_of
    else:
        best_rank, avg_rank_50,avg_rank = None, None,None
    
    #print(f"avg_rank_50 : {avg_rank_50} :: number_of : {number_of}  :: avg_rank {avg_rank}");
    return best_rank, avg_rank_50, avg_rank

def calculate_engagement_metrics(rating_data):
    if rating_data:
        first_contest = rating_data[0]["ratingUpdateTimeSeconds"]
        last_contest = rating_data[-1]["ratingUpdateTimeSeconds"]
        engagement = len(rating_data) / ((last_contest - first_contest) / (60 * 60 * 24 * 30))
    else:
        engagement = 0
    return engagement

def calculate_submission_metrics(status_data):
    if not status_data:
        return 0, 0, 0.0
    
    first_submission = status_data[-1]["creationTimeSeconds"]
    last_submission = status_data[0]["creationTimeSeconds"]
    days_active = (last_submission - first_submission) / (60 * 60 * 24)

    total_submissions = len(status_data)
    accepted_submissions = sum(1 for entry in status_data if entry.get("verdict") == "OK")
    avg_submissions_per_day = total_submissions / days_active if days_active > 0 else 0
    acceptance_ratio = accepted_submissions / total_submissions if total_submissions > 0 else 0

    return avg_submissions_per_day, acceptance_ratio

def get_user_data(handle, job_status):
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
        )/ unique_problems_solved
        if unique_problems_solved > 0
        else 0
    )
    rating_progression = [r.get("newRating") for r in rating_data]
    best_rating = 0
    for rating in rating_progression :
        best_rating=max(best_rating,rating)
        
    best_rank, avg_rank_50,avg_rank = calculate_performance_metrics(rating_data)
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
        "Current Job Status": job_status,
        "Engagement (Contests per Month)": engagement,
        "Average Submissions per Day": avg_submissions_per_day,
        "Acceptance Ratio": acceptance_ratio,
    }
    
    user_data.update(solved_counts)
    user_data.update(category_counts)
    
    #print(user_data)
    return user_data

def main():
    n = int(input("Enter the number of users: "))
    user_data_list = []
    
    for i in range(n):
        handle = input(f"Enter Codeforces ID #{i + 1}: ")
        job_status = input(f"Enter the current job status for {handle}: ")
        
        user_data = get_user_data(handle, job_status)
        user_data_list.append(user_data)
        
        time.sleep(1)
    
    output_file = "codeforces_user_data_with_category_counts.csv"
    df = pd.DataFrame(user_data_list)
    
    if os.path.exists(output_file):
        df.to_csv(output_file, mode='a', header=False, index=False)
    else:
        df.to_csv(output_file, mode='w', header=True, index=False)
    
    print(f"Data for {n} users saved to {output_file}")

if __name__ == "__main__":
    main()
