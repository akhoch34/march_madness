import pandas as pd

YEAR = 2025
DATA_DIR = f'../data/{YEAR}/'

submission_file = pd.read_csv(f'{DATA_DIR}SampleSubmissionStage2.csv')
mens_teams = pd.read_csv(f'{DATA_DIR}MTeams.csv').loc(axis=1)[['TeamID']].values.flatten().tolist()
womens_teams = pd.read_csv(f'{DATA_DIR}WTeams.csv').loc(axis=1)[['TeamID']].values.flatten().tolist()

# Function to classify rows
def classify_row(identifier):
    parts = identifier.split("_")
    team1, team2 = int(parts[-2]), int(parts[-1])
    if team1 in mens_teams and team2 in mens_teams:
        return "men"
    elif team1 in womens_teams and team2 in womens_teams:
        return "women"
    return None

# Apply classification
submission_file["category"] = submission_file.iloc[:, 0].apply(classify_row)

# Filter and write to respective files
submission_file[submission_file["category"] == "men"].iloc[:, :2].to_csv(f"{DATA_DIR}MSampleSubmissionStage2.csv", index=False)
submission_file[submission_file["category"] == "women"].iloc[:, :2].to_csv(f"{DATA_DIR}WSampleSubmissionStage2.csv", index=False)