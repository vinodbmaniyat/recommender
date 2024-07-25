import pandas as pd
import numpy as np
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, ndcg_score
import random

# Load Data from CSV Files
students_df = pd.read_csv('E:/Vinpyt/student_profiles.csv')
resources_df = pd.read_csv('E:/Vinpyt/educational_resources.csv')
feedback_df = pd.read_csv('E:/Vinpyt/historical_feedback.csv')

# Convert DataFrames to Dictionaries
students = {}
for _, row in students_df.iterrows():
    student_profile = {
        "student_id": row['student_id'],
        "grade": row['grade'],
        "learning_style": row['learning_style'],
        "performance_data": eval(row['performance_data']),
        "engagement_metrics": eval(row['engagement_metrics']),
        "emotional_wellbeing": eval(row['emotional_wellbeing'])
    }
    students[row['student_id']] = student_profile

# Map difficulty levels to numeric values
difficulty_mapping = {'easy': 1, 'medium': 2, 'hard': 3}
resources_df['difficulty'] = resources_df['difficulty'].map(difficulty_mapping)

resources = []
for _, row in resources_df.iterrows():
    resource = {
        "resource_id": row['resource_id'],
        "title": row['title'],
        "subject": row['subject'],
        "topic": row['topic'],
        "difficulty": row['difficulty'],
        "format": row['format'],
        "quality_score": row['quality_score']
    }
    resources.append(resource)

# Machine learning model for predicting resource effectiveness
class ResourceRecommender:
    def __init__(self):
        self.model = MLPRegressor(hidden_layer_sizes=(100, 100), max_iter=500)
        self.feature_names = ['quality_score', 'clarity', 'engagement', 'difficulty', 'stress_level', 'happiness_level']

    def train(self, X, y):
        self.model.fit(X, y)

    def predict(self, X):
        X_df = pd.DataFrame(X, columns=self.feature_names)
        return self.model.predict(X_df)

recommender = ResourceRecommender()

# Train the model using historical feedback data
def train_recommender(feedback_df, resources_df):
    feedback_with_quality = feedback_df.merge(resources_df[['resource_id', 'quality_score', 'difficulty']], on='resource_id', how='left')
    
    if 'difficulty_y' in feedback_with_quality.columns:
        feedback_with_quality.rename(columns={'difficulty_y': 'difficulty'}, inplace=True)
    if 'difficulty_x' in feedback_with_quality.columns:
        feedback_with_quality.drop(columns=['difficulty_x'], inplace=True)
    
    X = feedback_with_quality[recommender.feature_names]
    y = feedback_with_quality['rating']
    X = X.apply(pd.to_numeric)  # Ensure all data is numeric
    
    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Train the model on the training set
    recommender.train(X_train, y_train)
    
    # Evaluate the model on the testing set
    y_pred = recommender.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_test, y_pred)
    print(f"Mean Squared Error on the test set: {mse}")
    print(f"Root Mean Squared Error on the test set: {rmse}")
    print(f"Mean Absolute Error on the test set: {mae}")

# Ensure all necessary columns are present in the DataFrame
def ensure_columns(df, columns):
    for column in columns:
        if column not in df.columns:
            df[column] = 0  # or some default value
    return df

# Function to recommend resources based on student profile
def recommend_resources(student_id):
    student = students.get(student_id)
    if not student:
        return []

    recommended_resources = []

    for subject, topics in student['performance_data'].items():
        for topic, performance in topics.items():
            if performance == "struggling":
                filtered_resources = [resource for resource in resources
                                      if resource['subject'] == subject and
                                      resource['topic'] == topic and
                                      resource['format'] == student['learning_style']]
                
                # Prepare data for prediction
                if filtered_resources:
                    X = [[resource['quality_score'], 
                          0,  # Placeholder for clarity
                          student['engagement_metrics'][subject]['engagement_level'], 
                          resource['difficulty'], 
                          student['emotional_wellbeing']['stress_level'], 
                          student['emotional_wellbeing']['happiness_level']] for resource in filtered_resources]
                    predicted_scores = recommender.predict(X)

                    # Sort resources based on predicted scores
                    sorted_resources = [resource for _, resource in sorted(zip(predicted_scores, filtered_resources), key=lambda x: x[0], reverse=True)]
                
                    top_resources = sorted_resources[:3]
                    recommended_resources.extend(top_resources)

    return recommended_resources

# Function to collect detailed feedback
def collect_feedback(student_id, resource_id, rating, comments, clarity, engagement, difficulty, stress_level, happiness_level):
    print(f"Collected feedback from student {student_id} for resource {resource_id}: rating {rating}, comments: {comments}, clarity: {clarity}, engagement: {engagement}, difficulty: {difficulty}, stress_level: {stress_level}, happiness_level: {happiness_level}")
    
    # Update historical data for training
    feedback = pd.DataFrame([{
        "student_id": student_id,
        "resource_id": resource_id,
        "rating": rating,
        "comments": comments,
        "clarity": clarity,
        "engagement": engagement,
        "difficulty": difficulty,
        "stress_level": stress_level,
        "happiness_level": happiness_level
    }])
    
    global feedback_df
    feedback_df = pd.concat([feedback_df, feedback], ignore_index=True)
    train_recommender(feedback_df, resources_df)

# Function to update profiles and scores based on feedback
def update_profiles_and_scores(student_id, resource_id, rating, clarity, engagement, difficulty, stress_level, happiness_level):
    resource = next((r for r in resources if r['resource_id'] == resource_id), None)
    if resource:
        # Update quality score based on feedback
        resource['quality_score'] = (resource['quality_score'] + rating) / 2

    student = students.get(student_id)
    if student:
        # Adjust learning style dynamically (mock adjustment)
        if rating > 4:
            student['learning_style'] = "mixed"
        student['emotional_wellbeing']['stress_level'] = stress_level
        student['emotional_wellbeing']['happiness_level'] = happiness_level
        print(f"Updated learning style for student {student_id} to {student['learning_style']}")

# Example usage
train_recommender(feedback_df, resources_df)
student_id = students_df['student_id'].iloc[0]
recommendations = recommend_resources(student_id)

for resource in recommendations:
    print(f"Recommended Resource: {resource['title']} (Quality Score: {resource['quality_score']})")

# Simulate collecting detailed feedback
collect_feedback(student_id, resources[0]['resource_id'], 4.8, "Great explanation, very clear.", 5, 4, 3, 2, 8)
update_profiles_and_scores(student_id, resources[0]['resource_id'], 4.8, 5, 4, 3, 2, 8)

# Calculate additional metrics for recommendations
def calculate_additional_metrics(y_true, y_pred, k=10):
    ndcg = ndcg_score([y_true], [y_pred], k=k)
    return ndcg

# Function to calculate coverage metrics
def calculate_coverage_metrics(recommendations, all_users, all_items):
    recommended_users = set()
    recommended_items = set()
    for user, recs in recommendations.items():
        if recs:
            recommended_users.add(user)
            recommended_items.update([rec['resource_id'] for rec in recs])
    user_coverage = len(recommended_users) / len(all_users)
    item_coverage = len(recommended_items) / len(all_items)
    return user_coverage, item_coverage

# Function to calculate diversity metrics
def calculate_diversity_metrics(recommendations):
    intra_list_diversity = []
    all_recommended_items = []
    
    for user, recs in recommendations.items():
        items = [rec['resource_id'] for rec in recs]
        all_recommended_items.extend(items)
        
        if len(items) > 1:
            unique_items = len(set(items))
            diversity = unique_items / len(items)
            intra_list_diversity.append(diversity)
    
    aggregate_diversity = len(set(all_recommended_items)) / len(all_recommended_items)
    return np.mean(intra_list_diversity), aggregate_diversity

# Function to calculate novelty and serendipity metrics
def calculate_novelty_serendipity(recommendations, user_history):
    novelty_scores = []
    serendipity_scores = []
    
    for user, recs in recommendations.items():
        history = user_history.get(user, set())
        for rec in recs:
            if rec['resource_id'] not in history:
                novelty_scores.append(1)
                if random.random() > 0.5:  # Mock serendipity condition
                    serendipity_scores.append(1)
            else:
                novelty_scores.append(0)
                serendipity_scores.append(0)
    
    novelty = np.mean(novelty_scores)
    serendipity = np.mean(serendipity_scores)
    return novelty, serendipity

# Function to simulate and evaluate recommendations
def evaluate_recommendations():
    # Ensure feedback_df contains the necessary feature columns
    feedback_with_quality = feedback_df.merge(resources_df[['resource_id', 'quality_score', 'difficulty']], on='resource_id', how='left')
    feedback_with_quality = ensure_columns(feedback_with_quality, recommender.feature_names)
    
    y_true = feedback_with_quality['rating']
    y_pred = recommender.predict(feedback_with_quality[recommender.feature_names])
    
    # Calculate ranking metrics
    ndcg = calculate_additional_metrics(y_true, y_pred)
    print(f"Normalized Discounted Cumulative Gain (NDCG): {ndcg}")
    
    # Simulate recommendations for all users
    recommendations = {student_id: recommend_resources(student_id) for student_id in students.keys()}
    
    # Calculate coverage metrics
    user_coverage, item_coverage = calculate_coverage_metrics(recommendations, students.keys(), [r['resource_id'] for r in resources])
    print(f"User Coverage: {user_coverage}")
    print(f"Item Coverage: {item_coverage}")
    
    # Calculate diversity metrics
    intra_list_diversity, aggregate_diversity = calculate_diversity_metrics(recommendations)
    print(f"Intra-list Diversity: {intra_list_diversity}")
    print(f"Aggregate Diversity: {aggregate_diversity}")
    
    # Calculate novelty and serendipity metrics
    user_history = {student_id: set() for student_id in students.keys()}  # Mock user history
    novelty, serendipity = calculate_novelty_serendipity(recommendations, user_history)
    print(f"Novelty: {novelty}")
    print(f"Serendipity: {serendipity}")

# Evaluate recommendations
evaluate_recommendations()
