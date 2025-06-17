# Movie Rating Prediction using Random Forest Regressor

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import root_mean_squared_error, mean_absolute_error, r2_score

import warnings
warnings.filterwarnings("ignore")
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)


print("\nLoading the dataset...")
df = pd.read_csv(r'C:\vs\CODSOFT\Movie Rating Prediction\IMDb Movies India.csv', encoding='latin1')


print(f"\nInitial dataset shape: {df.shape}")

print("\nFirst 5 rows:")
print(df.head())

print("\nData Info:")
print(df.info())

print("\nMissing values before cleaning:")
print(df.isnull().sum())

# Data cleaning
# -----------------------------
# Year to numeric
df['Year'] = pd.to_numeric(df['Year'], errors='coerce')

# Convert Duration from '123 min' to float
df['Duration'] = df['Duration'].str.replace(' min', '')
df['Duration'] = pd.to_numeric(df['Duration'], errors='coerce')

# Clean up Votes column - remove commas and convert to float
df['Votes'] = df['Votes'].astype(str).str.replace(',', '')
df['Votes'] = pd.to_numeric(df['Votes'], errors='coerce')


df['Actor 1'] = df['Actor 1'].fillna('Unknown')
df['Actor 2'] = df['Actor 2'].fillna('Unknown')
df['Actor 3'] = df['Actor 3'].fillna('Unknown')
df['Votes'] = df['Votes'].fillna(0)
df['Genre'] = df['Genre'].fillna('Unknown')
df['Director'] = df['Director'].fillna('Unknown')

df = df.dropna(subset=['Rating'])

print(f"\nShape after initial cleaning: {df.shape}")

# Top 10 highest-rated movies in the dataset.
plt.figure(figsize=(10,6))
top_movies = df.sort_values('Rating', ascending=False).head(10)
sns.barplot(x='Rating', y='Name', data=top_movies, palette='coolwarm')
plt.title('Top 10 Movies by Rating')
plt.xlabel('Rating')
plt.ylabel('Movie Name')
plt.tight_layout()
plt.show()

# Distribution of movie ratings
plt.figure(figsize=(8,5))
sns.histplot(df['Rating'], bins=20, kde=True, color='mediumseagreen')
plt.title('Distribution of Movie Ratings')
plt.xlabel('Rating')
plt.ylabel('Count')
plt.tight_layout()
plt.show()

# Number of movies per genre
plt.figure(figsize=(12,6))
genre_counts = df['Genre'].value_counts().head(10)
sns.barplot(x=genre_counts.values, y=genre_counts.index, palette='cubehelix')
plt.title('Top 10 Genres by Number of Movies')
plt.xlabel('Number of Movies')
plt.ylabel('Genre')
plt.tight_layout()
plt.show()

# Data preprocessing
# -----------------------------
# Feature encoding
le = LabelEncoder()
df['Genre_encoded'] = le.fit_transform(df['Genre'])
df['Director_encoded'] = le.fit_transform(df['Director'])
df['Actor1_encoded'] = le.fit_transform(df['Actor 1'])
df['Actor2_encoded'] = le.fit_transform(df['Actor 2'])
df['Actor3_encoded'] = le.fit_transform(df['Actor 3'])


features = ['Year', 'Duration', 'Votes', 'Genre_encoded', 'Director_encoded', 'Actor1_encoded', 'Actor2_encoded', 'Actor3_encoded']
X = df[features]
y = df['Rating']

print(f"\nFinal dataset shape: {X.shape}")

# Check if we have enough data
if len(df) < 5:
    raise ValueError("Not enough data to split into train and test sets. Add more valid rows.")

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


print("\nTraining Random Forest Regressor...")
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Predict ratings 
y_pred = model.predict(X_test)

# Evaluate the model's performance 
mse = root_mean_squared_error(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("\nModel Evaluation:")
print(f"Root Mean Squared Error: {mse:.2f}")
print(f"Mean Absolute Error: {mae:.2f}")
print(f"R2 Score: {r2:.2f}")

# Feature importance
importance = model.feature_importances_
feat_imp = pd.DataFrame({'Feature': features, 'Importance': importance}).sort_values('Importance', ascending=False)

print("\nFeature Importance:")
print(feat_imp)

# Actual vs Predicted Ratings
plt.figure(figsize=(8,6))
plt.scatter(y_test, y_pred, alpha=0.7, color='teal')
plt.xlabel('Actual Ratings')
plt.ylabel('Predicted Ratings')
plt.title('Actual vs Predicted Movie Ratings')
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')  # Diagonal line
plt.tight_layout()
plt.show()


# Function to predict movie rating based on input features
def predict_movie_rating(year, duration, genre, director, actor1, actor2, actor3, votes):
    genre_encoded = le.fit_transform([genre])[0]
    director_encoded = le.fit_transform([director])[0]
    actor1_encoded = le.fit_transform([actor1])[0]
    actor2_encoded = le.fit_transform([actor2])[0]
    actor3_encoded = le.fit_transform([actor3])[0]

    input_data = np.array([[year, duration, genre_encoded, director_encoded, actor1_encoded, actor2_encoded, actor3_encoded, votes]])
    return model.predict(input_data)[0]

# Example usage
print("\nExample Prediction: Gunjan Saxena (Actual Rating: 5.5)")
pred = predict_movie_rating(2020, 112, 'Drama', 'Sharan Sharma', 'Janvi Kapoor', 'Pankaj Tripathi', 'Angad Bedi', 36000)
print(f"\nPredicted Rating: {pred:.2f}")


