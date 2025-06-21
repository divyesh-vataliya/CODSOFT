# Movie Rating Prediction

A machine learning project that predicts movie ratings using Random Forest Regressor based on various movie features.

## Features
- Predicts movie ratings based on:
  - Year of release
  - Duration
  - Genre
  - Director
  - Actors
  - Number of votes
- Visualizes movie data with various plots
- Provides model performance metrics

## Requirements
- Python 3.x
- pandas
- numpy
- matplotlib
- seaborn
- scikit-learn

## Installation
1. Clone this repository
2. Install required packages:
```bash
pip install pandas numpy matplotlib seaborn scikit-learn
```

## Usage
1. Run the script:
   ```bash
   python rating.py
   ```

## Dataset
The project uses the IMDb Movies India dataset, which is included in this repository.

## Output
The script will:
1. Display data analysis and visualizations
2. Show model performance metrics
3. Provide feature importance analysis
4. Allow you to predict ratings for new movies

## Example
```python
# Example prediction
predicted_rating = predict_movie_rating(
    year=2020,
    duration=112,
    genre='Drama',
    director='Sharan Sharma',
    actor1='Janvi Kapoor',
    actor2='Pankaj Tripathi',
    actor3='Angad Bedi',
    votes=36000
)
``` 