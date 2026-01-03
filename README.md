# Placement Prediction

This project predicts whether a student will get placed or not based on their academic and extracurricular data using **Logistic Regression**. It also explores the trends and factors influencing placement outcomes.

## Dataset

The dataset `Placement_BeginnerTask01.csv` contains the features:
- CGPA
- Internships
- Projects
- Workshops/Certifications
- Aptitude Test Score
- Soft Skills Rating
- Extracurricular Activities
- Placement Training
- SSC marks
- HSC marks
- Placement Status

Placement Status is the label or output class here.

## Model

- Uses Logistic Regression
- 80% of the dataset goes to training, while the remaining 20% is testing
- Performance: Achieves ~80% accuracy on test run

## Data Analysis

- Distribution plots for numeric features such as CGPA, Aptitude Test Score, and Soft Skills Rating
- Bar charts for features vs Placement Status
- Correlation bar chart showing feature influence on placement

All plots are saved as PNG images in `statistics/`.

## Usage

1. Install required packages

```bash
pip install -r requirements.txt
```

2. Run the script:

```bash
python main.py
```

## Requirements

- `numpy`
- `pandas`
- `matplotlib`
- `seaborn`
- `scikit-learn`
