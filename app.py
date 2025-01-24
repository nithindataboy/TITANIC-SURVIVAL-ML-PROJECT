from flask import Flask, render_template, request
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import os

# Set a non-interactive backend for Matplotlib
import matplotlib
matplotlib.use('Agg')

app = Flask(__name__)

# Load dataset
path = r'C:\Users\Appala nithin\Downloads\TITANIC-ML-PROJECT\titanic dataset.csv'
titanic = pd.read_csv(path, header=0, dtype={'Age': np.float64})

@app.route('/', methods=['GET', 'POST'])
def index():
    global titanic
    missing_values = titanic.isnull().sum().to_dict()  # Always compute missing values
    stats_summary = titanic.describe().to_html(classes="table table-striped")
    survived_counts = titanic['Survived'].value_counts().to_dict()

    # Handle POST request for cleaning
    message = None
    if request.method == 'POST':
        if 'fill_age' in request.form:
            mean_age = titanic['Age'].mean()
            titanic['Age'].fillna(mean_age, inplace=True)
            message = f"Filled missing 'Age' with mean value: {mean_age:.2f}"
        elif 'drop_cabin' in request.form:
            titanic.drop(columns=['Cabin'], inplace=True)
            message = "Dropped 'Cabin' column."
        elif 'fill_embarked' in request.form:
            mode_embarked = titanic['Embarked'].mode()[0]
            titanic['Embarked'].fillna(mode_embarked, inplace=True)
            message = f"Filled missing 'Embarked' with mode: {mode_embarked}"

    # Generate Visualizations
    sns.set(style="whitegrid")
    sns.countplot(x='Pclass', data=titanic, palette='pastel')
    plt.savefig('static/pclass_dist.png')
    plt.clf()

    sns.countplot(x='Sex', data=titanic, palette='Set2')
    plt.savefig('static/gender_dist.png')
    plt.clf()

    sns.histplot(titanic['Age'].dropna(), bins=20, kde=True, color='skyblue')
    plt.savefig('static/age_dist.png')
    plt.clf()

    return render_template(
        'index.html',
        data=titanic.head(20).to_html(classes="table table-striped"),
        missing_values=missing_values,  # Pass the missing_values variable here
        stats_summary=stats_summary,
        survived_counts=survived_counts,
        message=message
    )

if __name__ == '__main__':
    app.run(debug=True)
