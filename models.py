import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt

# Load data and drop missing values
df = pd.read_csv("StockandSentiment.csv", parse_dates=["Date"])
df = df.dropna(subset=["Headline", "Sentiment"])

# Feature engineering
df["DayOfWeek"] = df["Date"].dt.dayofweek
df["IsVolatile"] = df["Is Volatile"].astype(int)
df["Headline Count"] = df.groupby(["Symbol", "Date"])["Headline"].transform("count")

# Target variable
df["Percent Change Category"] = df["Percent Change Category"].map({"Drop": 0, "Stable": 1, "Spike": 2})

# Select features and target
features = df[["Sentiment", "Headline Count", "7-Day Volatility", "IsVolatile", "DayOfWeek"]]
target = df["Percent Change Category"]

# Split data
X_train, X_test, y_train, y_test = train_test_split(features, target)

# Scale features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Initialize models
models = {
    "Random Forest": RandomForestClassifier(
        n_estimators=50,
        max_depth=3,
        min_samples_split=10,
        min_samples_leaf=5,
    ),
    "KNN": KNeighborsClassifier(n_neighbors=5),
    "Gaussian NB": GaussianNB()
}

# Train and evaluate each model
results = {}
plt.figure(figsize=(15, 5))

for idx, (name, model) in enumerate(models.items(), 1):
    # Train model
    model.fit(X_train_scaled, y_train)
    
    # Make predictions
    y_pred = model.predict(X_test_scaled)
    
    # Store results
    results[name] = {
        'predictions': y_pred,
        'report': classification_report(y_test, y_pred, target_names=["Drop", "Stable", "Spike"], zero_division=0, output_dict=True)
    }
    
    # Print classification report
    print(f"\n{name} Classifier Report")
    print("-" * 50)
    print(classification_report(y_test, y_pred, target_names=["Drop", "Stable", "Spike"], zero_division=0))
    
    # Plot confusion matrix
    plt.subplot(1, 3, idx)
    cm = confusion_matrix(y_test, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Drop", "Stable", "Spike"])
    disp.plot(ax=plt.gca())
    plt.title(f"Confusion Matrix - {name}")

# Adjust layout and save combined confusion matrix plot
plt.tight_layout()
plt.savefig("confusion_matrix_combined.png")
plt.close()

# Compare model performances
print("\nModel Performance Comparison:")
print("-" * 50)
for name, result in results.items():
    accuracy = result['report']['accuracy']
    macro_f1 = result['report']['macro avg']['f1-score']
    print(f"{name}:")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Macro F1-score: {macro_f1:.4f}\n")