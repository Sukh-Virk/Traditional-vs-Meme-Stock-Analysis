import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt

# Load data and drop missing values
df = pd.read_csv("merged_stock_news.csv", parse_dates=["Date"])
df = df.dropna(subset=["Headline", "Sentiment"])

# Get feature columns
df["DayOfWeek"] = df["Date"].dt.dayofweek
df["IsVolatile"] = df["Is Volatile"].astype(int)
df["Headline Count"] = df.groupby(["Symbol", "Date"])["Headline"].transform("count")

# Target variable for classification
df["Percent Change Category"] = df["Percent Change Category"].map({"Drop": 0, "Stable": 1, "Spike": 2})

# Select features and target
features = df[["Sentiment", "Headline Count", "7-Day Volatility", "IsVolatile", "DayOfWeek"]]
target = df["Percent Change Category"]

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(features, target, random_state=42)

# Scale the features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train KNN
knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train_scaled, y_train)

# Make predictions
y_pred = knn.predict(X_test_scaled)

# Print classification report
print("K-Nearest Neighbors Classifier Report")
print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=["Drop", "Stable", "Spike"], zero_division=0))

# Plot confusion matrix
cm = confusion_matrix(y_test, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Drop", "Stable", "Spike"])
disp.plot()
plt.title("Confusion Matrix - KNN")
plt.tight_layout()
plt.savefig("confusion_matrix_knn.png")
plt.close()