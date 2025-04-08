import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
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
X_train, X_test, y_train, y_test = train_test_split(features, target)

# Train Random Forest
model = RandomForestClassifier(n_estimators=100)
model.fit(X_train, y_train)

# Predict and evaluate
y_pred = model.predict(X_test)

print("Random Forest Classifier Report")
print(classification_report(y_test, y_pred, target_names=["Drop", "Stable", "Spike"]))

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Drop", "Stable", "Spike"])
disp.plot()
plt.title("Confusion Matrix - Random Forest")
plt.tight_layout()
plt.savefig("confusion_matrix_rf.png")
plt.close()
