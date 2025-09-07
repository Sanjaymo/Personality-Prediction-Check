import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    confusion_matrix, ConfusionMatrixDisplay,
    accuracy_score, f1_score, precision_score, recall_score
)

# ================== Load dataset ==================
df = pd.read_csv("personality_dataset.csv")

target = 'Personality'
X = df.drop(target, axis=1)
y = df[target]

# Encode categorical features in X
encoders = {}
for col in X.columns:
    if X[col].dtype == 'object':
        le = LabelEncoder()
        X.loc[:, col] = le.fit_transform(X[col].str.lower())  # normalize to lowercase
        encoders[col] = le

# Encode target y
if y.dtype == 'object':
    le_y = LabelEncoder()
    y = le_y.fit_transform(y)

# Train/Test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Train model
clf = RandomForestClassifier(n_estimators=100, max_depth=3, random_state=42)
clf.fit(X_train, y_train)

# ================== Evaluation ==================
y_pred = clf.predict(X_test)

print("Accuracy (Test):", accuracy_score(y_test, y_pred))
print("F1 (Test):", f1_score(y_test, y_pred, average='weighted'))
print("Precision (Test):", precision_score(y_test, y_pred, average='weighted'))
print("Recall (Test):", recall_score(y_test, y_pred, average='weighted'))

cm = confusion_matrix(y_test, y_pred)
labels = le_y.inverse_transform(np.unique(y))
ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels).plot(cmap="Blues")
plt.title("Confusion Matrix - Test")
plt.show()

# ================== Helpers ==================
def get_int_input(prompt, min_val=0, max_val=None):
    while True:
        try:
            val = int(input(prompt))
            if val < min_val:
                print(f"Value must be >= {min_val}")
                continue
            if max_val is not None and val > max_val:
                print(f"Value must be <= {max_val}")
                continue
            return val
        except ValueError:
            print("Please enter a valid integer.")

def get_yes_no_input(prompt):
    while True:
        print(f"{prompt} (1 = Yes, 2 = No): ", end="")
        choice = input().strip()
        if choice == "1":
            return "yes"
        elif choice == "2":
            return "no"
        else:
            print("Invalid choice. Please type 1 or 2.")

# ================== Prediction Function ==================
def predict_personality(new_data: pd.DataFrame):
    for col in new_data.columns:
        if col in encoders:
            new_data.loc[:, col] = new_data[col].str.lower()
            new_data.loc[:, col] = encoders[col].transform(new_data[col])
    preds = clf.predict(new_data)
    return le_y.inverse_transform(preds)

# ================== User Input ==================
def get_user_input():
    rows = get_int_input("Enter number of samples: ", min_val=1)
    data = []
    for i in range(rows):
        print(f"\n--- Enter details for sample {i+1} ---")
        Time_spent_Alone = get_int_input("Time spent alone (hours/day): ", min_val=0, max_val=24)
        Stage_fear = get_yes_no_input("Stage fear")
        Social_event_attendance = get_int_input("Social event attendance (times/month): ", min_val=0)
        Going_outside = get_int_input("Going outside (times/week): ", min_val=0, max_val=7)
        Drained_after_socializing = get_yes_no_input("Drained after socializing")
        Friends_circle_size = get_int_input("Friends circle size: ", min_val=0)
        Post_frequency = get_int_input("Post frequency (posts/week): ", min_val=0)

        data.append([
            Time_spent_Alone, Stage_fear, Social_event_attendance,
            Going_outside, Drained_after_socializing,
            Friends_circle_size, Post_frequency
        ])

    cols = [
        'Time_spent_Alone', 'Stage_fear', 'Social_event_attendance',
        'Going_outside', 'Drained_after_socializing',
        'Friends_circle_size', 'Post_frequency'
    ]
    return pd.DataFrame(data, columns=cols)

# ================== Run Prediction ==================
if __name__ == "__main__":
    user_df = get_user_input()
    results = predict_personality(user_df)
    print("\n=== Predictions ===")
    for i, res in enumerate(results):
        print(f"Sample {i+1}: {res}")
