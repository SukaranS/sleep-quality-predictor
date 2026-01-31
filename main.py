import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score

# ===================== LOAD DATA =====================
data = pd.read_csv("data/sleep_data.csv")

# ===================== ENCODING =====================
gender_encoder = LabelEncoder()
bmi_encoder = LabelEncoder()
sleep_encoder = LabelEncoder()

data["Gender"] = gender_encoder.fit_transform(data["Gender"])
data["BMI_Category"] = bmi_encoder.fit_transform(data["BMI_Category"])
data["Sleep_Quality"] = sleep_encoder.fit_transform(data["Sleep_Quality"])

# ===================== SPLIT DATA =====================
X = data.drop("Sleep_Quality", axis=1)
y = data["Sleep_Quality"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# ===================== TRAIN MODEL =====================
model = DecisionTreeClassifier()
model.fit(X_train, y_train)

# ===================== MODEL ACCURACY =====================
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print("\nModel Accuracy:", accuracy)

# ===================== USER PREDICTION =====================
print("\n===== Sleep Quality Predictor =====")

age = int(input("Enter Age: "))
gender = input("Enter Gender (Male/Female): ").capitalize()
sleep_duration = float(input("Sleep Duration (hours): "))
activity = int(input("Physical Activity Level (0-100): "))
stress = int(input("Stress Level (1-10): "))
bmi = input("BMI Category (Normal/Overweight/Obese): ").capitalize()
heart_rate = int(input("Heart Rate: "))
steps = int(input("Daily Steps: "))

gender = gender_encoder.transform([gender])[0]
bmi = bmi_encoder.transform([bmi])[0]

new_data = pd.DataFrame([[age, gender, sleep_duration, activity, stress, bmi, heart_rate, steps]],
                        columns=X.columns)

prediction = model.predict(new_data)
result = sleep_encoder.inverse_transform(prediction)

print("\nPredicted Sleep Quality:", result[0])

# ===================== VISUALIZATION =====================
print("\nShowing Sleep Quality Distribution Graph...")
labels = sleep_encoder.inverse_transform(sorted(data["Sleep_Quality"].unique()))
counts = data["Sleep_Quality"].value_counts().sort_index()

plt.bar(labels, counts)
plt.title("Sleep Quality Distribution")
plt.xlabel("Sleep Quality")
plt.ylabel("Count")
plt.show()
