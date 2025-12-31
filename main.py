import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
import joblib

# 1. Load Dataset
df = pd.read_csv('parkinsons.csv')

# 2. Select Features (Based on the paper)
features = ['PPE', 'DFA']
X = df[features]
y = df['status']

# 3. Scale Data
scaler = MinMaxScaler(feature_range=(0, 1))
X_scaled = scaler.fit_transform(X)
X = pd.DataFrame(X_scaled, columns=features)

# 4. Split Data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 5. Choose Model (SVM)
model = SVC(kernel='rbf', random_state=42)
model.fit(X_train, y_train)

# 6. Test Accuracy
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Model Accuracy: {accuracy:.2f}")

# 7. Save Model
if accuracy >= 0.8:
    joblib.dump(model, 'my_model.joblib')
    print("Model saved to my_model.joblib")
else:
    print("Accuracy too low, model not saved.")
