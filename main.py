import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.pipeline import make_pipeline # ייבוא חדש וחשוב!
import joblib

# 1. Load Dataset
df = pd.read_csv('parkinsons.csv')

# 2. Select Features
# הקפד שהשמות כאן יהיו זהים לקובץ הקונפיגורציה
features = ['PPE', 'DFA']
X = df[features]
y = df['status']

# 3. Split Data
# אנחנו מפצלים את הנתונים הגולמיים (לפני נרמול)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 4. Build Pipeline (Scaler + Model)
# זה התיקון הגדול: אנחנו יוצרים "צינור" שקודם מנרמל ואז מפעיל את ה-SVM
# ככה המודל שנשמור יכיל בתוכו את הוראות הנרמול
model = make_pipeline(MinMaxScaler(feature_range=(-1, 1)), SVC(kernel='rbf', random_state=42))

# 5. Train the Pipeline
# שים לב: אנחנו מאמנים על X_train הגולמי, הפייפליין דואג לנרמל לבד
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
