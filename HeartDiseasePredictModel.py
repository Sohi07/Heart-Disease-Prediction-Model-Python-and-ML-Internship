import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import roc_auc_score, accuracy_score, roc_curve, auc, confusion_matrix, precision_recall_curve
import matplotlib.pyplot as plt
import seaborn as sns


data = pd.read_csv('heart.csv')
data['Sex'] = data['Sex'].map({'M': 1, 'F': 0})
data['ChestPainType'] = data['ChestPainType'].map({'ATA': 0, 'NAP': 1, 'ASY': 2, 'TA': 3})
data['RestingECG'] = data['RestingECG'].map({'Normal': 0, 'ST': 1, 'LVH': 2})
data['ExerciseAngina'] = data['ExerciseAngina'].map({'N': 0, 'Y': 1})
data['ST_Slope'] = data['ST_Slope'].map({'Up': 0, 'Flat': 1, 'Down': 2})


X = data.drop('HeartDisease', axis=1)
y = data['HeartDisease']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)


models = {
    'Logistic Regression': LogisticRegression(),
    'Decision Tree': DecisionTreeClassifier(random_state=42),
    'Random Forest': RandomForestClassifier(random_state=42),
    'Naive Bayes': GaussianNB()
}

results = {}
model_objects = {}

for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)[:, 1]
    
    accuracy = accuracy_score(y_test, y_pred)
    auc_score = roc_auc_score(y_test, y_pred_proba)
    
    results[name] = {'Accuracy': accuracy, 'AUC': auc_score}
    model_objects[name] = model


best_model_name = max(results, key=lambda x: (results[x]['Accuracy'], results[x]['AUC']))
best_model = model_objects[best_model_name]

print("Model Performance:")
for name, metrics in results.items():
    print(f"{name} - Accuracy: {metrics['Accuracy']:.2f}, AUC: {metrics['AUC']:.2f}")

print(f"\nThe best model is: {best_model_name}")


def get_user_input():
    age = int(input("Enter Age: "))
    sex = int(input("Enter Sex (1 for Male, 0 for Female): "))
    chest_pain_type = int(input("Enter Chest Pain Type (0: ATA, 1: NAP, 2: ASY, 3: TA): "))
    resting_ecg = int(input("Enter Resting ECG (0: Normal, 1: ST, 2: LVH): "))
    exercise_angina = int(input("Enter Exercise Angina (0: No, 1: Yes): "))
    st_slope = int(input("Enter ST Slope (0: Up, 1: Flat, 2: Down): "))
    
    new_data = pd.DataFrame({
        'Age': [age],
        'Sex': [sex],
        'ChestPainType': [chest_pain_type],
        'RestingECG': [resting_ecg],
        'ExerciseAngina': [exercise_angina],
        'ST_Slope': [st_slope]
    })
    
    return pd.DataFrame(new_data, columns=X.columns)

def make_prediction(model, scaler, new_data):
    new_data_scaled = scaler.transform(new_data)
    prediction = model.predict(new_data_scaled)
    prediction_proba = model.predict_proba(new_data_scaled)[:, 1]
    
    if prediction[0] == 1:
        print("The model predicts that the person has heart disease.")
    else:
        print("The model predicts that the person does not have heart disease.")
    
    print(f"Probability of having heart disease: {prediction_proba[0]:.2f}")

new_user_data = get_user_input()
make_prediction(best_model, scaler, new_user_data)



fpr, tpr, _ = roc_curve(y_test, best_model.predict_proba(X_test)[:, 1])
roc_auc = auc(fpr, tpr)


plt.figure()
plt.plot(fpr, tpr, color='blue', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], color='gray', linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) Curve')
plt.legend(loc="lower right")
plt.show()

precision, recall, _ = precision_recall_curve(y_test, best_model.predict_proba(X_test)[:, 1])
plt.figure()
plt.plot(recall, precision, color='purple', lw=2)
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('Precision-Recall Curve')
plt.show()


corr_matrix = data.corr()
plt.figure(figsize=(12, 8))
sns.heatmap(corr_matrix, annot=True, fmt=".2f", cmap='coolwarm', square=True, cbar_kws={"shrink": .8})
plt.title('Correlation Matrix of Heart Disease Dataset')
plt.show()


plt.figure(figsize=(10, 6))
plt.hist(data['Age'], bins=20, color='blue', edgecolor='black')
plt.title('Histogram of Age')
plt.xlabel('Age')
plt.ylabel('Frequency')
plt.grid(True)
plt.show()


cm = confusion_matrix(y_test, best_model.predict(X_test))
plt.figure(figsize=(8,6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.title('Confusion Matrix')
plt.ylabel('Actual')
plt.xlabel('Predicted')
plt.show()


if hasattr(best_model, 'feature_importances_'):
    importances = best_model.feature_importances_
    features = X.columns
    feature_importance_df = pd.DataFrame({'Feature': features, 'Importance': importances})
    feature_importance_df = feature_importance_df.sort_values(by='Importance', ascending=False)

   
    plt.figure(figsize=(10, 6))
    sns.barplot(x='Importance', y='Feature', data=feature_importance_df, palette='viridis')
    plt.title('Feature Importance')
    plt.show()
