from sklearn.model_selection import train_test_split
from sklearn.datasets import make_classification
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, roc_auc_score, f1_score, roc_curve
import matplotlib.pyplot as plt

def classification(X, y, model):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    model.fit(X_train_scaled, y_train)
    y_pred = model.predict(X_test_scaled)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Accuracy: {accuracy * 100:.2f}%")
    roc_auc = roc_auc_score(y_test, y_pred)
    print(f"AUC-ROC: {roc_auc:.2f}")
    f1 = f1_score(y_test, y_pred, pos_label = 3)
    print(f"F1 Score: {f1:.2f}")
    fpr, tpr, thresholds = roc_curve(y_test, y_pred, pos_label = 3)
    
    plt.figure()
    plt.plot(fpr, tpr, label='SVM (area = %0.2f)' % roc_auc)
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic')
    plt.legend(loc='lower right')
    plt.show()

def invert_dict(d):
    return {value: key for key, value in d.items()}