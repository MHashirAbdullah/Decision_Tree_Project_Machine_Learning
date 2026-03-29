import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import accuracy_score, classification_report

def run_implementation():
    print("--- Loading Iris Dataset ---")
    iris = load_iris()
    X = pd.DataFrame(iris.data, columns=iris.feature_names)
    y = iris.target

    # 80/20 Split as per standard academic practice
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # INITIALIZING ID3 LOGIC
    # We use 'entropy' to match Quinlan's 1986 Information Gain theory
    model = DecisionTreeClassifier(criterion='entropy', max_depth=4, random_state=42)
    model.fit(X_train, y_train)

    # EVALUATION
    y_pred = model.predict(X_test)
    print(f"Model Accuracy: {accuracy_score(y_test, y_pred) * 100:.2f}%")
    print("\nDetailed Classification Report:")
    print(classification_report(y_test, y_pred, target_names=iris.target_names))

    # VISUALIZATION
    plt.figure(figsize=(12, 8))
    plot_tree(model,
              feature_names=iris.feature_names,
              class_names=iris.target_names,
              filled=True,
              rounded=True)
    plt.title("Decision Tree Induction (Quinlan 1986 Implementation)")
    plt.show()

if __name__ == "__main__":
    run_implementation()
