import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.model_selection import train_test_split

# Page Config
st.set_page_config(page_title="Quinlan 1986 Research Implementation", layout="wide")

st.title("🌳 Research Implementation: Induction of Decision Trees")
st.subheader("Based on J.R. Quinlan (1986)")

# Sidebar for interactivity
st.sidebar.header("Model Controls")
tree_depth = st.sidebar.slider("Maximum Tree Depth", 1, 10, 3)
split_ratio = st.sidebar.slider("Test Split %", 10, 50, 20) / 100

# Load Data
iris = load_iris()
X = pd.DataFrame(iris.data, columns=iris.feature_names)
y = iris.target
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=split_ratio, random_state=42)

# Model Training
# Criterion is hardcoded to 'entropy' to respect the original paper's logic
model = DecisionTreeClassifier(criterion='entropy', max_depth=tree_depth, random_state=42)
model.fit(X_train, y_train)
acc = model.score(X_test, y_test)

# Dashboard Layout
col1, col2 = st.columns([1, 2])

with col1:
    st.write("### Model Performance")
    st.metric("Accuracy", f"{acc*100:.2f}%")
    st.write("**Methodology:** ID3 Algorithm")
    st.write("**Metric:** Information Gain (Entropy)")

    st.write("### Feature Importance")
    importance = pd.Series(model.feature_importances_, index=iris.feature_names)
    st.bar_chart(importance)

with col2:
    st.write("### Tree Visualization")
    fig, ax = plt.subplots(figsize=(10, 6))
    plot_tree(model,
              feature_names=iris.feature_names,
              class_names=iris.target_names,
              filled=True,
              rounded=True,
              ax=ax)
    st.pyplot(fig)

st.divider()
st.markdown("*Developed for Introduction to Machine Learning - 8th Semester Assignment*")
