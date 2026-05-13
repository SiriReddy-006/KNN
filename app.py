import streamlit as st

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from sklearn.neighbors import KNeighborsClassifier

# ---------------- TITLE ----------------

st.title("KNN Iris Flower Prediction App")

st.write("K-Nearest Neighbors Classification")

# ---------------- LOAD DATA ----------------

data = load_iris()

X = data.data
y = data.target

# ---------------- SPLIT DATA ----------------

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# ---------------- FEATURE SCALING ----------------

scaler = StandardScaler()

X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# ---------------- TRAIN MODEL ----------------

model = KNeighborsClassifier(n_neighbors=3)

model.fit(X_train, y_train)

# ---------------- USER INPUT ----------------

st.subheader("Enter Flower Measurements")

with st.form("prediction_form"):

    sepal_length = st.slider(
        "Sepal Length",
        4.0,
        8.0,
        5.1
    )

    sepal_width = st.slider(
        "Sepal Width",
        2.0,
        5.0,
        3.5
    )

    petal_length = st.slider(
        "Petal Length",
        1.0,
        7.0,
        1.4
    )

    petal_width = st.slider(
        "Petal Width",
        0.1,
        3.0,
        0.2
    )

    submit_button = st.form_submit_button("Predict Flower")

# ---------------- PREDICTION ----------------

if submit_button:

    input_data = scaler.transform([[
        sepal_length,
        sepal_width,
        petal_length,
        petal_width
    ]])

    prediction = model.predict(input_data)

    flower_names = [
        "Setosa",
        "Versicolor",
        "Virginica"
    ]

    st.success(
        f"Predicted Flower: {flower_names[prediction[0]]}"
    )