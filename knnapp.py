import streamlit as st

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsRegressor

# ---------------- TITLE ----------------

st.title("KNN Iris Regression App")

st.write("K-Nearest Neighbors Regression")

# ---------------- LOAD DATA ----------------

data = load_iris()

X = data.data[:, :3]
y = data.data[:, 3]

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

model = KNeighborsRegressor(n_neighbors=5)

model.fit(X_train, y_train)

# ---------------- USER INPUT ----------------

st.subheader("Enter Flower Measurements")

with st.form("prediction_form"):

    sepal_length = st.number_input(
        "Sepal Length",
        min_value=0,
        max_value=10,
        step=1,
        format="%d"
    )

    sepal_width = st.number_input(
        "Sepal Width",
        min_value=0,
        max_value=10,
        step=1,
        format="%d"
    )

    petal_length = st.number_input(
        "Petal Length",
        min_value=0,
        max_value=10,
        step=1,
        format="%d"
    )

    predict_button = st.form_submit_button(
        "Predict Petal Width"
    )

# ---------------- PREDICTION ----------------

if predict_button:

    input_data = scaler.transform([[
        sepal_length,
        sepal_width,
        petal_length
    ]])

    prediction = model.predict(input_data)

    st.success(
        f"Predicted Petal Width: {prediction[0]:.2f}"
    )