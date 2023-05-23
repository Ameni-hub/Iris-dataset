from sklearn import datasets
from sklearn.tree import DecisionTreeClassifier
import streamlit as st
from sklearn.model_selection import train_test_split
iris = datasets.load_iris()
print(iris.target)
y = iris.target
X = iris.data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

dtree = DecisionTreeClassifier(criterion='gini', splitter='best', max_leaf_nodes=3, min_samples_leaf=2,
                                   max_depth=None)

classifier = DecisionTreeClassifier(max_depth=3)
classifier.fit(X_train, y_train)
Y_pred = classifier.predict(X_test)
st.title('Iris dataset')
st.header('Prediction of the flower type')
X = iris.data
import numpy as np
# Find the minimum value in X
min_val = np.min(X, axis = 0)
max_val = np.max(X, axis = 0)
mean_val = np.mean(X,axis = 0)
sepal_length = st.slider("Sepal length", float(min_val[0]),float(max_val[0]), float(mean_val[0]))
sepal_width = st.slider("Sepal width", float(min_val[1]), float(max_val[1]),
                            float(mean_val[1]))
petal_length = st.slider("Petal length", float(min_val[2]), float(max_val[2]),
                             float(mean_val[2]))
petal_width = st.slider("Petal width", float(min_val[3]), float(max_val[3]),
                            float(mean_val[3]))
st.text('Selected: {}'.format(sepal_length,sepal_width,petal_length,petal_width))
Y_pred = classifier.predict(X_test)


target_names = iris.target_names


def predict_iris_flower(input_values):
    # Perform the prediction
    predicted_label = classifier.predict([input_values])[0]  # Replace 'classifier' with the appropriate classifier variable
    predicted_flower = target_names[predicted_label]

    return predicted_flower

def main():
    st.title("Iris Flower Prediction")
    sepal_length = st.number_input("Enter sepal length (cm)")
    sepal_width = st.number_input("Enter sepal width (cm)")
    petal_length = st.number_input("Enter petal length (cm)")
    petal_width = st.number_input("Enter petal width (cm)")

    if st.button("Predict"):
        input_values = [sepal_length, sepal_width, petal_length, petal_width]
        predicted_flower = predict_iris_flower(input_values)
        st.write("Predicted flower type:", predicted_flower)

if __name__ == '__main__':
    main()
