import numpy as np
import pickle
import streamlit as st

model = pickle.load(open('C:/Users/Mahdi/OneDrive/Desktop/Javascript New/Flower classification/model/flower.pkl', 'rb'))

def classifier(input):
    input_to_arr = np.asarray(input)
    reshaped_inp = input_to_arr.reshape(1, -1)
    prediction = model.predict(reshaped_inp)

    if prediction[0] == 0:
        return ('Iris-setosa')
    
    if prediction[0] == 1:
        return ('Iris-versicolor')

    return ('Iris-virginica')

def main():
    # SepalLengthCm,SepalWidthCm,PetalLengthCm,PetalWidthCm
    st.title("Flower Species classification")
    SepalLengthCm = st.number_input('Sepal Length Cm')
    SepalWidthCm = st.number_input('Sepal Width Cm')
    PetalLengthCm = st.number_input('Petal Length Cm')
    PetalWidthCm = st.number_input('Petal Width Cm')

    classify = ''

    if st.button('Classify'):
        classify = classifier([SepalLengthCm,SepalWidthCm,PetalLengthCm,PetalWidthCm])
        st.success(classify)

if __name__ == "__main__":
    main()