import numpy as np
import pickle
import streamlit as st

loaded_model = pickle.load(open('C:/Users/Mahdi/OneDrive/Desktop/Javascript New/House price prediction/HousingModel.sav', 'rb'))

def house_price(input):
    inp_to_array = np.asarray(input)
    reshaped_arr = inp_to_array.reshape(1, -1)
    prediction = loaded_model.predict(reshaped_arr)
    return (f"The price of the house is ${round(prediction[0], 4)}k")

def main():
    st.title("House price prediction web app")
    features= ['crim','zn','indus','chas','nox','rm','age','dis','rad','tax','ptratio','B','lstat']

    user_inp = []
    for feature in features:
        value = st.number_input(f'Enter {feature}', value= 0.0, format= '%.4f')
        user_inp.append(value)

    result = ''
    if(st.button('Check')):
        result = house_price(user_inp)
        st.success(result)

if __name__ == '__main__':
    main()