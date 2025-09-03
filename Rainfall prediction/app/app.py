import numpy as np
import pickle
import streamlit as st

model = pickle.load(open('C:/Users/Mahdi/OneDrive/Desktop/Projects/Rainfall prediction/model/Rainfall.pkl', 'rb'))

def rainfall(input):
    input_arr = np.asarray(input)
    reshaped = input_arr.reshape(1, -1) 
    prediction = model.predict(reshaped)
    if prediction[0] == 0:
        return 'No Rainfall'
    
    return 'Rainfall'

def main():
    st.title("Rainfall Prediction.")
    features = ['pressure', 'dewpoint', 'humidity', 'cloud', 'sunshine', 'winddirection', 'windspeed']
    user_inputs = []
    for feature in features:
        value = st.number_input(f"Enter {feature}")
        user_inputs.append(value)

    result = ''
    if(st.button('Predict')):
        result = rainfall(user_inputs)
        st.success(result)

if __name__ == '__main__':
    main()