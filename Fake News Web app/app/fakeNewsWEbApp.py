import pickle 
import streamlit as st

fake_news_vectorizer = pickle.load(open('C:/Users/Mahdi/OneDrive/Desktop/Projects/TFIdfVectorizer Fake News.sav', 'rb'))

loaded_model = pickle.load(open('C:/Users/Mahdi/OneDrive/Desktop/Projects/FakeNews.sav', 'rb'))

def fake_news(input_transform):
    input_transform = fake_news_vectorizer.transform(input_transform)
    reshaped_input = input_transform.reshape(1, -1)
    prediction = loaded_model.predict(reshaped_input)
    if(prediction[0] == 1):
        return ('Fake news')
    else:
        return ('Real news')
    
def main():
    st.title("Fake News prediction WEB APP")
    content = st.text_input("Paste the title and Authon name")
    
    test = ''
    if(st.button('Predict')):
        test = fake_news([content])
        st.success(test)

if __name__ == '__main__':
    main()

# import pickle 
# import streamlit as st

# # Load vectorizer and model
# fake_news_vectorizer = pickle.load(open('C:/Users/Mahdi/OneDrive/Desktop/Projects/TFIdfVectorizer Fake News.sav', 'rb'))
# loaded_model = pickle.load(open('C:/Users/Mahdi/OneDrive/Desktop/Projects/FakeNews.sav', 'rb'))

# def fake_news(input_text):
#     # Transform input text (do NOT use fit_transform)
#     input_features = fake_news_vectorizer.transform(input_text)
#     # Predict directly using sparse matrix
#     prediction = loaded_model.predict(input_features)
#     return 'Fake news' if prediction[0] == 1 else 'Real news'
    
# def main():
#     st.title("Fake News Prediction Web App")
#     content = st.text_input("Paste the title and Author name")
    
#     if st.button('Predict'):
#         test = fake_news([content])
#         st.success(test)

# if __name__ == '__main__':
#     main()
