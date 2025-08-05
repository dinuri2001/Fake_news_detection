import streamlit as st
import pickle
import pandas as pd

# Load models and vectorizer
LR = pickle.load(open("lr_model.pkl", "rb"))
DT = pickle.load(open("dt_model.pkl", "rb"))
GBC = pickle.load(open("gbc_model.pkl", "rb"))
RFC = pickle.load(open("rfc_model.pkl", "rb"))
vectorization = pickle.load(open("vectorizer.pkl", "rb"))

# Define wordopt and output_label
def output_lable(n):
    return "Not a Fake News" if n == 0 else " Fake News"


def wordopt(text):
    # define your preprocessing steps here
    return text.lower()  # minimal example

def manual_testing(news):
    news_df = pd.DataFrame({"text": [news]})
    news_df["text"] = news_df["text"].apply(wordopt)
    news_vector = vectorization.transform(news_df["text"])

    preds = {
        "Logistic Regression": output_lable(LR.predict(news_vector)[0]),
        "Decision Tree": output_lable(DT.predict(news_vector)[0]),
        "Gradient Boosting": output_lable(GBC.predict(news_vector)[0]),
        "Random Forest": output_lable(RFC.predict(news_vector)[0])
    }
    
    return preds

# Streamlit UI
st.title("📰 Fake News Detection App")
news_input = st.text_area("Enter the News Article", height=200)

if st.button("Check"):
    if news_input.strip() == "":
        st.warning("Please enter some text.")
    else:
        with st.spinner("Analyzing..."):
            result = manual_testing(news_input)
            st.success("Prediction complete!")

            for model, prediction in result.items():
                st.write(f"**{model}:** {prediction}")
