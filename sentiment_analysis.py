import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import nltk
# nltk.download("punkt")
# nltk.download("stopwords")
import re
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem.wordnet import WordNetLemmatizer
from textblob import TextBlob
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
import joblib
import streamlit as st
import altair as alt

wnet = WordNetLemmatizer()
data = pd.read_csv("Womens Clothing E-Commerce Reviews.csv")
st.set_page_config(page_title="Natural Learning Processing App", layout="wide")

st.write("""
# A Simple Website for Customers' Reviews Analysis

This application is for predicting customers reviews for a women clothing company 

""")

#st.write(data["Class Name"].unique())
st.selectbox("Select the class name of product", data["Class Name"].dropna().unique())

train_lr = joblib.load(open("C:/Users/Maryam Yusuf/Downloads/Setiment Analysis/trained_model.pkl", "rb"))


def predict_reviews(data):

    results = train_lr.predict(data)
    return results[0]


def predict_prob(data):
    results = train_lr.predict_proba(data)
    return results


def main():
    input_rating = st.number_input("Rate the product", 0, 5)

    with st.form(key="review clf form"):
        raw_text = st.text_area("Type Review Here")
        submit_text = st.form_submit_button(label="Submit")

    if submit_text:
        # cleaned data
        def cleaned_raw_text(data):

            df = data.lower()
            df = re.sub(r"[^\w\s]", "", df)
            df = word_tokenize(df)
            df = [word for word in df if not word in stopwords.words("english")]
            df1 = [wnet.lemmatize(word) for word in df]

            df1 = [" ".join(str(word) for word in df1)]
            df = CountVectorizer().fit_transform(df1)
            df = df.toarray()

            zeros = [i for i in range(15883)]
            a = 0
            df_list = []
            for i in zeros:
                if i in df:
                    if i != 0:
                        df_list.append(i)
                else:
                    df_list.append(0)

            df = pd.DataFrame([df_list])

             # get polarity
            polarity_score = [TextBlob(df).sentiment.polarity for df in df1]
            df_score = pd.DataFrame(data=polarity_score, columns=["pol_score"])

            # get rating to dataframe
            df_rating = pd.DataFrame({"Rating": [input_rating]})

            # get all the dataframe together
            global cleaned_text
            cleaned_text = pd.concat([df, df_score, df_rating], axis=1)
            cleaned_text["negative"] = cleaned_text["pol_score"].apply(lambda x: 1 if x < 0 else 0)
            cleaned_text["neutral"]  = cleaned_text["pol_score"].apply(lambda x: 1 if x == 0 else 0)
            cleaned_text["positive"] = cleaned_text["pol_score"].apply(lambda x: 1 if x > 0 else 0)
            return cleaned_text

        cleaned_raw_text(raw_text)


        # Creating column
        col1, col2 = st.columns(2)

        # Apply fxn

        prediction = predict_reviews(cleaned_text)
        probability = predict_prob(cleaned_text)

        with col1:
            st.success("Original Text")
            st.write(raw_text)

            st.success("Prediction")
            if prediction == 1:
                pred = "Good Review"
            else:
                pred = "Bad Review"
            st.write(pred)

        with col2:
            st.success("Prediction Probability")
            prob_df = pd.DataFrame(probability, columns=["Bad", "Good"])
            proba_df_clean = prob_df.T.reset_index()
            proba_df_clean.columns = ["Review", "Probability"]

            fig = alt.Chart(proba_df_clean).mark_bar().encode(x="Review", y="Probability", color="Review")
            st.altair_chart(fig, use_container_width = True)

if __name__ == "__main__":
    main()

