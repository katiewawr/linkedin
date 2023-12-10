import pandas as pd
s = pd.read_csv("social_media_usage.csv")

import numpy as np
def clean_sm (x):
    clean = np.where(x == 1, 1, 0)
    return clean

ss = pd.DataFrame({
    "income" : np.where(s["income"] > 9, np.nan, s["income"]),
    "education" : np.where(s["educ2"] > 8, np.nan, s["educ2"]),
    "parent" : clean_sm(s["par"]),
    "married" : clean_sm(s["marital"]),
    "female" : np.where(s["gender"] == 2, 1, 0),
    "age" : np.where(s["age"] > 98, np.nan, s["age"]),
    "sm_li" : clean_sm(s["web1h"])})

# drop any missing values
ss = ss.dropna()

# create a target vector (y) and feature set (X)
y = ss["sm_li"]
X = ss[["income", "education", "parent", "married", "female", "age"]]

# split the data into training and test sets
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X,
                                                    y,
                                                    stratify = y,
                                                    test_size = 0.2,
                                                    random_state = 1334)


# fit the model with the training data
from sklearn.linear_model import LogisticRegression
lr = LogisticRegression(class_weight = "balanced")
lr.fit(X_train, y_train)

import streamlit as st
st.title("Do you use LinkedIn?")

income = st.selectbox("What is your household income?",
                     ("Less than $10,000", 
                      "$10,000 - 20,000",
                      "$20,000 - 30,000",
                      "$30,000 - 40,000",
                      "$40,000 - 50,000",
                      "$50,000 - 75,000",
                      "$75,000 - 100,000",
                      "$100,000 - 150,000",
                      "$150,000+",
                      "Decline to Answer"), index = None)

if income == "Less than $10,000":
    income = 1
elif income == "$10,000 - 20,000":
    income = 2
elif income == "$20,000 - 30,000":
    income = 3
elif income == "$30,000 - 40,000":
    income = 4
elif income == "$40,000 - 50,000":
    income = 5
elif income == "$50,000 - 75,000":
    income = 6
elif income == "$75,000 - 100,000":
    income = 7
elif income == "$100,000 - 150,000":
    income = 8
else:
    income = 9


education = st.selectbox("What is your highest level of education?",
                        ("Less than High School", 
                         "High School Incomplete",
                         "High School Diploma or GED Certificate",
                         "Some College (no degree)",
                         "Two-year Associate Degree",
                         "Four-year Bachelor's Degree",
                         "Some Postgraduate Schooling (no degree)",
                         "Postgraduate Degree",
                         "Decline to Answer"), index = None)

if education == "Less than High School":
    education = 1
elif education == "High School Incomplete":
    education = 2
elif education == "High School Diploma or GED Certificate":
    education = 3
elif education == "Some College (no degree)":
    education = 4
elif education == "Two-year Associate Degree":
    education = 5
elif education == "Four-year Bachelor's Degree":
    education = 6
elif education == "Some Postgraduate Schooling (no degree)":
    education = 7
else:
    education = 8

parent = st.radio("Are you a parent?", ["Yes", "No", "Decline to Answer"], index = None)

if parent == "Yes":
    parent = 1
else:
    parent = 0

married = st.radio("Marital Status", ["Married", "Living with a Partner", "Divorced", "Separated", "Widowed", "Single", "Decline to Answer"], index = None)

if married == "Married":
    married = 1
else:
    married = 0

female = st.radio("Gender", ["Male", "Female", "Decline to Answer"], index = None)

if female == "Female":
    female = 1
else:
    female = 0

age = st.number_input("How old are you?", 0, 100)

person = [income, education, parent, married, female, age]

probs = lr.predict_proba([person])
predicted_class = lr.predict([person]) 
if predicted_class == 1:
   predicted = f"Yes, there is a {(probs[0][1]*100).round(1)}% chance you are a LinkedIn user!"
else:
   predicted = f"There is a {(probs[0][1]*100).round(1)}% chance you do not have a LinkedIn account."

if st.button("Get Results Here!!!"):
    st.write(predicted)

if predicted == f"There is a {(probs[0][1]*100).round(1)}% chance you do not have a LinkedIn account.":
    st.link_button("Join now!", "https://www.linkedin.com/signup?trk=guest_homepage-basic_nav-header-join")
else:
    st.write("")