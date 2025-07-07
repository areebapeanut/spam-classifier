import streamlit as st  #for building web app
import pandas as pd #for handling/reading tabular data or excel csv files 
from sklearn.feature_extraction.text import CountVectorizer #converts text to numerical format
from sklearn.linear_model import LogisticRegression #ml model to classify ham or spam
from sklearn.model_selection import train_test_split #splits data into training and testing 
import joblib #used to save and load ur trained model

df = pd.read_excel("hamorspam.xlsx")[['v1', 'v2']]  #reads ur file and selects only 2 cols
df.columns = ['label', 'text'] #renames cols to simpler names ,labels and text
df = df.dropna()  #removes rows with missing date
df['text'] = df['text'].astype(str) #makes sure all text is string and not int/numbers

print(df['label'].value_counts()) #tells how many spam and ham messages in dataset

X_train_text,X_test_text, y_train, y_test = train_test_split(
    df['text'], df['label'], test_size = 0.3 , random_state = 42 , stratify= df ['label'] #30% for testing/ 70% for training 
) #X = email and y = spam/ham   
#random_state = 42 ensures same random split each time (so accuracy doesnt change) (number doesnt matter can be any number)
# stratify = .... ensures both train and test sets have equal ratio of spam and ham    

vectorizer = CountVectorizer() #counts how often each word appears in each email
X_train = vectorizer.fit_transform(X_train_text) #learns vocab from training data , then transforms it 
X_test = vectorizer.transform(X_test_text) # applies same transformation on test data  (return a sparse matrix of numbers)
#fit_transform : the model learns all the words used in email then turns into vectors/numbers, used on training data
#transform : doesnt learn anything new, uses what it learned to turn ur emails to numbers 

model = LogisticRegression() 
model.fit(X_train,y_train) #creates and trains logistic regression model to classify emails

joblib.dump(model, 'spam_model.pkl')   #saves model and vectorizer to disk so next time u dont need to retrain
joblib.dump(vectorizer, 'vectorizer.pkl')  # (pickle) file

model = joblib.load('spam_model.pkl') #loads the saved model and vectorizer when run
vectorizer = joblib.load('vectorizer.pkl')

st.set_page_config(page_title = "spam or ham")   #sets page title and displays description on stream lit 
st.title("Spam vs Ham email classifier")
st.write("Type in an email message and the model will predict if the email is spam or ham")

accuracy = model.score(X_test, y_test)  #returns accuracy or how many predictions were correct  
#accuracy = correct predictions/ total predictions
y_pred = model.predict(X_test) 
correct_predictions = (y_pred == y_test).sum() #counts how many predicitons were exactly right
total_predictions = len(y_test) #total number of test emails

print(f"Model accuracy on test data: {accuracy:.2f}")  #printing accuracy
print(f"{correct_predictions} out of {total_predictions} test emails predicted correctly.")

email = st.text_area("enter email here: ")  #text box where user can paste an email to classify

if st.button("Predict"):     #button used for predicting if spam/ham
    if not email.strip():  #checks if input empty 
        st.warning("Please enter an email")
    else:
        email_vector = vectorizer.transform([email])  #transforms your new email into the same vector format the model was trained on. (array of words count)
        prediction = model.predict(email_vector)   #predicts with trained model , sends the numberized email into your trained model.

        pred_label = str(prediction[0]) #grabs first item in list either spam or ham 
        #y does .predict() return a list , cuz the model is built to predict for many inputs at once, but my model considers whole input as one email thats y we r taking the lists first result [0]
        if pred_label == "spam": 
            st.error("this email is Spam")   #result
        else: 
            st.success("this email is Ham") #result

