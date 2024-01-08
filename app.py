#importing all the important libraries
from ast import Break
import streamlit as st
import sqlite3
import json
import requests
from PIL import Image
import os
import numpy as np
import pandas as pd
import plotly.express as px
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from streamlit_chat import message
#from transformers import BlenderbotTokenizer
#from transformers import BlenderbotForConditionalGeneration
import time
from streamlit_lottie import st_lottie
#building the sidebar of the web app which will help us navigate through the different sections of the entire application
#Home Page 
dictionary = {'Home': ['Home','RYLO Chatbot','Routine Developer'], 'RYLO Health Analyser': ['Covid-19', 'Diabetes' , 'Heart Disease'] ,'Plots Page': ['Plots'], 'RYLO Lifestyle Analyser':['lifestyle Tracker']}

conn = sqlite3.connect('example.db')
c = conn.cursor()

def load_lottiefile(filepath: str):
    with open(filepath, "r") as f:
        return json.load(f)


def load_lottieurl(url: str):
    r = requests.get(url)
    if r.status_code != 200:
        return None
    return r.json()

def create_table():
    c.execute('CREATE TABLE IF NOT EXISTS users(username TEXT, password TEXT)')
    c.execute('''CREATE TABLE IF NOT EXISTS example_table
                 (id INTEGER PRIMARY KEY, name TEXT, role TEXT , age INTEGER )''')


def insert_data(name, role, age):
    c.execute('''INSERT INTO example_table (name, role , age) VALUES (?, ?, ?)''', (name, role ,age))
    conn.commit()
    
def get_name(name):
    c.execute('''SELECT * FROM example_table WHERE name = ?''', (name,))
    data = c.fetchone()
    return data

def get_age(name):
    c.execute('''SELECT age FROM example_table WHERE name = ?''', (name,))
    agedata = c.fetchone()
    if agedata :
        return agedata
    else :
        return None
    
def get_role(name):
    c.execute('''SELECT role FROM example_table WHERE name = ?''', (name,))
    roledata = c.fetchone()
    if roledata :
        return roledata
    else :
        return None
    
def add_user(username, password):
    c.execute('INSERT INTO users (username, password) VALUES (?, ?)', (username, password))
    conn.commit()

# Define a function to authenticate the user against the database
def authenticate(username, password):
    c.execute('SELECT * FROM users WHERE username = ? AND password = ?', (username, password))
    result = c.fetchone()
    if result:
        return True
    return False

# Define the login page
def login():
    st.title("Login")
    username = st.text_input("Username" , key = "login_username")
    password = st.text_input("Password", type="password" ,key="login_password")
    if st.button("Login", key="login_button"):
        if authenticate(username, password):
            st.success("Logged in as {}".format(username))
            st.session_state.authenticated = True
            st.session_state.username = username
            st.experimental_rerun()
        else:
            st.error("Invalid username or password")

# Define the registration page
def register():
    st.title("Register")
    username = st.text_input("Username", key="register_username")
    password = st.text_input("Password", type="password", key="register_password")
    confirm_password = st.text_input("Confirm Password", type="password", key="confirm_password")
    if st.button("Register", key="register_button"):
        if password == confirm_password:
            add_user(username, password)
            st.success("Account created for {}".format(username))
            st.session_state.authenticated = True
            st.session_state.username = username
            st.experimental_rerun()
        else:
            st.error("Passwords do not match")

#displays all the available disease prediction options in the web app
def home():
    Navigation = st.sidebar.selectbox("Navigation Menu:", sorted(dictionary.keys()))
    rad = st.sidebar.radio("Choose any page:", sorted(dictionary[Navigation]))
    if rad=="Home":

        # Title and header
        st.title("🤖RYLO LIFESTYLE AI")
        st.subheader("Raising Youth's Lifestyle Optimizer")
        st.image("images/Lifestyle-home.png")

        # Input fields
        user = st.text_input('Enter your Name')
        role = st.selectbox("Select your Role", ['School Student', 'College Student', 'Employee', 'Senior Citizen'])
        age = st.slider("Select your age", 10, 70)

        # Photo upload
        photo = st.file_uploader("Upload a profile photo", type=["jpg", "jpeg", "png"])
        if photo:
            st.success("Photo uploaded successfully!")

        # Submit button
        if st.button('Submit'):
            insert_data(user, role, age)
            data = get_name(user)
            st.write(f"Hi {data[1]}, welcome to RYLO Lifestyle AI!")
            roledata = get_role(user)
            st.write(f"Your role is {roledata[0]}")
            agedata = get_age(user)
            st.write(f"Your age is {agedata[0]}")

            # Animated welcome message
            lottie_hello = load_lottieurl("https://assets1.lottiefiles.com/packages/lf20_puciaact.json")
            st_lottie(
                lottie_hello,
                speed=1,
                reverse=False,
                loop=True,
                quality="high",
                height=None,
                width=None,
                key=None,
            )

            # Introduction and instructions
            st.subheader("Welcome to the world of RYLO")
            st.write("RYLO is an Lifestyle Optimizing AI which helps you to engage on your Lifestyle and analyze your health conditions.")
            st.write("Please navigate through our features below.")

        # Footer
        st.write("---")
        st.write("Developed by Kannan S , Panneer Subramanian M , Rijoe T S")

        
            
    if rad == "Routine Developer":
        st.title("🤖RYLO Routine Developer")
        lottie_task = load_lottieurl("https://assets8.lottiefiles.com/packages/lf20_zpo7qo16.json")

        st_lottie(
            lottie_task,
            speed=1,
            reverse=False,
            loop=True,
            quality="low", # medium ; high
            height=None,
            width=None,
            key=None,
        )
        task_lists = {
                "Kids": {
                    "Go for a walk": "walk.png",
                    "Brush your teeth": "brush.png",
                    "Play a game": "game.png",
                    "Read a book": "book.png",
                    "Draw a picture": "draw.png"
                },
                "Teens": {
                    "Go for a run": "run.png",
                    "Study for an hour": "study.png",
                    "Have lunch": "lunch.png",
                    "Take a nap": "nap.png",
                    "Go grocery shopping": "grocery.png"
                },
                "Adults": {
                    "Exercise for 30 minutes": "exercise.png",
                    "Work on a project": "project.png",
                    "Attend a meeting": "meeting.png",
                    "Take a break": "break.png",
                    "Cook dinner": "cook.png"
                }
            }

            # Display age selection dropdown
        age_group = st.selectbox("Select your age group", ["Kids", "Teens", "Adults"])

            # Load images from the 'images' folder
        image_folder = os.path.join(os.getcwd(), 'images')

            # Display tasks and checkboxes
        for i, task in enumerate(task_lists[age_group].keys()):
                task_status = st.empty()
                status_image = st.empty()
                
                task_status.subheader(f"{task}")
                task_image = Image.open(os.path.join(image_folder, task_lists[age_group][task]))
                status_image.image(task_image, use_column_width=True)
                
                if st.checkbox("Completed", key=f"completed_{i}"):
                    task_status.subheader(f"You have completed {task} successfully! ✅")
                elif st.checkbox("Skip", key=f"skip_{i}"):
                    task_status.subheader(f"You have skipped {task}. ❌")
            
                
        lottie_comp = load_lottieurl("https://assets6.lottiefiles.com/packages/lf20_1oky66dx.json")

        st_lottie(
                    lottie_comp,
                    speed=1,
                    reverse=False,
                    loop=True,
                    quality="low", # medium ; high
                    height=None,
                    width=None,
                    key=None,
                )

        
            
    if rad == "RYLO Chatbot":
        
        st.title("🤖RYLO Chat Bot")
        
        lottie_bot = load_lottieurl("https://assets7.lottiefiles.com/private_files/lf30_lhLisE.json")

        st_lottie(
            lottie_bot,
            speed=1,
            reverse=False,
            loop=True,
            quality="low", # medium ; high
            height=200,
            width=200,
            key=None,
            )
        
        
        user = st.text_input("Talk to RYLO Chatbot🎙️")
        message_man =[]
        message_bot =[]
        man = [
            'hi', 'Good Morning', 'Good Afternoon', 'bye', 'I am kannan, can you give me some suggestion on how to stay Healthy',
            'What else is important to stay healthy?', 'How often can I eat junk food?',
            'What do you think is the right balance to keep fit?', 'Can you explain the importance of Physical Exercise?',
            'Why do i feel nauseous all the time for no reason?', 'What can I do about the stress in my life?',
            'Thanks for your suggestions'
        ]
        bot = [
            'Hi My name is RYLO. I am here to help you to improve your lifestyle by conveying suggestion',
            'Good Morning RYLO is Ready', 'Good Afternon RYLO On Charge', 'Bye bye Take care my friend',
            'You need to make sure that you eat the right foods.',
            'You need to follow the suggested routines.',
            'Junk food is not healthy at all. You can have it once in a while, not daily.',
            'Get enough sleep, Active Lifestyle, Balanced diet, Exercise daily, Drink more water.',
            'It can make you feel happy, help you to weight loss, good for muscles and bones.',
            'Nausea is not a disease itself, but can be a symptom of many disorders related to the digestive system.',
            'You have two choices: Remove the stressor, or learn how to react to it in a healthier way',
            'Okay. Have a great day!!. Lets keep in touch.'
        ]
        
        i=0
        for question in man:
            if user == question:
                message_man.append(man[i])
                message_bot.append(bot[i])      
            i=i+1
        z=0
        for m in message_man:
            message(message_man[z],is_user=True,avatar_style="personas")
            message(message_bot[z],is_user=False)
            z=z+1
            
    if rad == "lifestyle Tracker":
        st.title("🤖RYLO Lifestyle Analyser")
        lottie_track = load_lottieurl("https://assets10.lottiefiles.com/packages/lf20_ayopewsc.json")

        st_lottie(
            lottie_track,
            speed=1,
            reverse=False,
            loop=True,
            quality="low", # medium ; high
            height=None,
            width=None,
            key=None,
            )
        
        st.subheader("Answer the following questions to examine your lifestyle")
        

        # Define questions and corresponding emojis and types
        questions = {
            "How many minutes of exercise do you get each day?": ("🏋️‍♀️", "number"),
            "How many hours of sleep do you get each night?": ("💤", "number"),
            "Do you smoke?": ("🚬", "binary"),
            "How often do you drink alcohol?": ("🍸", "select"),
            "How many servings of fruits and vegetables do you eat each day?": ("🍎🥦", "number"),
            "How often do you eat fast food or processed snacks?": ("🍔🍟", "select"),
            "How much water do you drink per day?": ("💧", "number"),
            "How often do you practice mindfulness or meditation?": ("🧘‍♀️", "binary"),
            "Do you have regular medical check-ups?": ("👩‍⚕️", "binary"),
            "How do you manage stress?": ("🧘‍♂️", "text")
        }

        # Define advice for different score ranges
        advice = {
            "0-3": "Your lifestyle is not very healthy. Consider making some changes to improve your overall well-being.",
            "4-6": "Your lifestyle is moderately healthy. There is room for improvement, so try to make some positive changes.",
            "7-10": "Congratulations! Your lifestyle is very healthy. Keep up the good work!"
        }

        # Initialize score variable
        score = 0

        # Display questions and score input widgets
        for i, question in enumerate(questions.keys()):
            st.subheader(f"{questions[question][0]} {question}")
            if questions[question][1] == "number":
                answer = st.number_input("Enter a number", key=f"number_{i}")
                score += answer
            elif questions[question][1] == "binary":
                answer = st.radio("Select one:", ["Yes", "No"], key=f"binary_{i}")
                if answer == "Yes":
                    score += 1
            elif questions[question][1] == "select":
                answer = st.selectbox("Select an option:", ["Never", "Rarely", "Sometimes", "Often", "Always"], key=f"select_{i}")
                if answer == "Never":
                    score += 0
                elif answer == "Rarely":
                    score += 1
                elif answer == "Sometimes":
                    score += 2
                elif answer == "Often":
                    score += 3
                elif answer == "Always":
                    score += 4
            elif questions[question][1] == "text":
                answer = st.text_input("Enter your answer:", key=f"text_{i}")
                if answer:
                    score += len(answer)

        # Display score and advice
        score_range = ""
        if score <= 10:
            score_range = "0-3"
        elif score <= 20:
            score_range = "4-6"
        else:
            score_range = "7-10"

        st.subheader(f"Your total score is: {score}")
        st.write(f"Advice: {advice[score_range]}")





        
        lottie_success = load_lottieurl("https://assets7.lottiefiles.com/private_files/lf30_hxmzmij0.json")

        st_lottie(
                lottie_success,
                speed=1,
                reverse=False,
                loop=True,
                quality="low", # medium ; high
                height=None,
                width=None,
                key=None,
                )
            
            
                
                
        
        

    #Covid-19 Prediction

    #loading the Covid-19 dataset
    df1=pd.read_csv("Covid-19 Predictions.csv")
    #cleaning the data by dropping unneccessary column and dividing the data as features(x1) & target(y1)
    x1=df1.drop("Infected with Covid19",axis=1)
    x1=np.array(x1)
    y1=pd.DataFrame(df1["Infected with Covid19"])
    y1=np.array(y1)
    #performing train-test split on the data
    x1_train,x1_test,y1_train,y1_test=train_test_split(x1,y1,test_size=0.2,random_state=0)
    #creating an object for the model for further usage
    model1=RandomForestClassifier()
    #fitting the model with train data (x1_train & y1_train)
    model1.fit(x1_train,y1_train)

    #Covid-19 Page

    #heading over to the Covid-19 section
    if rad=="Covid-19":
        st.header("Know If You Are Affected By Covid-19")
        lottie_covid = load_lottieurl("https://assets8.lottiefiles.com/packages/lf20_nw19osms.json")

        st_lottie(
            lottie_covid,
            speed=1,
            reverse=False,
            loop=True,
            quality="low", # medium ; high
            height=None,
            width=None,
            key=None,
            )
        st.write("All The Values Should Be In Range Mentioned")
        #taking the 4 most important features as input as features -> Dry Cough (drycough), Fever (fever), Sore Throat (sorethroat), Breathing Problem (breathingprob)
        #a min value (min_value) & max value (max_value) range is set so that user can enter value within that range
        #incase user enters a value which is not in the range then the value will not be taken whereas an alert message will pop up
        drycough=st.number_input("Rate Of Dry Cough (0-20)",min_value=0,max_value=20,step=1)
        fever=st.number_input("Rate Of Fever (0-20)",min_value=0,max_value=20,step=1)
        sorethroat=st.number_input("Rate Of Sore Throat (0-20)",min_value=0,max_value=20,step=1)
        breathingprob=st.number_input("Rate Of Breathing Problem (0-20)",min_value=0,max_value=20,step=1)
        #the variable prediction1 predicts by the health state by passing the 4 features to the model
        prediction1=model1.predict([[drycough,fever,sorethroat,breathingprob]])[0]
        
        #prediction part predicts whether the person is affected by Covid-19 or not by the help of features taken as input
        #on the basis of prediction the results are displayed
        if st.button("Predict"):
            if prediction1=="Yes":
                st.warning("You Might Be Affected By Covid-19")
            elif prediction1=="No":
                st.success("You Are Safe")

    #Diabetes Prediction

    #loading the Diabetes dataset
    df2=pd.read_csv("Diabetes Predictions.csv")
    #cleaning the data by dropping unneccessary column and dividing the data as features(x2) & target(y2)
    x2=df2.iloc[:,[1,4,5,7]].values
    x2=np.array(x2)
    y2=y2=df2.iloc[:,[-1]].values
    y2=np.array(y2)
    #performing train-test split on the data
    x2_train,x2_test,y2_train,y2_test=train_test_split(x2,y2,test_size=0.2,random_state=0)
    #creating an object for the model for further usage
    model2=RandomForestClassifier()
    #fitting the model with train data (x2_train & y2_train)
    model2.fit(x2_train,y2_train)

    #Diabetes Page

    #heading over to the Diabetes section
    if rad=="Diabetes":
        st.header("Know If You Are Affected By Diabetes")
        lottie_diabetes = load_lottieurl("https://assets8.lottiefiles.com/packages/lf20_l13zwx3i.json")

        st_lottie(
            lottie_diabetes,
            speed=1,
            reverse=False,
            loop=True,
            quality="low", # medium ; high
            height=None,
            width=None,
            key=None,
            )
        st.write("All The Values Should Be In Range Mentioned")
        #taking the 4 most important features as input as features -> Glucose (glucose), Insulin (insulin), Body Mass Index-BMI (bmi), Age (age)
        #a min value (min_value) & max value (max_value) range is set so that user can enter value within that range
        #incase user enters a value which is not in the range then the value will not be taken whereas an alert message will pop up
        glucose=st.number_input("Enter Your Glucose Level (0-200)",min_value=0,max_value=200,step=1)
        insulin=st.number_input("Enter Your Insulin Level In Body (0-850)",min_value=0,max_value=850,step=1)
        bmi=st.number_input("Enter Your Body Mass Index/BMI Value (0-70)",min_value=0,max_value=70,step=1)
        age=st.number_input("Enter Your Age (20-80)",min_value=20,max_value=80,step=1)
        #the variable prediction1 predicts by the health state by passing the 4 features to the model
        prediction2=model2.predict([[glucose,insulin,bmi,age]])[0]
        
        #prediction part predicts whether the person is affected by Diabetes or not by the help of features taken as input
        #on the basis of prediction the results are displayed
        if st.button("Predict"):
            if prediction2==1:
                st.warning("You Might Be Affected By Diabetes")
            elif prediction2==0:
                st.success("You Are Safe")

    #Heart Disease Prediction

    #loading the Heart Disease dataset
    df3=pd.read_csv("Heart Disease Predictions.csv")
    #cleaning the data by dropping unneccessary column and dividing the data as features(x3) & target(y3)
    x3=df3.iloc[:,[2,3,4,7]].values
    x3=np.array(x3)
    y3=y3=df3.iloc[:,[-1]].values
    y3=np.array(y3)
    #performing train-test split on the data
    x3_train,x3_test,y3_train,y3_test=train_test_split(x3,y3,test_size=0.2,random_state=0)
    #creating an object for the model for further usage
    model3=RandomForestClassifier()
    #fitting the model with train data (x3_train & y3_train)
    model3.fit(x3_train,y3_train)

    #Heart Disease Page

    #heading over to the Heart Disease section
    if rad=="Heart Disease":
        st.header("Know If You Are Affected By Heart Disease")
        lottie_heart = load_lottieurl("https://assets8.lottiefiles.com/packages/lf20_smpuejte.json")

        st_lottie(
            lottie_heart,
            speed=1,
            reverse=False,
            loop=True,
            quality="low", # medium ; high
            height=None,
            width=None,
            key=None,
            )
        st.write("All The Values Should Be In Range Mentioned")
        #taking the 4 most important features as input as features -> Chest Pain (chestpain), Blood Pressure-BP (bp), Cholestrol (cholestrol), Maximum HR (maxhr)
        #a min value (min_value) & max value (max_value) range is set so that user can enter value within that range
        #incase user enters a value which is not in the range then the value will not be taken whereas an alert message will pop up
        chestpain=st.number_input("Rate Your Chest Pain (1-4)",min_value=1,max_value=4,step=1)
        bp=st.number_input("Enter Your Blood Pressure Rate (95-200)",min_value=95,max_value=200,step=1)
        cholestrol=st.number_input("Enter Your Cholestrol Level Value (125-565)",min_value=125,max_value=565,step=1)
        maxhr=st.number_input("Enter You Maximum Heart Rate (70-200)",min_value=70,max_value=200,step=1)
        #the variable prediction1 predicts by the health state by passing the 4 features to the model
        prediction3=model3.predict([[chestpain,bp,cholestrol,maxhr]])[0]
        
        #prediction part predicts whether the person is affected by Heart Disease or not by the help of features taken as input
        #on the basis of prediction the results are displayed
        if st.button("Predict"):
            if str(prediction3)=="Presence":
                st.warning("You Might Be Affected By Diabetes")
            elif str(prediction3)=="Absence":
                st.success("You Are Safe")
                                            
    #Plots Page

    #heading over to the plots section
    #plots are displayed for each disease prediction section 
    if rad=="Plots":
        #
        st.title("RYLO Plots💹")
        chart_data = pd.DataFrame(
        np.random.randn(50, 3),
        columns=["People", "Lifestyle", "Trends"])

        st.bar_chart(chart_data)
        type=st.selectbox("Which Plot Do You Want To See?",["Covid-19","Diabetes","Heart Disease"])
        if type=="Covid-19":
            fig=px.scatter(df1,x="Difficulty in breathing",y="Infected with Covid19")
            st.plotly_chart(fig)

        elif type=="Diabetes":
            fig=px.scatter(df2,x="Glucose",y="Outcome")
            st.plotly_chart(fig)
        elif type=="Heart Disease":
            fig=px.scatter(df3,x="BP",y="Heart Disease")
            st.plotly_chart(fig)


create_table()

# Check if the user is authenticated, and display the appropriate page
if "authenticated" not in st.session_state:
    st.session_state.authenticated = False

if st.session_state.authenticated:
    home()
else:
    col1, col2 = st.columns(2)
    with col1:
        login()
    with col2:
        register()