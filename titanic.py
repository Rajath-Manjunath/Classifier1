import streamlit as st
import pandas as pd
import pickle
import sklearn

cars_df = pd.read_csv("./Titanic-Dataset.csv")

st.title("Titanic Survival Prediction")
st.dataframe(cars_df.head())

with open('classifier.pkl', 'rb') as f:
    model = pickle.load(f)

# sklearn model which is trained on cars24 data.



#col1, col2 = st.columns(2)

# dropdown
#fuel_type = col1.selectbox("Select the fuel type",
#                          ["Diesel", "Petrol", "CNG", "LPG", "Electric"])

#engine = col1.slider("Set the Engine Power",
#                     500, 5000, step=100)

#transmission_type = col2.selectbox("Select the transmission type",
#                                   ["Manual", "Automatic"])

#seats = col2.selectbox("Enter the number of seats",
#                       [4,5,7,9,11])


## Encoding Categorical features
## Use the same encoding as used during the training. 
#encode_dict = {
#    "fuel_type": {'Diesel': 1, 'Petrol': 2, 'CNG': 3, 'LPG': 4, 'Electric': 5},
#    "seller_type": {'Dealer': 1, 'Individual': 2, 'Trustmark Dealer': 3},
#    "transmission_type": {'Manual': 1, 'Automatic': 2}
#}


#if st.button("Get Price"):
    # predict here

#    encoded_fuel_type = encode_dict['fuel_type'][fuel_type]
#    encoded_transmission_type = encode_dict['transmission_type'][transmission_type]

#    input_car = [2012.0,2,120000,encoded_fuel_type,encoded_transmission_type,19.7,engine,46.3,seats]
#    price = model.predict([input_car])[0]

#    st.header(round(price,2))
pclass = st.selectbox("Select the passenger class",["1st","2nd","3rd"])
sex=st.selectbox("Select your gender",["Male","Female"])
age=st.slider("Set your Age",10,100,step=1)
SibSp=st.slider("Select how many siblings/spouse with you",1,5,step=1)
parch=st.slider("Select how many parents/children with you",1,5,step=1)
fare=st.slider("How much fare did you pay",0,550,step=50)
embarked=st.selectbox("Where did you embark",["Cherbourg","Queenstown","Southampton"])

encode_dict = {
    "pclass": {'1st': 1, '2nd': 2, '3rd': 3},
    "sex": {'Male': 0, 'Female': 1},
    "embarked": {'Southampton': 0, 'Cherbourg': 1, 'Queenstown': 2}
}

encoded_pclass = encode_dict['pclass'][pclass]
encoded_sex = encode_dict['sex'][sex]
encoded_embarked = encode_dict['embarked'][embarked]


if st.checkbox("Check if they survived"):
    prediction=model.predict([[encoded_pclass,encoded_sex,age,SibSp,parch,fare,encoded_embarked]])
    if prediction[0]==0:
        st.header("Oh no You did not survive")
    if prediction[0]==1:
        st.header("Yeeeeeeaaaaa You survived")






# scaler = StandardScaler()
# scaler.fit_transform(X)

# pickle.dump(scaler)

# scaler.transform()
