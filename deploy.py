import streamlit as st
import pandas as pd
import pickle
import numpy as np
from PIL import Image

# use the current working directory to define the path of the dementia.csv file
datafile = "dementia_dataset.csv"

# use the current working directory to define the path of the trained_model.pkl file
trained_model = "trained_model.pkl"

# use the current working directory to define the path of the images folder
image_folder = "images"

@st.cache
def load_image(imagefile):
    img = Image.open(imagefile)
    return img

def main():
    st.title("Dementia Risk Prediction System")
    st.write("Welcome to our dementia prediction app. We have created this app to help predict whether an individual has dementia based on various factors such as age, gender, education level, socioeconomic status, Mini-Mental State Examination (MMSE) score, and brain volumes.")
    st.write("Our app utilizes a machine learning model that was trained on a dementia dataset to make predictions.")

    menu = ["About", "Dataset", "Prediction"]
    choice = st.sidebar.selectbox("Menu", menu)

    if choice == "Dataset":
        st.write("The dataset is shown below.")
        st.subheader("Dataset")
        df = pd.read_csv(datafile)
        st.dataframe(df)
        
       
    elif choice == "Prediction":
        st.subheader("Prediction")

        # Input fields with sliders and text inputs
        Visit = st.slider("Visit", 0, 10, step=1)
        MR_Delay = st.slider("MR Delay", 0, 4000, step=1)
        Gender = st.selectbox("Gender", ["Male", "Female"])
        Age = st.slider("Age", 60, 100, step=1)
        EDUC = st.slider("Education Level (Years)", 0, 20, step=1)
        SES = st.slider("Socioeconomic Status", 1, 5, step=1)
        MMSE = st.slider("Mini-Mental State Examination (MMSE) Score", 0, 30, step=1)
        CDR = st.slider("Clinical Dementia Rating (CDR)", 0.0, 3.0, step=0.5)
        eTIV = st.slider("Estimated Total Intracranial Volume (eTIV)", 1000, 2000, step=1)
        nWBV = st.slider("Normalized Whole-Brain Volume (nWBV)", 0.5, 1.0, step=0.01)
        ASF = st.slider("Atlas Scaling Factor (ASF)", 0.8, 1.5, step=0.01)

        # Convert gender to numeric
        Gender = 1 if Gender == "Male" else 0
        
        # loading the saved model
        loaded_model = pickle.load(open(trained_model, 'rb'))

        if st.button("Predict"):
            prediction = loaded_model.predict([[Visit, MR_Delay, Gender, Age, EDUC, SES, MMSE, CDR, eTIV, nWBV, ASF]])
            if prediction == 0:
                st.success("The patient is not demented.")
            else:
                st.warning("The patient is likely to be demented.")

    elif choice == "About":
        
        st.subheader("About")
        # imagefile = f"{image_folder}/dementia_info.jpg"
        # st.image(load_image(imagefile), width=300)
        
        image_url = "https://ch-api.healthhub.sg/api/public/content/44bcb8d6bd504474b0f3a5476ed3ec6a?v=698b258e&t=livehealthyheaderimage"  # Replace with your image URL
        st.image(image_url, width=300)
        
        st.write("This app is designed to help individuals, doctors, and healthcare providers better understand and manage dementia. The app takes into account various factors such as age, gender, education level, socioeconomic status, MMSE score, brain volumes, and other related factors to make predictions.")
        st.write("The app's machine learning model was trained on a dementia dataset that contains information about patients with and without dementia.")
        st.write("We hope that this app will help individuals and healthcare providers better understand and manage dementia, ultimately leading to improved health outcomes for individuals with dementia.")
        st.write("We are constantly working to improve the app and welcome any feedback or suggestions you may have. Feel free to contact us with any questions or concerns.")

if __name__ == "__main__":
    main()
