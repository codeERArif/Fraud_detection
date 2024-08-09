# import streamlit as st
# import pickle
# import numpy as np
# from PIL import Image

# from streamlit_option_menu import option_menu

# with st.sidebar:
#     selected = option_menu(
#         menu_title="Menu",
#         options=["Home", "About", "Contact"],
#         icons=["house-heart-fill", "calendar2-heart-fill", "envelope-heart-fill"],
#         menu_icon="emoji-heart-eyes-fill",
#         default_index=0,
#     )

# if selected == "Home":
#     st.title("Welcome to the Project Page of Streamlit")
#     img = Image.open("fraud.png")
#     st.image(img,
#              width=1000
#              )

# st.header("Credit Card Fraud Detection Model")

# input_df = st.text_input("Please provide all the required feature details: ")
# input_df_split = input_df.split(',')
# print(input_df)
# submit = st.button("Submit")

# if submit:
#     model = pickle.load(open('model.pkl', 'rb'))
#     features = np.asarray(input_df_split, dtype=np.float64)
#     prediction = model.predict(features.reshape(1, -1))

#     if prediction[0] == 0:
#         st.write("Legitimate Transaction")
#     else:
#         st.write("Fraudulent Transaction")

# if selected == "About":
#     st.title("Welcome to the About page")
#     st.markdown(
#         "The main objective of this work is to explore whether a credit card transaction (before being processed) is "
#         "fraudulent or not. And if it is found out to be fraudulent, then the credit card owner must be notified of "
#         "the same.The system currently used to detect fraud is plagued by classification and highly false "
#         "positives. So, our motive is to not miss any fraud cases (high recall) as well as not predict too many "
#         "non-fraud cases as fraud cases (high precision) as if this happens, it will lead to poor reviews for the "
#         "concerned credit card company. We demonstrate various methods to deal with the imbalance of the data such as "
#         "choosing appropriate metrics for evaluation of models and resampling of data.")

# if selected == "Contact":
#     st.title("Welcome to the Contact page")
#     st.markdown("Email - arf4852@gmail.com")
#     st.markdown("Github - https://github.com/codeERArif")
#     st.markdown("LinkedIn - https://www.linkedin.com/in/arif-khan-5155931ba/")


import streamlit as st
import pickle
import numpy as np
from PIL import Image
from streamlit_option_menu import option_menu



# Sidebar menu
with st.sidebar:
    selected = option_menu(
        menu_title="Menu",
        options=["Home", "About", "Contact"],
        icons=["house-heart-fill", "calendar2-heart-fill", "envelope-heart-fill"],
        menu_icon="emoji-heart-eyes-fill",
        default_index=0,
    )

# Home page
if selected == "Home":
    st.title("Welcome to the Project Page of Streamlit")
    img = Image.open("card.jpeg")
    st.image(img, width=1000)

    st.header("Credit Card Fraud Detection Model")

    input_df = st.text_input(r"$\textsf{\Huge Enter required featured details here}$")
    submit = st.button("Submit")

    if submit:
        try:
            features = np.asarray(input_df.split(','), dtype=np.float64)
            model = pickle.load(open('model.pkl', 'rb'))
            prediction = model.predict(features.reshape(1, -1))

            if prediction[0] == 0:
                st.write("Genuine Transaction")
            else:
                st.write("Fraudulent Transaction")
        except ValueError:
            st.error("Invalid input format. Please enter comma-separated numerical values.")
        except FileNotFoundError:
            st.error("Model file not found. Please ensure 'model.pkl' is in the correct directory.")
        except Exception as e:
            st.error(f"An error occurred: {e}")

# About page
if selected == "About":
    st.title("Welcome to the About page")
    st.markdown(
        "The main objective of this work is to explore whether a credit card transaction (before being processed) is "
        "fraudulent or not. If it is found to be fraudulent, the credit card owner must be notified. The system "
        "currently used to detect fraud is plagued by classification issues and high false positives. Our goal is to "
        "maximize recall (not miss any fraud cases) and precision (minimize false positives), ensuring a balance that "
        "maintains good reviews for the credit card company. We demonstrate various methods to deal with data "
        "imbalance, including choosing appropriate metrics for model evaluation and data resampling."
    )

# Contact page
if selected == "Contact":
    st.title("Welcome to the Contact page")
    st.markdown("Email - arf4852@gmail.com")
    st.markdown("Github - [codeERArif](https://github.com/codeERArif)")
    st.markdown("LinkedIn - [Arif Khan](https://www.linkedin.com/in/arif-khan-5155931ba/)")
