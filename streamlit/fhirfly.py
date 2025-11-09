import streamlit as st
import numpy as np

logo_path = "streamlit/FHIRFly_logo.png" 

st.html("""
  <style>
    [alt=Logo] {
      padding-top: 1rem;
      height: 8rem;
    }
  </style>
        """)

st.logo(logo_path, size="large", link="https://fhirfly.streamlit.app/") 

st.write("""
# FHIRFly
Clinical Notes Summarizer - SLM Build Day
Input a clinical or doctor's note to get a structured JSON summarization!
""")

clincal_note = st.text_input("Enter your note:")

if clincal_note:
    st.write(f"Original note:\n {clincal_note}")

    with st.spinner('Analyzing note contents...'):

        # model
#         genres = model.predict_genre('output.wav', 40)

        # create tabs for each subtopic
        tab1, tab2, tab3 = st.tabs(["Diagnosis", "Medications", "Structured JSON"])
        
        # diagnosis
        with tab1:
            st.header("Diagnosis")
            
            st.write("""
            The diagnosis of the note is:

            """)
        
        # medication
        with tab2:
            st.header("Medications")
            st.write("""
            These are the medications recommended for the diagnosis:
            
            """)
                     
        # structured json 
        with tab3:
            st.header("Structured JSON Output")
            st.write("""
            Here is the full structured JSON of your note:
            
            """)
