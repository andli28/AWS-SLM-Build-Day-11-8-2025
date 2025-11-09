import streamlit as st
import numpy as np
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import json
import re

logo_path = "streamlit/FHIRFly_logo.png" 

st.html("""
  <style>
    [alt=Logo] {
      padding-top: 2.5rem;
      height: 8rem;
    }
  </style>
        """)

st.logo(logo_path, size="large", link="https://fhirfly.streamlit.app/") 

@st.cache_resource
def load_model():
    tokenizer = AutoTokenizer.from_pretrained("TinyLlama/TinyLlama-1.1B-Chat-v1.0")
    model = AutoModelForCausalLM.from_pretrained("TinyLlama/TinyLlama-1.1B-Chat-v1.0")
    return tokenizer, model

def extract_medical_info(clinical_note, tokenizer, model):
    prompt = f"""Extract conditions and medications from this clinical note and return only a JSON object:

Clinical Note: {clinical_note}

Return JSON format:
{{
  "conditions": [
    {{"text": "condition name", "status": "active/resolved/potential"}}
  ],
  "medications": [
    {{"text": "medication name and dosage", "dosage": "frequency"}}
  ]
}}

JSON:"""
    
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512)
    with torch.no_grad():
        outputs = model.generate(**inputs, max_new_tokens=200, temperature=0.7, do_sample=True)
    
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    json_match = re.search(r'\{.*\}', response, re.DOTALL)
    
    if json_match:
        try:
            return json.loads(json_match.group())
        except:
            pass
    
    return {"conditions": [], "medications": []}

st.write("""
# FHIRFly
Clinical Notes Summarizer - SLM Build Day

Input a clinical or doctor's note to get a structured JSON summarization!
""")

clincal_note = st.text_area("Enter your note:")

if clincal_note:
    st.write(f"Original note:\n\n {clincal_note}")

    with st.spinner('Analyzing note contents...'):
        tokenizer, model = load_model()
        result = extract_medical_info(clincal_note, tokenizer, model)

        tab1, tab2, tab3 = st.tabs(["Diagnosis", "Medications", "Structured JSON"])
        
        with tab1:
            st.header("Diagnosis")
            if result["conditions"]:
                for condition in result["conditions"]:
                    st.write(f"• {condition['text']} ({condition['status']})")
            else:
                st.write("No conditions identified.")
        
        with tab2:
            st.header("Medications")
            if result["medications"]:
                for medication in result["medications"]:
                    st.write(f"• {medication['text']} - {medication['dosage']}")
            else:
                st.write("No medications identified.")
                     
        with tab3:
            st.header("Structured JSON Output")
            st.json(result)
