import boto3
import json
import pandas as pd

def extract_discharge_components(discharge_text):
    bedrock = boto3.client('bedrock-runtime', region_name='us-west-2')
    
    prompt = f"""You are an expert clinical data extraction bot. 
    Extract diagnosed conditions and prescribed medications from this clinical note. Return ONLY valid JSON:

{{
  "conditions": [
    {{"text": "condition name", "status": "active/resolved/potential"}}
  ],
  "medications": [
    {{"text": "medication name and dosage", "dosage": "frequency"}}
  ]
}}

Clinical note:
{discharge_text}"""

    body = {
        "anthropic_version": "bedrock-2023-05-31",
        "max_tokens": 1000,
        "messages": [{
            "role": "user",
            "content": prompt
        }]
    }
    
    response = bedrock.invoke_model(
        modelId='anthropic.claude-3-5-sonnet-20241022-v2:0',
        body=json.dumps(body)
    )
    
    result = json.loads(response['body'].read())
    return result['content'][0]['text']

def process_dataframe(df, text_column):
    """Process pandas DataFrame and extract components from specified column"""
    results = []
    for idx, row in df.iterrows():
        try:
            extracted = extract_discharge_components(row[text_column])
            print(extracted)
            results.append(extracted)
        except Exception as e:
            results.append(f"Error: {str(e)}")
    
    df['extracted_json'] = results
    return df


if __name__ == "__main__":
    df = pd.read_csv('clinical_notes_with_qa_medications.csv')
    df = process_dataframe(df, 'note')
    df.to_csv('clinical_notes_with_qa_medications_labeled.csv', index=False)
    
#     # Test with sample data
#     sample_text = """Discharge Summary:

# Patient Name: [REDACTED]
# DOB: [REDACTED]
# Sex: Female
# Admission Date: [REDACTED]
# Discharge Date: [REDACTED]

# Reason for Admission:
# The patient was admitted with shortness of breath, cough, myalgias, and malaise in the setting of a positive SARS-CoV-2 test result.

# Hospital Course:
# Upon admission, the patient was started on supplemental oxygen, remdesivir, dexamethasone, furosemide, azithromycin, and enoxaparin for venous thromboembolism prophylaxis. The patient's clinical course was challenging, despite multiple measures, and required more aggressive management. Repeat chest X-ray showed slight interval improvement of bilateral pulmonary infiltrates. The patient's condition gradually improved over the course of her hospital stay, and she was eventually determined to be fit for discharge.

# Discharge Medications:
# Patient's discharge medications will be managed by her primary care provider (PCP).

# Follow-Up:
# The patient should follow up with her PCP within seven days and continue to monitor her symptoms.

# Summary:
# The patient was admitted due to shortness of breath, cough, myalgias, and malaise in the setting of a positive SARS-CoV-2 test result. Despite multiple measures, the patient's clinical course was challenging and required more aggressive management. The patient's clinical condition gradually improved, and she was eventually discharged. The patient will follow up with her PCP within seven days and continue to monitor her symptoms."""

    # result = extract_discharge_components(sample_text)
    # print(result)