import boto3
import json
import pandas as pd

def add_pii_to_clinical_note(redacted_text):
    bedrock = boto3.client('bedrock-runtime', region_name='us-west-2')
    
    prompt = f"""Replace all redacted PII placeholders with realistic synthetic data. Keep all medical content identical.
    Use variety of PII. Do not repeat.

Replace patterns like:
- [**Name**] or [**Patient Name**] → realistic names
- [**Date**] or dates like [**2024-1-1**] → realistic dates
- [**Phone**] or [**Telephone**] → realistic phone numbers
- [**Address**] or [**Location**] → realistic addresses
- [**Hospital**] or [**Hospital Name**] → realistic hospital names
- [**Doctor**] or [**Provider**] → realistic doctor names
- [**MRN**] or [**Medical Record Number**] → realistic MRN

Return ONLY the clinical note with PII filled in:

{redacted_text}"""

    body = {
        "anthropic_version": "bedrock-2023-05-31",
        "max_tokens": 2000,
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
    """Process pandas DataFrame and add PII to redacted clinical notes"""
    results = []
    for idx, row in df.iterrows():
        try:
            note_with_pii = add_pii_to_clinical_note(row[text_column])
            print(note_with_pii)
            results.append(note_with_pii)
        except Exception as e:
            results.append(f"Error: {str(e)}")
    
    df['note_with_pii'] = results
    return df


if __name__ == "__main__":
    df = pd.read_csv('clinical_notes_with_qa_medications.csv')
    df = process_dataframe(df, 'note')
    df.to_csv('clinical_notes_with_pii_added.csv', index=False)