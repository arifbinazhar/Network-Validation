import google.generativeai as genai
from mistralai.client import MistralClient

def build_prompt(query, retrieved_docs):
    context = ""

    for doc in retrieved_docs:
        context += f"""
        PMID: {doc['pmid']}
        Gene: {doc['gene']}
        Disease: {doc['disease']}
        Relation Type: {doc['relation_type']}
        Score: {doc['ot_score']}
        Text: {doc['text']}
        ---
        """

    prompt = f"""
    You are a biomedical reasoning assistant.

    Use the context primarily but you may reason moderately.
    Cite PMIDs when possible.

    Context:
    {context}

    Question:
    {query}

    Provide:
    - Evidence summary
    - Mechanistic insight
    - Strength of association
    - Confidence
    - Supporting PMIDs
    """

    return prompt


def ask_gemini(prompt, api_key):
    genai.configure(api_key=api_key)
    model = genai.GenerativeModel("gemini-1.5-pro")
    response = model.generate_content(prompt)
    return response.text


def ask_mistral(prompt, api_key):
    client = MistralClient(api_key=api_key)
    response = client.chat(
        model="mistral-large-latest",
        messages=[{"role": "user", "content": prompt}]
    )
    return response.choices[0].message.content