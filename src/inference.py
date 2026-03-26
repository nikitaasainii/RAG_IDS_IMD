from groq import Groq
from .config import GROQ_API_KEY, MODEL_NAME
from .preprocessing import row_to_string

class RAGInference:
    def __init__(self):
        """Initializes the Groq client using the API key from config."""
        if not GROQ_API_KEY:
            raise ValueError("GROQ_API_KEY not found. Please check your .env file.")
        
        self.client = Groq(api_key=GROQ_API_KEY)
        self.model = MODEL_NAME

    def generate_analysis(self, test_row, retrieved_cases):
        """
        Combines the test row with historical context and sends it to Llama 3.1.
        """
        # 1. Prepare the query string
        query_str = row_to_string(test_row)
        
        # 2. Format the retrieved cases into a readable context
        context = ""
        for i, (doc, metadata) in enumerate(zip(retrieved_cases['documents'][0], retrieved_cases['metadatas'][0])):
            context += f"Historical Case {i+1}: Features: [{doc}] | Label: {metadata['label']}\n"

        # 3. Create the Expert System Prompt
        prompt = f"""
        SYSTEM: You are a high-level Cybersecurity Analyst specializing in Intrusion Detection (IDS).
        Your task is to classify a NEW CONNECTION based on HISTORICAL CONTEXT.

        HISTORICAL CONTEXT FROM KNOWLEDGE BASE:
        {context}

        NEW CONNECTION DATA TO ANALYZE:
        {query_str}

        INSTRUCTIONS:
        1. Compare the new data to the historical cases.
        2. Identify if it is 'normal' or an 'anomaly'.
        3. Even if the historical label is generic, use your knowledge to predict the attack category (DoS, Probe, R2L, or U2R).
        4. Provide a technical reason focusing on specific features (e.g., serror_rate, count).

        OUTPUT FORMAT (Strictly follow this):
        Label: [normal/anomaly]
        Category: [DoS/Probe/R2L/U2R/None]
        Reason: [Brief technical explanation]
        """

        # 4. Call the Groq API
        try:
            chat_completion = self.client.chat.completions.create(
                messages=[
                    {"role": "system", "content": "You are a helpful cybersecurity assistant."},
                    {"role": "user", "content": prompt}
                ],
                model=self.model,
                temperature=0.1, # Low temperature for consistent, factual results
            )
            return chat_completion.choices[0].message.content
        except Exception as e:
            return f"Error during inference: {str(e)}"