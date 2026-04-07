import os

class RAGGenerator:
    """ Modular handler that uses an LLM to generate an answer based purely on context """
    def __init__(self, api_key: str = None):
        # We now look for GROQ_API_KEY (which is completely free!)
        self.api_key = api_key or os.getenv("GROQ_API_KEY")
        if self.api_key:
            # Groq ingeniously uses the exact same OpenAI library, we just change the URL!
            from openai import OpenAI
            self.client = OpenAI(
                api_key=self.api_key,
                base_url="https://api.groq.com/openai/v1"
            )
        else:
            self.client = None

    def generate_response(self, query: str, contexts: list[str]) -> str:
        if not self.client:
            return "RAG Generation is currently disabled. Please provide a free GROQ_API_KEY in the environment."
            
        # Combine the distinct document chunks into one readable context
        context_block = "\n\n".join([f"Doc {i+1}: {text}" for i, text in enumerate(contexts)])
        
        system_prompt = (
            "You are a helpful, expert intelligent assistant. Use ONLY the provided context "
            "to answer the user's question. If the answer cannot be confidently answered based on "
            "the provided documents, clearly say you don't know. Do not hallucinate."
        )
        user_prompt = f"Context Documents:\n{context_block}\n\nUser Question: {query}"
        
        try:
            # Calling the completely free Groq Llama-3 API
            response = self.client.chat.completions.create(
                model="llama-3.3-70b-versatile",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=0.3,
                max_tokens=300
            )
            return response.choices[0].message.content
        except Exception as e:
            return f"Error executing RAG inference: {str(e)}"
