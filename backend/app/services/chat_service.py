from groq import Groq
import os
from typing import List, Dict

class ChatService:
    def __init__(self, api_key: str):
        """Initialize Groq Chat Service"""
        self.client = Groq(api_key=api_key)
        self.model = "llama-3.3-70b-versatile"  # Fast, free model
        
    def get_medical_advice(
        self, 
        disease: str, 
        confidence: float,
        user_question: str = None
    ) -> str:
        """
        Get AI doctor advice based on prediction
        """
        
        # Build system prompt (AI doctor personality)
        system_prompt = """You are Dr. EyeTrust, a compassionate ophthalmologist AI assistant. 

Your role:
- Explain eye diseases in simple, understandable language
- Provide general guidance and recommendations
- Be empathetic and reassuring
- ALWAYS emphasize: "This is AI-generated guidance. Please consult a real ophthalmologist for diagnosis and treatment."

Keep responses:
- Clear and concise (2-3 paragraphs max)
- Actionable
- Empathetic
- Professional but friendly"""

        # Build user message based on prediction
        if user_question:
            user_message = f"""
The patient's eye image shows: {disease} (AI confidence: {confidence*100:.1f}%)

Patient's question: {user_question}

Please provide helpful guidance.
"""
        else:
            user_message = f"""
The patient's eye image shows: {disease} (AI confidence: {confidence*100:.1f}%)

Please explain:
1. What is {disease}?
2. What are common symptoms?
3. What should the patient do next?
4. Any immediate care tips?
"""
        
        try:
            # Call Groq API
            response = self.client.chat.completions.create(
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_message}
                ],
                model=self.model,
                temperature=0.7,
                max_tokens=500
            )
            
            return response.choices[0].message.content
            
        except Exception as e:
            return f"Unable to generate advice: {str(e)}"
    
    def chat(
        self, 
        user_message: str,
        conversation_history: List[Dict] = None
    ) -> str:
        """
        General chat with AI doctor
        """
        
        system_prompt = """You are Dr. EyeTrust, a friendly ophthalmologist AI assistant.

Answer questions about:
- Eye health and diseases
- Symptoms and when to see a doctor
- General eye care tips
- Prevention and healthy habits

Always remind users: "I'm an AI assistant. For diagnosis and treatment, please consult a real ophthalmologist."

Keep responses concise and helpful."""

        # Build messages list
        messages = [{"role": "system", "content": system_prompt}]
        
        # Add conversation history if provided
        if conversation_history:
            messages.extend(conversation_history)
        
        # Add current user message
        messages.append({"role": "user", "content": user_message})
        
        try:
            response = self.client.chat.completions.create(
                messages=messages,
                model=self.model,
                temperature=0.7,
                max_tokens=400
            )
            
            return response.choices[0].message.content
            
        except Exception as e:
            return f"Unable to respond: {str(e)}"

# Global chat service
chat_service = None

def get_chat_service() -> ChatService:
    """Get or create chat service"""
    global chat_service
    if chat_service is None:
        api_key = os.getenv("GROQ_API_KEY")
        if not api_key:
            raise ValueError("GROQ_API_KEY not found in environment")
        chat_service = ChatService(api_key=api_key)
    return chat_service