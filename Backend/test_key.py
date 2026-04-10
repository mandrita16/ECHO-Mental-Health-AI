import os
import google.generativeai as genai
from dotenv import load_dotenv

load_dotenv(override=True) 

api_key = os.getenv("GEMINI_API_KEY")
genai.configure(api_key=api_key)

try:
    print("Testing connection...")
    response = genai.GenerativeModel("gemini-2.0-flash").generate_content("Say hi!")
    print(f"✅ Success! AI says: {response.text}")
except Exception as e:
    print(f"❌ Still blocked: {e}")