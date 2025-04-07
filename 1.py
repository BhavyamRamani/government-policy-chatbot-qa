import google.generativeai as genai

genai.configure(api_key="your_api_key")

response = genai.list_models()
print(response)
