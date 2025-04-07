import google.generativeai as genai

genai.configure(api_key="AIzaSyBNIY7M8dspr7ckk8r9F7KKMHBJNxqIE5Y")

response = genai.list_models()
print(response)
