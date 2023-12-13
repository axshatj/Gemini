import streamlit as st
import PIL.Image
import google.generativeai as genai

GOOGLE_API_KEY = "AIzaSyBOjah1qM9ZXSzr7FKQe68L5qrf3ITY22Y"
genai.configure(api_key=GOOGLE_API_KEY)

vision_model = genai.GenerativeModel('gemini-pro-vision')
text_model = genai.GenerativeModel('gemini-pro')

def handle_message():
    vision_chat = vision_model.start_chat(history=[])
    text_chat = text_model.start_chat(history=[])
    if uploaded_file and send_button:
        response_text = process_image(uploaded_file,vision_chat)
        output_box.write(response_text)
    elif input_box and send_button:
        response_text = process_text(input_box,text_chat)
        output_box.write(response_text)
        print(text_chat.history)

def process_image(uploaded_file,vision_chat):
    img = PIL.Image.open(uploaded_file)
    response = vision_model.generate_content(img,
                                             safety_settings={'HARASSMENT': 'block_none'},
                                             generation_config=genai.types.GenerationConfig(
                                             max_output_tokens=4000,
                                             temperature=0.7)
                                             )
    return response.text

def process_text(input_text,text_chat):
    response = text_chat.send_message(input_text,
                                      safety_settings={'HARASSMENT': 'block_none'},
                                      generation_config=genai.types.GenerationConfig(
                                      max_output_tokens=4000,
                                      temperature=0.9)
                                     )
    return response.text

st.title("Chat with Gemini")

uploaded_file = st.file_uploader("Upload an image")
input_box = st.text_input("Your Message: ")
send_button = st.button("Send")
output_box = st.empty()


if __name__ == "__main__":
    handle_message()
