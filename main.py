import streamlit as st
import os
import json
import google.generativeai as genai
import tempfile

st.set_page_config(page_title="Chat Image Processor", page_icon="💬", layout="centered")

st.markdown(
    """
    <style>
        #MainMenu {visibility: hidden;}
        footer {visibility: hidden;}
        header {visibility: hidden;}
    </style>
    """,
    unsafe_allow_html=True
)


def initialize_gemini():
    api_key = os.getenv("GEMINI_API_KEY") or st.secrets.get("GEMINI_API_KEY")
    if not api_key:
        st.error("Please set the GEMINI_API_KEY.")
        st.stop()
    
    genai.configure(api_key=api_key)
    return genai.GenerativeModel(
        model_name="gemini-2.0-flash-exp",
        generation_config={
            "temperature": 0.25,
            "top_p": 0.95,
            "top_k": 40,
            "max_output_tokens": 8192,
            "response_mime_type": "application/json",
        },
        system_instruction=r"""
        Please analyze the given chat image and extract the following information in JSON format:

        - **message**: The text content of the message.
        - If the message contains both text and emojis, return the full text including emojis.
        - If the message contains only emojis, return them as they are in the "message" field.

        - **timestamp**: The timestamp in ISO format (YYYY-MM-DDTHH:mm:ssZ).
        - If the timestamp is missing, return `null`.

        - **emojis**: An array of Unicode escape sequences representing the emojis in the message.
        - Convert each emoji character to its Unicode escape sequence using **`\uXXXX` or `\UXXXXXXXX` format**.
        - If no emojis are present, return `null`.

        ### **Edge Cases Handling:**
        1. If any field does not exist, set it to `null`. **Do not generate incorrect information.**  
        2. If a message consists only of emojis:
        - The "message" field should contain the actual emoji characters.
        - The "emojis" field should contain their Unicode escape sequences.  
        3. Return only the JSON object without any additional text, explanation, or formatting errors.

        ### **Response Format Example:**
        ```
        {
            "message": "Hello 😊",
            "timestamp": null,
            "emojis": ["\\ud83d\\ude0a"]
        }
        ```
        ---

        #### **Example 1:** (Text + Emoji)
        **Input:** `"Hey 👋, how are you? 😊"`  
        **Output:**
        ```
        {
            "message": "Hey 👋, how are you? 😊",
            "timestamp": null,
            "emojis": ["\\ud83d\\udc4b", "\\ud83d\\ude0a"]
        }
        ```
        ---

        #### **Example 2:** (Only Emojis)
        **Input:** `"😂🔥❤️"`  
        **Output:**
        ```
        {
            "message": "😂🔥❤️",
            "timestamp": null,
            "emojis": ["\\ud83d\\ude02", "\\ud83d\\udd25", "\\u2764"]
        }
        ```
        ---

        #### **Example 3:** (No Emoji)
        **Input:** `"Good morning!"`  
        **Output:**
        ```
        {
            "message": "Good morning!",
            "timestamp": null,
            "emojis": null
        }
        ```

        ### **Fix Summary:**
        ✅ **Ensures proper Unicode conversion**  
        ✅ **Handles emoji-only messages correctly**  
        ✅ **Strict JSON compliance**  
        ✅ **No incorrect or malformed emoji encoding**



        """
    )

def process_image(image_file, model):
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix='.png') as tmp_file:
            tmp_file.write(image_file.getvalue())
            tmp_path = tmp_file.name
        
        uploaded_file = genai.upload_file(tmp_path, mime_type="image/png")
        os.unlink(tmp_path)
        
        chat = model.start_chat()
        response = chat.send_message([uploaded_file, "Analyze this image carefully."])
        
        return json.loads(response.text)
    except Exception as e:
        return {"error": str(e)}

def main():
    st.title("Chat Image Processor 💬")
    st.write("Upload a chat screenshot to extract messages, timestamps, and emojis.")
    
    model = initialize_gemini()
    uploaded_file = st.file_uploader("Choose a chat screenshot", type=['png', 'jpg', 'jpeg'])
    
    if uploaded_file and st.button("Process Image"):
        with st.spinner("Processing..."):
            results = process_image(uploaded_file, model)
            
            if "error" in results:
                st.error(results["error"])
            else:
                st.success("Processing complete!")
                st.json(results, expanded=True)

if __name__ == "__main__":
    main()
