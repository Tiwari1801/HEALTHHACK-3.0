import streamlit as st
import google.generativeai as genai
from PIL import Image
import PyPDF2
import tempfile
import os
import requests
from google.api_core import exceptions
from dotenv import load_dotenv
import time


load_dotenv()

api_key = os.getenv("GEMINI_API_KEY")
google_maps_api_key = os.getenv("GOOGLE_MAPS_API_KEY")  # For doctor recommendations

if not api_key:
    st.error("Gemini API key not found. Please set the GEMINI_API_KEY environment variable.")
    st.stop()

genai.configure(api_key=api_key)
model = genai.GenerativeModel('gemini-1.5-flash')

MAX_RETRIES = 3
RETRY_DELAY = 2  

def analyze_medical_report(content, content_type):
    prompt = """
    Analyze the provided medical report and provide:
    1. Current health issues detected.
    2. Possible future problems if untreated.
    3. Recommended doctors and specializations required.
    4. Necessary dietary restrictions based on the conditions found.
    5. Additional medical tests or investigations suggested.
    6. Lifestyle changes to improve overall health.
    Provide your response in a structured manner.
    """

    for attempt in range(MAX_RETRIES):
        try:
            if content_type == "image":
                response = model.generate_content([prompt, content])
            else:  
                response = model.generate_content(f"{prompt}\n\n{content}")
            
            return response.text
        except exceptions.GoogleAPIError as e:
            if attempt < MAX_RETRIES - 1:
                st.warning(f"An error occurred. Retrying in {RETRY_DELAY} seconds... (Attempt {attempt + 1}/{MAX_RETRIES})")
                time.sleep(RETRY_DELAY)
            else:
                st.error(f"Failed to analyze the report after {MAX_RETRIES} attempts. Error: {str(e)}")
                return fallback_analysis(content, content_type)


def fallback_analysis(content, content_type):
    st.warning("Using fallback analysis method due to API issues.")
    if content_type == "image":
        return "Unable to analyze the image due to API issues. Please try again later."
    else:  # text
        word_count = len(content.split())
        return f"""
        Fallback Analysis:
        - Document Type: Text-based medical report
        - Word Count: {word_count} words
        - Unable to analyze content in detail due to AI issues. Please consult a doctor manually.
        """

def extract_text_from_pdf(pdf_file):
    pdf_reader = PyPDF2.PdfReader(pdf_file)
    text = ""
    for page in pdf_reader.pages:
        text += page.extract_text()
    return text

def get_doctors(location, specialization="General Practitioner"):
    if not google_maps_api_key:
        return ["Google Maps API Key not found. Cannot fetch doctor recommendations."]
    
    url = f"https://maps.googleapis.com/maps/api/place/textsearch/json?query={specialization}+doctor+near+{location}&key={google_maps_api_key}"
    response = requests.get(url)
    
    if response.status_code == 200:
        results = response.json().get("results", [])
        return [f"{doc['name']} - {doc['formatted_address']}" for doc in results[:5]]
    return ["No doctors found or API issue occurred."]

def main():
    st.title("HEALTH DIAGNOSIS")
    st.write("Upload a medical report (image or PDF) for AI-based analysis.")

    location = st.text_input("Enter your city/location for doctor recommendations (e.g., Bangalore, India):")

    file_type = st.radio("Select file type:", ("Image", "PDF"))

    if file_type == "Image":
        uploaded_file = st.file_uploader("Upload a medical report image", type=["jpg", "jpeg", "png"])
        if uploaded_file is not None:
            with tempfile.NamedTemporaryFile(delete=False, suffix='.png') as tmp_file:
                tmp_file.write(uploaded_file.getvalue())
                tmp_file_path = tmp_file.name

            image = Image.open(tmp_file_path)
            st.image(image, caption="Uploaded Medical Report", use_container_width=True)

            if st.button("Analyze Image Report"):
                with st.spinner("Analyzing the medical report image..."):
                    analysis = analyze_medical_report(image, "image")
                    st.subheader("Report Analysis Results:")
                    st.write(analysis)

                    if location:
                        doctors = get_doctors(location)
                        st.subheader("Recommended Doctors:")
                        st.write("\n".join(doctors))

            os.unlink(tmp_file_path)

    else:  
        uploaded_file = st.file_uploader("Upload a medical report PDF", type=["pdf"])
        if uploaded_file is not None:
            st.write("PDF uploaded successfully")

            if st.button("Analyze PDF Report"):
                with st.spinner("Analyzing the medical report PDF..."):
                    with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as tmp_file:
                        tmp_file.write(uploaded_file.getvalue())
                        tmp_file_path = tmp_file.name

                    with open(tmp_file_path, 'rb') as pdf_file:
                        pdf_text = extract_text_from_pdf(pdf_file)

                    analysis = analyze_medical_report(pdf_text, "text")
                    st.subheader("Report Analysis Results:")
                    st.write(analysis)

                    if location:
                        doctors = get_doctors(location)
                        st.subheader("Recommended Doctors:")
                        st.write("\n".join(doctors))

                    os.unlink(tmp_file_path)

if __name__ == "__main__":
    main()
