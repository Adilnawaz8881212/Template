import os
import streamlit as st
from modules.audio_processor import transcribe_audio
from modules.nlp_extractor import extract_data
from modules.template_matcher import match_template
from modules.pdf_filler import fill_pdf

# Hard-coded template info
TEMPLATES = {
    "invoice": {
        "path": "templates/sample_fillable_form.pdf",
        "description": "Invoice for service payment"
    },
    "agreement": {
        "path": "templates/agreement_template.pdf",
        "description": "Service level agreement form"
    }
    # Add more if needed
}

# Streamlit UI
st.title("Audio-to-PDF Processor")

# Audio file upload
uploaded_file = st.file_uploader("Upload an audio file", type=["mp3", "wav", "m4a"])

if uploaded_file is not None:
    # Save the uploaded file temporarily
    with open("temp_audio_file", "wb") as f:
        f.write(uploaded_file.getbuffer())

    st.write("Audio file uploaded successfully.")

    if st.button("Process Audio"):
        # Process the uploaded audio
        audio_path = "temp_audio_file"
        
        # 1) Transcribe
        transcript = transcribe_audio(audio_path)
        st.write("\n>> Transcript:\n", transcript)

        # 2) Extract Data
        extracted = extract_data(transcript)
        st.write("\n>> Extracted Data:\n", extracted)

        # 3) Pick Template
        chosen = match_template(transcript, TEMPLATES)
        st.write("\n>> Matched Template:\n", chosen)

        # 4) Fill PDF
        template_path = TEMPLATES[chosen]["path"]
        base_name = os.path.splitext(os.path.basename(audio_path))[0]
        output_path = f"output/filled_{base_name}.pdf"

        # Ensure output folder exists
        os.makedirs("output", exist_ok=True)
        
        fill_pdf(template_path, output_path, extracted)
        st.write(f"\n>> PDF created at: {output_path}")
        
        # Display download link
        with open(output_path, "rb") as f:
            st.download_button("Download PDF", f, file_name=f"filled_{base_name}.pdf")
