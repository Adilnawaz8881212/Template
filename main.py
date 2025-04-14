import os
import json
import tempfile
import streamlit as st
import sounddevice as sd
import soundfile as sf
import numpy as np
from faster_whisper import WhisperModel
import spacy
import re
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
from reportlab.lib import colors
from sentence_transformers import SentenceTransformer, util
import time
from datetime import datetime

# Set page config
st.set_page_config(page_title="Audio-to-PDF Processor", layout="wide")

# Load models
@st.cache_resource
def load_whisper_model():
    return WhisperModel("base")

@st.cache_resource
def load_spacy_model():
    return spacy.load("en_core_web_sm")

@st.cache_resource
def load_embedding_model():
    return SentenceTransformer("all-MiniLM-L6-v2")

whisper_model = load_whisper_model()
nlp = load_spacy_model()
embed_model = load_embedding_model()

# Audio Transcription
def transcribe_audio(audio_path):
    """Transcribe audio file using faster-whisper"""
    if not os.path.isfile(audio_path):
        raise FileNotFoundError(f"Audio file not found: {audio_path}")
    
    segments, info = whisper_model.transcribe(audio_path)
    transcript = " ".join(segment.text for segment in segments)
    return transcript

# Data Extraction
def extract_data(text):
    """Extract relevant data with spaCy and regex"""
    doc = nlp(text)
    data = {}
    
    # Extract entities
    for ent in doc.ents:
        if ent.label_ == "PERSON" and "Name" not in data:
            data["Name"] = ent.text
        elif ent.label_ == "DATE" and "Date" not in data:
            data["Date"] = ent.text
        elif ent.label_ == "ORG" and "Organization" not in data:
            data["Organization"] = ent.text
        elif ent.label_ == "GPE" and "Location" not in data:
            data["Location"] = ent.text
        elif ent.label_ == "MONEY" and "Amount" not in data:
            data["Amount"] = ent.text
            
    # Phone number extraction
    phone_match = re.search(r"(\+?\d{1,2}\s?)?\(?\d{3}\)?[\s-]?\d{3}[\s-]?\d{4}", text)
    if phone_match:
        data["Phone"] = phone_match.group(0)
    
    # Email extraction
    email_match = re.search(r"[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}", text)
    if email_match:
        data["Email"] = email_match.group(0)
    
    # If date isn't found, use today's date
    if "Date" not in data:
        data["Date"] = datetime.now().strftime("%B %d, %Y")
    
    # Add more information based on the transcript content
    if "invoice" in text.lower() or "bill" in text.lower() or "payment" in text.lower():
        data["Document_Type"] = "Invoice"
        if "Amount" not in data:
            # Try to find an amount
            amount_match = re.search(r"\$\s*\d+(?:,\d{3})*(?:\.\d{2})?|\d+(?:,\d{3})*(?:\.\d{2})?\s*dollars", text)
            if amount_match:
                data["Amount"] = amount_match.group(0)
    elif "agreement" in text.lower() or "contract" in text.lower():
        data["Document_Type"] = "Agreement"
    else:
        data["Document_Type"] = "Application"
    
    return data

# Template Matching
def match_template(text, templates):
    """Match transcript to the most appropriate template"""
    query_emb = embed_model.encode(text, convert_to_tensor=True)
    scores = {}

    for key, tdata in templates.items():
        desc_emb = embed_model.encode(tdata["description"], convert_to_tensor=True)
        similarity = util.pytorch_cos_sim(query_emb, desc_emb).item()
        scores[key] = similarity

    best_match = max(scores, key=scores.get)
    return best_match

# Generate nice PDF with formatting
def generate_pdf(output_path, data_dict, doc_type="invoice"):
    """Generate a professionally formatted PDF with the extracted data"""
    doc = SimpleDocTemplate(output_path, pagesize=letter)
    styles = getSampleStyleSheet()
    elements = []
    
    # Add company logo placeholder
    # elements.append(Image("logo.png", width=2*inch, height=1*inch))
    
    # Add title based on document type
    if doc_type == "invoice":
        title = "INVOICE"
    elif doc_type == "agreement":
        title = "SERVICE AGREEMENT"
    else:
        title = "APPLICATION FORM"
        
    elements.append(Paragraph(f"<font size=20><b>{title}</b></font>", styles["Title"]))
    elements.append(Spacer(1, 20))
    
    # Add reference number
    ref_num = f"REF: {int(time.time())}"
    elements.append(Paragraph(f"<b>{ref_num}</b>", styles["Normal"]))
    elements.append(Spacer(1, 20))
    
    # Create a 2-column table for the form data
    data_table = []
    
    # Add date to table
    if "Date" in data_dict:
        data_table.append(["Date:", data_dict.get("Date", "")])
    
    # Add all extracted data to table
    for key, value in data_dict.items():
        if key != "Date" and key != "Document_Type":  # Skip already added or internal fields
            data_table.append([f"{key}:", value])
    
    # Create the table
    table = Table(data_table, colWidths=[120, 350])
    table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (0, -1), colors.lightgrey),
        ('TEXTCOLOR', (0, 0), (0, -1), colors.black),
        ('ALIGN', (0, 0), (0, -1), 'RIGHT'),
        ('ALIGN', (1, 0), (1, -1), 'LEFT'),
        ('FONTNAME', (0, 0), (0, -1), 'Helvetica-Bold'),
        ('FONTNAME', (1, 0), (1, -1), 'Helvetica'),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 6),
        ('TOPPADDING', (0, 0), (-1, -1), 6),
        ('GRID', (0, 0), (-1, -1), 0.5, colors.grey),
    ]))
    elements.append(table)
    elements.append(Spacer(1, 20))
    
    # Add specific fields based on document type
    if doc_type == "invoice":
        elements.append(Paragraph("<b>Payment Details</b>", styles["Heading2"]))
        elements.append(Paragraph("Please make payment within 30 days of the invoice date.", styles["Normal"]))
        elements.append(Spacer(1, 10))
        
        # Add payment table
        payment_data = [
            ["Payment Method", "Details"],
            ["Bank Transfer", "Account #: 12345678\nBank: Example Bank\nRouting #: 987654321"],
            ["Check", "Please mail to: 123 Business St., City, State ZIP"],
            ["Online", "www.example.com/pay"]
        ]
        
        payment_table = Table(payment_data, colWidths=[120, 350])
        payment_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (1, 0), colors.grey),
            ('TEXTCOLOR', (0, 0), (1, 0), colors.whitesmoke),
            ('ALIGN', (0, 0), (1, 0), 'CENTER'),
            ('FONTNAME', (0, 0), (1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (1, 0), 12),
            ('BOTTOMPADDING', (0, 0), (1, 0), 8),
            ('TOPPADDING', (0, 0), (1, 0), 8),
            ('GRID', (0, 0), (-1, -1), 1, colors.grey),
            ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
        ]))
        elements.append(payment_table)
        
    elif doc_type == "agreement":
        elements.append(Paragraph("<b>Agreement Terms</b>", styles["Heading2"]))
        elements.append(Paragraph("This agreement sets forth the terms and conditions of the service to be provided.", styles["Normal"]))
        elements.append(Spacer(1, 10))
        elements.append(Paragraph("1. SERVICES: The service provider agrees to perform the services as discussed.", styles["Normal"]))
        elements.append(Paragraph("2. PAYMENT: Payment shall be made according to the agreed terms.", styles["Normal"]))
        elements.append(Paragraph("3. TERM: This agreement shall commence on the date specified above.", styles["Normal"]))
        
    else:  # application
        elements.append(Paragraph("<b>Application Details</b>", styles["Heading2"]))
        elements.append(Paragraph("Thank you for your application. We will review your information and contact you shortly.", styles["Normal"]))
    
    # Add signature area
    elements.append(Spacer(1, 50))
    elements.append(Paragraph("Signature: _______________________________", styles["Normal"]))
    elements.append(Spacer(1, 20))
    elements.append(Paragraph("Date: _______________________________", styles["Normal"]))
    
    # Build PDF
    doc.build(elements)
    return output_path

# Create example templates
TEMPLATES = {
    "invoice": {
        "description": "Invoice for payment including client details, amount, and services rendered"
    },
    "agreement": {
        "description": "Service level agreement with customer details and service terms"
    },
    "application": {
        "description": "Application form with personal information including name, contact details"
    }
}

# Streamlit UI
st.title("Audio-to-PDF Processor")
st.write("Upload an audio file or record live audio to generate a filled PDF document")

# Create tabs for different input methods
tab1, tab2, tab3 = st.tabs(["Upload Audio", "Record Live Audio", "Sample Audio Files"])

with tab1:
    # Audio file upload
    uploaded_file = st.file_uploader("Upload an audio file", type=["mp3", "wav", "m4a", "ogg"])
    
    if uploaded_file is not None:
        # Save the uploaded file temporarily
        temp_dir = tempfile.mkdtemp()
        temp_path = os.path.join(temp_dir, uploaded_file.name)
        
        with open(temp_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        
        st.success("Audio file uploaded successfully.")
        st.audio(temp_path)
        
        if st.button("Process Uploaded Audio"):
            with st.spinner("Processing audio..."):
                # Progress bar
                progress_bar = st.progress(0)
                
                # 1) Transcribe
                progress_bar.progress(20)
                transcript = transcribe_audio(temp_path)
                st.write("### Transcript:")
                st.write(transcript)
                
                # 2) Extract Data
                progress_bar.progress(40)
                extracted = extract_data(transcript)
                st.write("### Extracted Data:")
                st.json(extracted)
                
                # Save JSON to file
                progress_bar.progress(60)
                json_path = os.path.join(temp_dir, "extracted_data.json")
                with open(json_path, "w") as f:
                    json.dump(extracted, f, indent=4)
                
                # Offer JSON download
                with open(json_path, "r") as f:
                    st.download_button(
                        "Download JSON data",
                        f,
                        file_name="extracted_data.json",
                        mime="application/json"
                    )
                
                # 3) Pick Template
                progress_bar.progress(80)
                doc_type = extracted.get("Document_Type", "application").lower()
                st.write(f"### Document Type: {doc_type}")
                
                # 4) Generate PDF
                output_path = os.path.join(temp_dir, f"filled_{os.path.basename(temp_path)}.pdf")
                
                try:
                    progress_bar.progress(90)
                    generate_pdf(output_path, extracted, doc_type)
                    progress_bar.progress(100)
                    st.success(f"PDF created successfully!")
                    
                    # Display download link
                    with open(output_path, "rb") as f:
                        st.download_button(
                            "Download PDF",
                            f,
                            file_name=f"filled_form.pdf",
                            mime="application/pdf"
                        )
                except Exception as e:
                    st.error(f"Error generating PDF: {str(e)}")

with tab2:
    # Live audio recording
    st.write("### Record live audio")
    
    # Audio recording parameters
    col1, col2 = st.columns(2)
    with col1:
        duration = st.slider("Recording duration (seconds)", 5, 60, 10)
    with col2:
        sample_rate = st.selectbox("Sample rate", [16000, 22050, 44100], index=0)
    
    if st.button("Start Recording"):
        try:
            # Record audio
            st.write(f"Recording for {duration} seconds...")
            progress_bar = st.progress(0)
            
            # Simulate recording progress
            recording = sd.rec(int(duration * sample_rate), samplerate=sample_rate, channels=1)
            
            # Update progress bar during recording
            for i in range(100):
                time.sleep(duration/100)
                progress_bar.progress(i + 1)
            
            sd.wait()
            st.success("Recording complete!")
            
            # Save recorded audio
            temp_dir = tempfile.mkdtemp()
            audio_filename = os.path.join(temp_dir, "live_recording.wav")
            sf.write(audio_filename, recording, sample_rate)
            
            # Let user listen to the recording
            st.audio(audio_filename)
            
            # Process audio button
            if st.button("Process Recording"):
                with st.spinner("Processing audio..."):
                    # 1) Transcribe
                    transcript = transcribe_audio(audio_filename)
                    st.write("### Transcript:")
                    st.write(transcript)
                    
                    # 2) Extract Data
                    extracted = extract_data(transcript)
                    st.write("### Extracted Data:")
                    st.json(extracted)
                    
                    # Save JSON to file
                    json_path = os.path.join(temp_dir, "extracted_data.json")
                    with open(json_path, "w") as f:
                        json.dump(extracted, f, indent=4)
                    
                    # Offer JSON download
                    with open(json_path, "r") as f:
                        st.download_button(
                            "Download JSON data",
                            f,
                            file_name="extracted_data.json",
                            mime="application/json"
                        )
                    
                    # 3) Pick Template
                    doc_type = extracted.get("Document_Type", "application").lower()
                    st.write(f"### Document Type: {doc_type}")
                    
                    # 4) Generate PDF
                    output_path = os.path.join(temp_dir, "filled_form.pdf")
                    
                    try:
                        generate_pdf(output_path, extracted, doc_type)
                        st.success(f"PDF created successfully!")
                        
                        # Display download link
                        with open(output_path, "rb") as f:
                            st.download_button(
                                "Download PDF",
                                f,
                                file_name=f"filled_form.pdf",
                                mime="application/pdf"
                            )
                    except Exception as e:
                        st.error(f"Error generating PDF: {str(e)}")
        except Exception as e:
            st.error(f"Error recording audio: {str(e)}")
            st.info("Make sure you have installed required packages: pip install sounddevice soundfile")

with tab3:
    # Sample audio files
    st.write("### Sample Audio Files")
    st.write("Select a sample audio file to process:")
    
    # Sample audio options
    sample_option = st.selectbox(
        "Choose a sample audio file",
        [
            "Invoice Request",
            "Agreement Discussion",
            "Application Submission"
        ]
    )
    
    # Define sample audio content (this would be used to create AI-generated voice samples)
    sample_content = {
        "Invoice Request": "Hello, this is John Smith. I need to create an invoice for services rendered on April 10, 2025. The amount is $1,500 for web development work. Please contact me at 555-123-4567 or john.smith@example.com.",
        "Agreement Discussion": "Hi, my name is Sarah Johnson from ABC Company. We'd like to establish a service agreement beginning on May 15, 2025. Please send the contract to our office at 123 Business Street, New York.",
        "Application Submission": "Good day, I'm David Brown. I'm submitting my application for the software developer position. I can be reached at 800-555-7890. I have 5 years of experience in the field."
    }
    
    # Display the sample content
    st.write("#### Sample Content:")
    st.write(sample_content[sample_option])
    
    if st.button("Process Sample"):
        with st.spinner("Processing sample audio..."):
            # For demo purposes, we'll skip the actual audio transcription and use the text directly
            transcript = sample_content[sample_option]
            
            # Extract Data
            extracted = extract_data(transcript)
            st.write("### Extracted Data:")
            st.json(extracted)
            
            # Save JSON to file
            temp_dir = tempfile.mkdtemp()
            json_path = os.path.join(temp_dir, "extracted_data.json")
            with open(json_path, "w") as f:
                json.dump(extracted, f, indent=4)
            
            # Offer JSON download
            with open(json_path, "r") as f:
                st.download_button(
                    "Download JSON data",
                    f,
                    file_name="extracted_data.json",
                    mime="application/json"
                )
            
            # Determine document type
            doc_type = extracted.get("Document_Type", "application").lower()
            st.write(f"### Document Type: {doc_type}")
            
            # Generate PDF
            output_path = os.path.join(temp_dir, f"{doc_type}_sample.pdf")
            
            try:
                generate_pdf(output_path, extracted, doc_type)
                st.success(f"PDF created successfully!")
                
                # Display download link
                with open(output_path, "rb") as f:
                    st.download_button(
                        "Download PDF",
                        f,
                        file_name=f"{doc_type}_form.pdf",
                        mime="application/pdf"
                    )
            except Exception as e:
                st.error(f"Error generating PDF: {str(e)}")

# Setup instructions in sidebar
st.sidebar.title("Setup Instructions")
st.sidebar.markdown("""
## Required Packages
```
pip install streamlit faster-whisper spacy sentence-transformers sounddevice soundfile reportlab
python -m spacy download en_core_web_sm
```

## How to Use
1. Upload an audio file or record live audio
2. The app will transcribe and extract relevant information
3. A PDF document will be generated with the extracted data
4. Download the JSON data and PDF document

## Example Usage
Try saying: "Hello, my name is John Smith. I need an invoice for $1,500 for web development work completed on April 10, 2025."
""")

# Add information about the PDF generation method
st.sidebar.markdown("""
## PDF Generation
This app generates professional PDF documents without requiring templates. The PDFs are created using ReportLab, which allows for high-quality document generation with tables, formatting, and styling.

If you prefer to use PDF templates:
1. Create fillable PDF forms in Adobe Acrobat
2. Place them in a 'templates' directory
3. Modify the code to use pdfrw for filling the template fields
""")
