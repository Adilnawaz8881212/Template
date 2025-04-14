import os

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

def process_audio(audio_path: str):
    # 1) Transcribe
    transcript = transcribe_audio(audio_path)
    print("\n>> Transcript:\n", transcript)

    # 2) Extract Data
    extracted = extract_data(transcript)
    print("\n>> Extracted Data:\n", extracted)

    # 3) Pick Template
    chosen = match_template(transcript, TEMPLATES)
    print("\n>> Matched Template:\n", chosen)

    # 4) Fill PDF
    template_path = TEMPLATES[chosen]["path"]
    base_name = os.path.splitext(os.path.basename(audio_path))[0]
    output_path = f"output/filled_{base_name}.pdf"
    fill_pdf(template_path, output_path, extracted)
    print(f"\n>> PDF created at: {output_path}\n")

def main():
    os.makedirs("output", exist_ok=True)
    audio_path = "audio/sample_audio.mp3"
    process_audio(audio_path)

if __name__ == "__main__":
    main()
