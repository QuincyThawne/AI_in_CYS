import os
from PyPDF2 import PdfReader

def extract_and_split_pdf(pdf_path, output_folder, split_keyword):
    """
    Extracts text from a PDF and splits it into .txt files based on a given keyword.

    :param pdf_path: Path to the PDF file
    :param output_folder: Folder to save the .txt files
    :param split_keyword: Keyword to split the text on
    """
    # Create output folder if it doesn't exist
    os.makedirs(output_folder, exist_ok=True)

    # Read the PDF
    reader = PdfReader(pdf_path)
    full_text = ""
    
    # Extract text from each page
    for page in reader.pages:
        full_text += page.extract_text()

    # Split text based on the keyword
    sections = full_text.split(split_keyword)

    # Save each section as a separate .txt file
    for i, section in enumerate(sections):
        # Skip empty sections
        if not section.strip():
            continue

        # Define the filename for each split
        filename = os.path.join(output_folder, f"Expt_{i}.txt")
        with open(filename, "w", encoding="utf-8") as file:
            # Add the split keyword back at the beginning of each section (if not the first)
            if i > 0:
                file.write(split_keyword + "\n")
            file.write(section.strip())
        
        print(f"Saved: {filename}")

if __name__ == "__main__":
    # User input for PDF file and keyword
    pdf_path = r"C:\Users\Admin\Downloads\AIinCYS\22CY904 Lab Manual.pdf"
    output_folder = "experiments of AI"
    split_keyword = "Exp No"

    # Extract and split the PDF
    try:
        extract_and_split_pdf(pdf_path, output_folder, split_keyword)
        print("PDF split successfully!")
    except Exception as e:
        print(f"An error occurred: {e}")
