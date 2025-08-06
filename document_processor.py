import os
import requests
from pypdf import PdfReader
from io import BytesIO

def process_pdf_from_url(url: str) -> list[str]:
    """
    Downloads a PDF from a URL, extracts text, and splits it into chunks.

    Args:
        url: The URL of the PDF document.

    Returns:
        A list of text chunks.
    """
    print(f"Processing document from: {url}")
    try:
        response = requests.get(url)
        response.raise_for_status()  # Raises an exception for bad status codes

        # Use BytesIO to handle the PDF content in memory
        pdf_file = BytesIO(response.content)
        
        reader = PdfReader(pdf_file)
        full_text = "".join(page.extract_text() for page in reader.pages if page.extract_text())

        # Simple chunking strategy: 1000 characters per chunk with 200 overlap
        chunk_size = 1000
        overlap = 200
        chunks = [full_text[i:i+chunk_size] for i in range(0, len(full_text), chunk_size - overlap)]
        
        print(f"Document processed into {len(chunks)} chunks.")
        return chunks

    except requests.exceptions.RequestException as e:
        print(f"Error downloading the document: {e}")
        return []

if __name__ == '__main__':
    # Example usage for testing this module directly
    test_url = "https://hackrx.blob.core.windows.net/assets/policy.pdf?sv-2023-01-03&st="
    text_chunks = process_pdf_from_url(test_url)
    if text_chunks:
        print("First chunk:")
        print(text_chunks[0])
