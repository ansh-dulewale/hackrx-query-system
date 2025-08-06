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
        A list of text chunks, or an empty list if processing fails.
    """
    print(f"Processing document from: {url}")
    
    # Add a User-Agent header to mimic a browser request
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
    }

    try:
        # Make the request with the headers and a timeout
        response = requests.get(url, headers=headers, timeout=10)
        response.raise_for_status()  # Raises an exception for bad status codes (like 404 Not Found)

        # Use BytesIO to handle the PDF content in memory
        pdf_file = BytesIO(response.content)
        
        reader = PdfReader(pdf_file)
        # Ensure the PDF has pages before trying to extract text
        if not reader.pages:
            print("Error: PDF file is empty or corrupted.")
            return []

        full_text = "".join(page.extract_text() for page in reader.pages if page.extract_text())

        # Simple chunking strategy: 1000 characters per chunk with 200 overlap
        chunk_size = 1000
        overlap = 200
        chunks = [full_text[i:i+chunk_size] for i in range(0, len(full_text), chunk_size - overlap)]
        
        print(f"Document processed into {len(chunks)} chunks.")
        return chunks

    except requests.exceptions.RequestException as e:
        # This will catch connection errors, timeouts, and bad status codes
        print(f"Error downloading or accessing the document: {e}")
        return []
    except Exception as e:
        # Catch other potential errors, e.g., with pypdf
        print(f"An error occurred during PDF processing: {e}")
        return []

if __name__ == '__main__':
    # Example usage for testing this module directly
    test_url = "https://hackrx.blob.core.windows.net/assets/policy.pdf?sv-2023-01-03&st="
    text_chunks = process_pdf_from_url(test_url)
    if text_chunks:
        print("First chunk:")
        print(text_chunks[0])
    else:
        print("Failed to process the test URL.")
