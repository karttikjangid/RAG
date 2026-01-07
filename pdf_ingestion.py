"""
PDF Text Extraction Module

This module provides functionality to extract raw text from PDF files.
It uses the pypdf library to read PDF documents and returns clean text
from all pages combined.
"""

from pypdf import PdfReader
import os


def get_pdf_text(file_path):
    """
    Extract all text from a PDF file.
    
    Args:
        file_path (str): Path to the PDF file (e.g., "my_book.pdf")
    
    Returns:
        str: A single string containing text from all pages combined
    
    Raises:
        Returns error message string if file doesn't exist or is invalid
    
    Example:
        >>> text = get_pdf_text("sample.pdf")
        >>> print(text[:500])
    """
    
    # Error Handling: Check if file exists
    if not os.path.exists(file_path):
        return f"❌ ERROR: File not found - {file_path}"
    
    # Error Handling: Check if file is a PDF
    if not file_path.lower().endswith('.pdf'):
        return f"❌ ERROR: File is not a PDF - {file_path}"
    
    try:
        # Step 1: Create a PdfReader object
        print(f"📄 Opening PDF: {file_path}")
        reader = PdfReader(file_path)
        
        # Get number of pages
        num_pages = len(reader.pages)
        print(f"📊 Total pages: {num_pages}")
        
        # Step 2: Initialize result string
        full_text = ""
        
        # Step 3: Loop through every page
        print("🔄 Extracting text from pages...")
        for page_num, page in enumerate(reader.pages, start=1):
            # Step 4: Extract text from each page
            page_text = page.extract_text()
            
            # Append to result string
            full_text += page_text
            
            # Progress indicator (for large PDFs)
            if page_num % 10 == 0 or page_num == num_pages:
                print(f"   ✓ Processed {page_num}/{num_pages} pages")
        
        # Clean up extra whitespace
        full_text = ' '.join(full_text.split())
        
        print(f"✅ Successfully extracted {len(full_text)} characters")
        print(f"📝 Word count: {len(full_text.split())} words")
        
        return full_text
        
    except Exception as e:
        return f"❌ ERROR: Failed to read PDF - {str(e)}"


if __name__ == "__main__":
    # Test the function
    print("=" * 70)
    print("PDF Text Extraction Test")
    print("=" * 70)
    
    # You can test with your own PDF file
    # Replace this path with an actual PDF on your system
    test_pdf = "sample.pdf"
    
    # Check if test file exists, if not create a simple one or ask for path
    if not os.path.exists(test_pdf):
        print(f"\n⚠️  Test file '{test_pdf}' not found.")
        print("\nTo test this script:")
        print("1. Place a PDF file in the RAG directory, OR")
        print("2. Edit this script and change 'test_pdf' to your PDF path")
        print("\nExample usage:")
        print(">>> from pdf_ingestion import get_pdf_text")
        print(">>> text = get_pdf_text('your_file.pdf')")
        print(">>> print(text[:500])  # Print first 500 characters")
    else:
        print(f"\n📂 Testing with: {test_pdf}\n")
        
        # Extract text
        result = get_pdf_text(test_pdf)
        
        # Print results
        print("\n" + "=" * 70)
        print("EXTRACTION RESULTS")
        print("=" * 70)
        
        # Check if it's an error message
        if result.startswith("❌"):
            print(result)
        else:
            print("\n📄 First 500 characters:")
            print("-" * 70)
            print(result[:500])
            print("-" * 70)
            
            if len(result) > 500:
                print("\n📄 Last 200 characters:")
                print("-" * 70)
                print(result[-200:])
                print("-" * 70)
