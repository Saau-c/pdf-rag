# utils.py
import pdfplumber
import re
from typing import List

# --------------------------
# PDF TEXT EXTRACTION
# --------------------------
def extract_text_from_pdf(path: str) -> str:
    """
    Extract text from a PDF using pdfplumber and clean it.
    """
    texts = []
    with pdfplumber.open(path) as pdf:
        for page in pdf.pages:
            txt = page.extract_text()
            if txt:
                texts.append(txt)

    # Join all pages
    full_text = "\n\n".join(texts)

    # Clean text: normalize spaces and line breaks
    full_text = re.sub(r"\n{2,}", "\n", full_text)       # multiple newlines -> single
    full_text = re.sub(r"[ \t]{2,}", " ", full_text)     # multiple spaces -> single
    full_text = full_text.strip()

    return full_text

# --------------------------
# SIMPLE SENTENCE SPLITTER
# --------------------------
def split_into_sentences(text: str) -> List[str]:
    """
    Simple sentence splitter using regex.
    Splits on '.', '!', '?' followed by space or line break.
    """
    sentence_endings = re.compile(r'(?<=[.!?])\s+')
    sentences = sentence_endings.split(text)
    # Remove empty strings and strip spaces
    sentences = [s.strip() for s in sentences if s.strip()]
    return sentences

# --------------------------
# CHUNK TEXT
# --------------------------
def simple_chunk_text(text: str, max_sentences: int = 5, overlap_sentences: int = 1) -> List[str]:
    """
    Break text into overlapping chunks of sentences.
    
    Args:
        text: The input string.
        max_sentences: Maximum number of sentences per chunk.
        overlap_sentences: Number of overlapping sentences between chunks.
    
    Returns:
        List of text chunks.
    """
    sentences = split_into_sentences(text)
    chunks = []

    start = 0
    while start < len(sentences):
        end = start + max_sentences
        chunk_sentences = sentences[start:end]
        chunk = " ".join(chunk_sentences).strip()
        if chunk:
            chunks.append(chunk)
        # Move start forward with overlap
        start = end - overlap_sentences
        if start < 0:
            start = 0

    return chunks
