# utils/document_processor.py
import PyPDF2
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from utils.config import CHUNK_SIZE, CHUNK_OVERLAP
import re
import os
from typing import List


def process_pdf(file_path: str) -> List[Document]:
    """
    Processes a PDF file, splits it into structural chunks, and extracts metadata.
    It attempts to divide the document by logical sections before splitting it into fixed-size chunks.

    Args:
        file_path: The path to the PDF file.

    Returns:
        A list of LangChain Document objects, each representing a chunk.
    """
    print(f"Processing file with structural chunking: {file_path}")
    doc_chunks = []
    file_name = os.path.basename(file_path)

    try:
        with open(file_path, 'rb') as file:
            pdf_reader = PyPDF2.PdfReader(file)

            # 1. Extract all text and map text to its page number
            pages_content = []
            full_text = ""
            for page_num, page in enumerate(pdf_reader.pages):
                text = page.extract_text()
                if text:
                    pages_content.append({"text": text, "page": page_num + 1})
                    full_text += text + "\n"

            # 2. Split the text into logical sections using a regex for titles.
            # This heuristic looks for lines that are mostly uppercase, short, and end with a newline.
            # It may need adjustments for different policy formats.
            # Examples: "ELIGIBILITY", "SECTION 1: DEFINITIONS", "Benefit Reductions"
            section_pattern = r"(?m)^([A-Z][A-Z0-9\s\-:]{5,})\n"
            sections = re.split(section_pattern, full_text)

            # re.split returns [text_before, title1, text1, title2, text2, ...]
            # We process it in (title, content) pairs
            if len(sections) > 1:
                # The first element is the text before the first title (header/intro)
                content_list = [("Introduction", sections[0])]
                content_list.extend(zip(sections[1::2], sections[2::2]))
            else:
                # If no sections were found, treat the entire document as a single one
                content_list = [("General", full_text)]

            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=CHUNK_SIZE,
                chunk_overlap=CHUNK_OVERLAP,
                separators=["\n\n", "\n", ". ", " ", ""],
                length_function=len
            )

            chunk_id_counter = 0
            # 3. Process each section
            for section_title, section_text in content_list:
                section_title = section_title.strip()
                if not section_text.strip():
                    continue

                chunks = text_splitter.split_text(section_text)

                for i, chunk_text in enumerate(chunks):
                    # Approximate the original page of the chunk by searching for its content
                    page_num = "N/A"
                    for page_info in pages_content:
                        # We search for a portion of the chunk in the page's text
                        if chunk_text[:max(50, len(chunk_text))] in page_info["text"]:
                            page_num = page_info["page"]
                            break

                    metadata = {
                        "source": file_name,
                        "page": page_num,
                        "chunk_id": f"chunk_{chunk_id_counter}",
                        "section": section_title,  # New important metadata!
                    }
                    chunk_id_counter += 1

                    doc_chunk = Document(page_content=chunk_text, metadata=metadata)
                    doc_chunks.append(doc_chunk)

        print(f"Document processed. Generated {len(doc_chunks)} chunks from {len(content_list)} sections.")
        return doc_chunks

    except FileNotFoundError:
        print(f"Error: The file was not found at path {file_path}")
        return []
    except Exception as e:
        print(f"An unexpected error occurred while processing the PDF: {e}")
        return []