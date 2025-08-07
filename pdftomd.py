import os
import re
import json
import pandas as pd
import fitz  # PyMuPDF
import pymupdf4llm
import nltk
from pathlib import Path

# Ensure NLTK punkt tokenizer is available
try:
    nltk.data.find('tokenizers/punkt_tab')
except LookupError:
    nltk.download('punkt_tab')

# Define directories and metadata file
RAW_DIR = 'data/raw/'
PROCESSED_DIR = 'data/processed/'
CLAUSES_DIR = 'data/clauses/'
METADATA_FILE = 'data/metadata.csv'

# Function to update metadata with page counts (1.3)
def update_metadata():
    if not os.path.exists(METADATA_FILE):
        print("Metadata file not found.")
        return
    df = pd.read_csv(METADATA_FILE)
    for row in df.itertuples():
        pdf_path = os.path.join(RAW_DIR, row.domain, row.filename)
        if not os.path.exists(pdf_path):
            print(f"PDF not found: {pdf_path}")
            continue
        try:
            doc = fitz.open(pdf_path)
            df.at[row.Index, 'page_count'] = len(doc)
            doc.close()
        except Exception as e:
            print(f"Error processing {pdf_path}: {e}")
    df.to_csv(METADATA_FILE, index=False)
    print("Metadata updated with page counts.")

# Function to clean markdown text
def clean_markdown(text):
    if not text:
        return ''
    # Softer cleaning: only remove exact page number patterns
    patterns = [r'^Page \d+$', r'^-\s*\d+\s*-$']
    lines = text.split('\n')
    cleaned_lines = [line for line in lines if not any(re.match(p, line.strip()) for p in patterns)]
    return '\n'.join(cleaned_lines).strip()

# Function to extract markdown and save (1.4)
def extract_and_save_markdown(pdf_path, output_path, source_file):
    try:
        # Extract markdown with PyMuPDF4LLM
        md_text = pymupdf4llm.to_markdown(pdf_path)
        # Clean the extracted markdown
        cleaned_md = clean_markdown(md_text)
        if not cleaned_md:
            print(f"Warning: No content after cleaning for {source_file}")
        # Save to output path
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        Path(output_path).write_text(cleaned_md, encoding='utf-8')
        print(f"Extracted and cleaned {source_file} -> {output_path}")
    except Exception as e:
        print(f"Error processing {source_file}: {e}")

# Function to segment markdown into clauses (1.5)
def segment_clauses(md_path, source_file, pdf_path):
    # Get page breaks using PyMuPDF for accurate page tracking
    doc = fitz.open(pdf_path)
    page_texts = [page.get_text() for page in doc]
    doc.close()
    
    with open(md_path, 'r', encoding='utf-8') as f:
        lines = f.read().split('\n')
    
    clauses = []
    current_clause = None
    clause_text = []
    clause_page = 1
    # Relaxed header pattern to match various formats
    header_pattern = r'^(#+\s*)?(Section|Article|Clause|Paragraph)?\s*(\d+(\.\d+)*)\s*(.*)?$'
    
    for i, line in enumerate(lines):
        # Check for page boundary (approximate, based on text length)
        if i > 0 and sum(len(l) for l in lines[:i]) > sum(len(t) for t in page_texts[:clause_page]):
            clause_page += 1
        header_match = re.match(header_pattern, line, re.IGNORECASE)
        if header_match:
            # Save previous clause
            if current_clause and clause_text:
                clause_id = f"{source_file}_clause{current_clause}"
                clause_content = '\n'.join(clause_text).strip()
                if clause_content:
                    print(f"Clause found: {clause_id} on page {clause_page}")
                    clauses.append({
                        'clause_id': clause_id,
                        'text': clause_content,
                        'source_document': source_file,
                        'section': current_clause,
                        'page_number': clause_page
                    })
                else:
                    print(f"Empty clause skipped: {clause_id}")
            # Start new clause
            current_clause = header_match.group(3)  # e.g., "3.2"
            clause_text = []
            print(f"Detected header: {line} -> Section {current_clause}")
        elif current_clause:
            clause_text.append(line)
    
    # Save the last clause
    if current_clause and clause_text:
        clause_id = f"{source_file}_clause{current_clause}"
        clause_content = '\n'.join(clause_text).strip()
        if clause_content:
            print(f"Clause found: {clause_id} on page {clause_page}")
            clauses.append({
                'clause_id': clause_id,
                'text': clause_content,
                'source_document': source_file,
                'section': current_clause,
                'page_number': clause_page
            })
        else:
            print(f"Empty clause skipped: {clause_id}")
    
    if not clauses:
        print(f"No clauses found for {source_file}")
    return clauses

# Main execution
if __name__ == '__main__':
    # Step 1.3: Update metadata
    update_metadata()
    
    # Get all domains from raw directory
    domains = [d for d in os.listdir(RAW_DIR) if os.path.isdir(os.path.join(RAW_DIR, d))]
    
    # Process each domain for steps 1.4 and 1.5
    for domain in domains:
        domain_clauses = []
        domain_raw_dir = os.path.join(RAW_DIR, domain)
        for filename in os.listdir(domain_raw_dir):
            if filename.lower().endswith('.pdf'):
                source_file = os.path.splitext(filename)[0]
                pdf_path = os.path.join(domain_raw_dir, filename)
                output_md = os.path.join(PROCESSED_DIR, domain, f"{source_file}.md")
                
                # Step 1.4: Extract and clean markdown
                extract_and_save_markdown(pdf_path, output_md, source_file)
                
                # Step 1.5: Segment into clauses
                if os.path.exists(output_md):
                    clauses = segment_clauses(output_md, source_file, pdf_path)
                    domain_clauses.extend(clauses)
                else:
                    print(f"Markdown file not found: {output_md}")
        
        # Save clauses to JSON file
        clauses_file = os.path.join(CLAUSES_DIR, f"{domain}_clauses.json")
        os.makedirs(CLAUSES_DIR, exist_ok=True)
        if domain_clauses:
            with open(clauses_file, 'w', encoding='utf-8') as f:
                json.dump(domain_clauses, f, ensure_ascii=False, indent=4)
            print(f"Clauses for {domain} saved to {clauses_file}")
        else:
            print(f"No clauses saved for {domain}")