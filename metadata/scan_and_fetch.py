
"""
scan_and_fetch.py

This script scans a specified directory for .txt files, extracts the document ID from each filename,
queries the talpa.org.ua API for metadata, and collects the following fields for each file:

- txt_path: The relative path to the .txt file in your directory.
- author: The name of the community or organization that authored the document, as provided by the API.
- document_url: The original URL to the document as provided by the API.

The results are written to document_info.csv.
"""
import os
import csv
import json
import requests
import argparse
from pathlib import Path

def process_directory(directory_path):
    # Prepare output CSV file
    output_file = 'document_info.csv'
    
    # Create/open CSV file with header
    with open(output_file, 'w', newline='', encoding='utf-8') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['txt_path', 'author', 'document_url'])
        
        # Scan directory for .txt files
        for file in Path(directory_path).glob('*.txt'):
            try:
                # Get document ID from filename
                doc_id = file.stem  # Gets filename without extension
                
                # Construct API URL
                api_url = f'https://talpa.org.ua/api/documents/{doc_id}'
                
                # Fetch JSON data
                response = requests.get(api_url)
                response.raise_for_status()  # Raise exception for error status codes
                data = response.json()
                
                # Extract required information
                author = data['document']['author']['name']
                doc_url = data['document']['url']
                
                # Write to CSV
                writer.writerow([str(file), author, doc_url])
                
                print(f"Processed {file.name} successfully")
                
            except Exception as e:
                print(f"Error processing {file.name}: {str(e)}")

def main():
    parser = argparse.ArgumentParser(description='Scan directory for .txt files and fetch document information')
    parser.add_argument('directory', help='Directory path to scan for .txt files')
    
    args = parser.parse_args()
    
    if not os.path.isdir(args.directory):
        print(f"Error: {args.directory} is not a valid directory")
        return
    
    process_directory(args.directory)
    print("Processing complete!")

if __name__ == "__main__":
    main()
