#!/usr/bin/env python3
"""
Anki Converter for ScribeWise
Converts flashcards JSON to Anki-importable format
"""

import json
import os
import sys
import csv

def convert_json_to_anki(json_file, output_format="txt"):
    """Convert flashcards JSON to Anki-importable format"""
    print(f"Converting {json_file} to Anki format...")
    
    # Read the JSON file
    with open(json_file, 'r') as f:
        data = json.load(f)
    
    if 'flashcards' not in data:
        print("Error: Invalid flashcards JSON format (missing 'flashcards' key)")
        return False
    
    # Get the base name for output
    base_name = os.path.splitext(json_file)[0]
    
    if output_format.lower() == "csv":
        # CSV format
        output_file = f"{base_name}.csv"
        with open(output_file, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(["front", "back"])  # Header
            for card in data['flashcards']:
                writer.writerow([card['front'], card['back']])
    else:
        # Default: tab-delimited text format (most compatible with Anki)
        output_file = f"{base_name}.txt"
        with open(output_file, 'w') as f:
            for card in data['flashcards']:
                f.write(f"{card['front']}\t{card['back']}\n")
    
    print(f"Successfully converted to {output_file}")
    print(f"Import instructions:")
    print("1. Open Anki")
    print("2. Click 'Import File'")
    print(f"3. Select {output_file}")
    print("4. Ensure field separator is set correctly (tab for .txt, comma for .csv)")
    print("5. Make sure the fields map correctly to Anki's 'Front' and 'Back' fields")
    
    return output_file

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python anki_converter.py <flashcards_json_file> [output_format]")
        print("Output formats: txt (default), csv")
        sys.exit(1)
    
    json_file = sys.argv[1]
    output_format = sys.argv[2] if len(sys.argv) > 2 else "txt"
    
    if not os.path.exists(json_file):
        print(f"Error: File {json_file} not found")
        sys.exit(1)
    
    output_file = convert_json_to_anki(json_file, output_format)
    if not output_file:
        sys.exit(1) 