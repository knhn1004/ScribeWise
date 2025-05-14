#!/usr/bin/env python3
"""
Anki Converter for ScribeWise
Converts flashcards JSON to Anki-importable formats
"""

import json
import os
import sys
import csv
import hashlib

def convert_json_to_anki(json_file, output_format="txt", deck_name=None):
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
    
    # Determine output format
    if output_format.lower() == "apkg":
        try:
            import genanki
        except ImportError:
            print("Error: To create .apkg files, install genanki:")
            print("    pip install genanki")
            return False
            
        # Create an Anki package
        return create_anki_package(data['flashcards'], base_name, deck_name)
    elif output_format.lower() == "csv":
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

def create_anki_package(flashcards, base_name, deck_name=None):
    """Create an Anki .apkg file using genanki"""
    import genanki
    
    # Use base name for deck name if not provided
    if not deck_name:
        deck_name = os.path.basename(base_name).replace("_flashcards", "").replace("_", " ").title()
    
    # Generate a consistent model ID from the deck name
    model_id = int(hashlib.md5(deck_name.encode('utf-8')).hexdigest()[:8], 16)
    deck_id = model_id + 1  # Just ensure it's different from model_id
    
    # Create the model (note type)
    model = genanki.Model(
        model_id,
        'ScribeWise Basic',
        fields=[
            {'name': 'Question'},
            {'name': 'Answer'},
        ],
        templates=[
            {
                'name': 'Card',
                'qfmt': '{{Question}}',
                'afmt': '{{FrontSide}}<hr id="answer">{{Answer}}',
            },
        ],
    )
    
    # Create the deck
    deck = genanki.Deck(deck_id, deck_name)
    
    # Add cards to the deck
    for card in flashcards:
        note = genanki.Note(
            model=model,
            fields=[card['front'], card['back']]
        )
        deck.add_note(note)
    
    # Create the package and save it
    output_file = f"{base_name}.apkg"
    genanki.Package(deck).write_to_file(output_file)
    
    print(f"Successfully created Anki deck: {output_file}")
    print(f"Import instructions:")
    print("1. Open Anki")
    print("2. Click 'File' > 'Import'")
    print(f"3. Select {output_file}")
    
    return output_file

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python anki_converter.py <flashcards_json_file> [output_format] [deck_name]")
        print("Output formats: txt (default), csv, apkg")
        print("Example: python anki_converter.py outputs/video_flashcards.json apkg 'My Deck Name'")
        sys.exit(1)
    
    json_file = sys.argv[1]
    output_format = sys.argv[2] if len(sys.argv) > 2 else "txt"
    deck_name = sys.argv[3] if len(sys.argv) > 3 else None
    
    if not os.path.exists(json_file):
        print(f"Error: File {json_file} not found")
        sys.exit(1)
    
    output_file = convert_json_to_anki(json_file, output_format, deck_name)
    if not output_file:
        sys.exit(1) 