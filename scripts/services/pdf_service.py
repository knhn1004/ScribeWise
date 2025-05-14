"""
PDF Service for ScribeWise
- Handles PDF document upload and processing
- Extracts text and chunks for analysis
- Integrates with summarization service
"""

import os
import tempfile
import logging
from typing import Dict, List, Optional, Tuple, Any
import uuid
import json
from pathlib import Path

# For PDF processing
try:
    import fitz  # PyMuPDF
    PYMUPDF_AVAILABLE = True
except ImportError:
    PYMUPDF_AVAILABLE = False
    print("Warning: PyMuPDF not available, PDF processing will be disabled")

# For text splitting
from langchain_text_splitters import RecursiveCharacterTextSplitter

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class PDFService:
    def __init__(self, output_dir: str = "outputs", download_dir: str = "downloads"):
        """Initialize the PDF Service"""
        self.output_dir = output_dir
        self.download_dir = download_dir
        
        # Ensure directories exist
        os.makedirs(output_dir, exist_ok=True)
        os.makedirs(download_dir, exist_ok=True)
        
        # Initialize text splitter for chunking
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=2000,
            chunk_overlap=200,
            separators=["\n\n", "\n", ". ", " ", ""]
        )
        
        if not PYMUPDF_AVAILABLE:
            logger.warning("PyMuPDF not available. Install with: pip install pymupdf")
    
    def _check_pdf_dependencies(self):
        """Check if required PDF processing dependencies are available"""
        if not PYMUPDF_AVAILABLE:
            raise ImportError("PyMuPDF is required for PDF processing. Install with: pip install pymupdf")
        return True
    
    async def save_uploaded_pdf(self, file_data: bytes, filename: Optional[str] = None) -> Dict[str, str]:
        """Save an uploaded PDF file and return file info
        
        Args:
            file_data: Raw bytes of the PDF file
            filename: Optional filename (will generate one if not provided)
            
        Returns:
            Dictionary with file path and ID
        """
        self._check_pdf_dependencies()
        
        # Generate unique ID and filename if not provided
        pdf_id = str(uuid.uuid4())
        if not filename:
            filename = f"{pdf_id}.pdf"
        elif not filename.lower().endswith('.pdf'):
            filename = f"{filename}.pdf"
            
        # Sanitize filename
        safe_filename = "".join(c for c in filename if c.isalnum() or c in "._- ").replace(" ", "_")
        file_path = os.path.join(self.download_dir, safe_filename)
        
        # Save the file
        with open(file_path, "wb") as f:
            f.write(file_data)
            
        logger.info(f"Saved uploaded PDF to {file_path}")
        
        # Get basic file info
        file_size = os.path.getsize(file_path)
        
        return {
            "pdf_path": file_path,
            "pdf_id": pdf_id,
            "filename": safe_filename,
            "file_size": file_size
        }
    
    async def extract_text_from_pdf(self, pdf_path: str) -> Tuple[str, Dict[str, Any]]:
        """Extract text and metadata from a PDF file
        
        Args:
            pdf_path: Path to the PDF file
            
        Returns:
            Tuple of (full_text, metadata)
        """
        self._check_pdf_dependencies()
        
        if not os.path.exists(pdf_path):
            raise FileNotFoundError(f"PDF file not found: {pdf_path}")
        
        full_text = ""
        metadata = {
            "page_count": 0,
            "title": None,
            "author": None,
            "creation_date": None,
            "page_texts": []
        }
        
        try:
            # Open the PDF
            doc = fitz.open(pdf_path)
            
            # Extract metadata
            metadata["page_count"] = len(doc)
            metadata["title"] = doc.metadata.get("title", "Untitled")
            metadata["author"] = doc.metadata.get("author", "Unknown")
            metadata["creation_date"] = doc.metadata.get("creationDate", None)
            
            # Extract text from each page
            for page_num, page in enumerate(doc):
                page_text = page.get_text()
                full_text += page_text + "\n\n"
                metadata["page_texts"].append({
                    "page_num": page_num + 1,
                    "text_length": len(page_text),
                    "has_text": len(page_text.strip()) > 0
                })
                
            logger.info(f"Extracted {len(full_text)} characters from {metadata['page_count']} pages in {pdf_path}")
            
        except Exception as e:
            logger.error(f"Error extracting text from PDF: {str(e)}")
            raise Exception(f"Error extracting text from PDF: {str(e)}")
            
        return full_text, metadata
    
    async def chunk_text(self, text: str) -> List[str]:
        """Split text into manageable chunks for analysis
        
        Args:
            text: The full text to split
            
        Returns:
            List of text chunks
        """
        if not text.strip():
            return []
            
        chunks = self.text_splitter.split_text(text)
        logger.info(f"Split text into {len(chunks)} chunks")
        
        return chunks
    
    async def save_intermediate_results(self, pdf_id: str, chunk_results: List[Dict], result_type: str) -> str:
        """Save intermediate results from chunk processing
        
        Args:
            pdf_id: The PDF identifier
            chunk_results: List of results from each chunk
            result_type: Type of result (notes, flashcards, etc.)
            
        Returns:
            Path to the saved file
        """
        # Create PDF-specific directory
        pdf_dir = os.path.join(self.output_dir, pdf_id)
        os.makedirs(pdf_dir, exist_ok=True)
        
        # Save to intermediate file
        intermediate_path = os.path.join(pdf_dir, f"intermediate_{result_type}.json")
        with open(intermediate_path, "w") as f:
            json.dump(chunk_results, f, indent=2)
            
        logger.info(f"Saved intermediate {result_type} results to {intermediate_path}")
        return intermediate_path
    
    async def combine_notes(self, chunk_notes: List[str]) -> str:
        """Combine notes from multiple chunks into a single document
        
        Args:
            chunk_notes: List of markdown notes from each chunk
            
        Returns:
            Combined markdown notes
        """
        if not chunk_notes:
            return ""
            
        # Simple combination with section dividers
        combined = "# Combined Notes\n\n"
        
        for i, notes in enumerate(chunk_notes):
            combined += f"## Section {i+1}\n\n"
            combined += notes.strip()
            combined += "\n\n---\n\n"
            
        logger.info(f"Combined {len(chunk_notes)} chunks of notes")
        return combined
    
    async def combine_flashcards(self, chunk_flashcards: List[Dict]) -> Dict:
        """Combine flashcards from multiple chunks and deduplicate
        
        Args:
            chunk_flashcards: List of flashcard JSON from each chunk
            
        Returns:
            Combined flashcards as a single JSON object
        """
        if not chunk_flashcards:
            return {"flashcards": []}
            
        # Extract all flashcards
        all_cards = []
        for chunk in chunk_flashcards:
            if isinstance(chunk, str):
                # Parse JSON if it's a string
                try:
                    chunk_data = json.loads(chunk)
                    if "flashcards" in chunk_data:
                        all_cards.extend(chunk_data["flashcards"])
                except json.JSONDecodeError:
                    logger.warning(f"Could not parse flashcard JSON: {chunk[:100]}...")
            elif isinstance(chunk, dict) and "flashcards" in chunk:
                all_cards.extend(chunk["flashcards"])
                
        # Remove duplicates (based on front side)
        unique_cards = {}
        for card in all_cards:
            front = card.get("front", "").strip()
            if front and front not in unique_cards:
                unique_cards[front] = card
                
        logger.info(f"Combined and deduplicated flashcards: {len(unique_cards)} unique cards from {len(all_cards)} total")
        
        return {"flashcards": list(unique_cards.values())}
    
    async def process_pdf(self, pdf_path: str, summarization_service) -> Dict[str, Any]:
        """Process a PDF file and generate summarized content
        
        Args:
            pdf_path: Path to the PDF file
            summarization_service: Instance of SummarizationService
            
        Returns:
            Dictionary with all generated content and paths
        """
        # Extract the PDF ID from the filename or path
        pdf_id = Path(pdf_path).stem
        
        # Extract text from PDF
        full_text, metadata = await self.extract_text_from_pdf(pdf_path)
        
        # Process the entire document for mindmap
        logger.info("Processing entire document for mindmap...")
        full_doc_content = {
            "text": full_text,
            "pdf_id": pdf_id,
            "title": metadata.get("title", f"PDF {pdf_id}")
        }
        
        mindmap_result = await summarization_service.process_content_for_mindmap(full_doc_content)
        
        # Chunk the text for notes and flashcards
        chunks = await self.chunk_text(full_text)
        
        # Process each chunk
        chunk_notes = []
        chunk_flashcards = []
        
        for i, chunk in enumerate(chunks):
            logger.info(f"Processing chunk {i+1}/{len(chunks)}...")
            chunk_content = {
                "text": chunk,
                "pdf_id": pdf_id,
                "chunk_id": i+1,
                "title": f"{metadata.get('title', 'PDF')} - Chunk {i+1}"
            }
            
            # Generate notes and flashcards for the chunk
            chunk_result = await summarization_service.process_content_for_chunk(chunk_content)
            
            if "notes" in chunk_result:
                chunk_notes.append(chunk_result["notes"])
                
            if "flashcards" in chunk_result:
                chunk_flashcards.append(chunk_result["flashcards"])
        
        # Save intermediate results
        await self.save_intermediate_results(pdf_id, chunk_notes, "notes")
        await self.save_intermediate_results(pdf_id, chunk_flashcards, "flashcards")
        
        # Combine results
        combined_notes = await self.combine_notes(chunk_notes)
        combined_flashcards = await self.combine_flashcards(chunk_flashcards)
        
        # Save combined results
        pdf_dir = os.path.join(self.output_dir, pdf_id)
        notes_path = os.path.join(pdf_dir, f"{pdf_id}_notes.md")
        flashcards_path = os.path.join(pdf_dir, f"{pdf_id}_flashcards.json")
        
        with open(notes_path, "w") as f:
            f.write(combined_notes)
            
        with open(flashcards_path, "w") as f:
            json.dump(combined_flashcards, f, indent=2)
        
        # Create Anki package if possible
        anki_path = os.path.join(pdf_dir, f"{pdf_id}_flashcards.apkg")
        anki_result = summarization_service._create_anki_package(
            json.dumps(combined_flashcards), 
            metadata.get("title", f"ScribeWise {pdf_id}"),
            anki_path
        )
        
        # Compile final result
        result = {
            "pdf_info": {
                "pdf_id": pdf_id,
                "pdf_path": pdf_path,
                "title": metadata.get("title", "Untitled"),
                "page_count": metadata.get("page_count", 0)
            },
            "notes": {
                "combined_notes": combined_notes,
                "notes_path": notes_path,
                "chunk_count": len(chunk_notes)
            },
            "flashcards": {
                "combined_flashcards": combined_flashcards,
                "flashcards_path": flashcards_path,
                "total_cards": len(combined_flashcards.get("flashcards", [])),
                "anki_path": anki_path if anki_result else None
            }
        }
        
        # Add mindmap results
        if mindmap_result:
            result["mindmap"] = mindmap_result
        
        logger.info(f"PDF processing complete: {pdf_id}")
        return result 