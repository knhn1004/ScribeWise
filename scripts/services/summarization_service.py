"""
Summarization Service for ScribeWise
- Uses LLMs (llama3.2) with Langchain agents
- Generates structured outputs (notes, flashcards, mindmaps)
- Implements Retrieval Augmented Generation (RAG) for enhanced quality
"""

import os
import json
import asyncio
from typing import Dict, List, Optional, Any
from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import PydanticOutputParser, StrOutputParser
from langchain_core.language_models import BaseLLM
from langchain_core.documents import Document
from langchain_community.tools import BaseTool, StructuredTool, tool
from langchain_text_splitters.character import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from pydantic import BaseModel, Field
from dotenv import load_dotenv
import markdown

# Load environment variables
load_dotenv()

# API keys
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")


# Define output schemas
class FlashcardModel(BaseModel):
    """Model for an Anki flashcard"""

    front: str = Field(
        description="Front side of the flashcard with the question or concept"
    )
    back: str = Field(
        description="Back side of the flashcard with the answer or explanation"
    )


class FlashcardOutput(BaseModel):
    """Model for a list of flashcards"""

    flashcards: List[FlashcardModel] = Field(description="List of generated flashcards")


class MindmapNode(BaseModel):
    """Model for a node in a mindmap"""

    id: str = Field(description="Unique identifier for the node")
    text: str = Field(description="Text content of the node")
    parent_id: Optional[str] = Field(description="Parent node ID (None for root)")


class MindmapOutput(BaseModel):
    """Model for a complete mindmap"""

    title: str = Field(description="Title of the mindmap")
    nodes: List[MindmapNode] = Field(description="List of nodes in the mindmap")


class SummarizationService:
    def __init__(self, output_dir: str = "outputs"):
        """Initialize the Summarization Service with LLM and Langchain components"""
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)

        # Initialize Groq LLM
        if not GROQ_API_KEY:
            print(
                "ERROR: GROQ_API_KEY not found in environment variables. Summarization will not work."
            )
            self.llm = None
        else:
            # Use explicit Groq LLM with specific model
            try:
                # Initialize ChatGroq - it will automatically use the GROQ_API_KEY environment variable
                self.llm = ChatGroq(
                    model="llama3-70b-8192",  # Using Llama 3 70B model via Groq
                    temperature=0.2,  # Lower temperature for more factual responses
                    max_tokens=4000,  # Allow for longer generations
                    top_p=0.9,
                )
                print(
                    f"Successfully initialized Groq Chat LLM with model: llama3-70b-8192"
                )
            except Exception as e:
                print(f"Error initializing Groq Chat LLM: {str(e)}")
                self.llm = None

        # Text splitter for RAG
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=2000, chunk_overlap=200
        )

        # Output parsers
        self.flashcard_parser = PydanticOutputParser(pydantic_object=FlashcardOutput)
        self.mindmap_parser = PydanticOutputParser(pydantic_object=MindmapOutput)

        # Initialize markdown converter
        self.md = markdown.Markdown(extensions=["extra"])

    async def prepare_rag(self, text: str) -> Optional[FAISS]:
        """Prepare a RAG system from the given text"""
        if not OPENAI_API_KEY:
            print("Warning: OPENAI_API_KEY not found, RAG functionality limited")
            return None

        # Split text into chunks
        chunks = self.text_splitter.split_text(text)
        documents = [Document(page_content=chunk) for chunk in chunks]

        # Create vector store
        embeddings = OpenAIEmbeddings(api_key=OPENAI_API_KEY)
        vectorstore = FAISS.from_documents(documents, embeddings)

        return vectorstore

    async def _ensure_llm(self) -> bool:
        """Ensure LLM is initialized and working"""
        if not self.llm:
            print("Error: Groq LLM not initialized (missing API key)")
            return False
        return True

    async def process_content(self, content: Dict) -> Dict[str, Any]:
        """Process video content (transcription and optionally images) to generate all outputs"""
        # Check if LLM is available
        if not await self._ensure_llm():
            return {"error": "Groq LLM not initialized (missing API key)"}

        # Extract text from content
        text = ""
        if "text" in content and content["text"]:
            text = content["text"]
        elif "segments" in content and content["segments"]:
            # Combine segments into text
            text = " ".join(
                [
                    segment["text"]
                    for segment in content["segments"]
                    if "text" in segment
                ]
            )

        # Check if we have valid text to process
        if not text or len(text.strip()) < 10:  # Require at least 10 characters
            return {
                "error": "No valid text content found in input or text is too short"
            }

        # Prepare RAG if possible
        vectorstore = await self.prepare_rag(text)

        # Generate content directly without using @tool methods which are causing issues
        try:
            notes = self._generate_notes_direct(text)
            flashcards = self._generate_flashcards_direct(text)
            mindmap = self._generate_mindmap_direct(text)
        except Exception as e:
            error_msg = f"Error during content generation: {str(e)}"
            print(error_msg)
            return {"error": error_msg}

        # Save outputs to files
        if "video_id" in content:
            video_id = content["video_id"]
        else:
            video_id = "unknown"

        notes_path = f"{self.output_dir}/{video_id}_notes.md"
        flashcards_path = f"{self.output_dir}/{video_id}_flashcards.json"
        mindmap_path = f"{self.output_dir}/{video_id}_mindmap.md"

        with open(notes_path, "w") as f:
            f.write(notes)
        with open(flashcards_path, "w") as f:
            f.write(flashcards)
        with open(mindmap_path, "w") as f:
            f.write(mindmap)

        return {
            "notes": notes,
            "notes_path": notes_path,
            "flashcards": flashcards,
            "flashcards_path": flashcards_path,
            "mindmap": mindmap,
            "mindmap_path": mindmap_path,
        }
        
    def _generate_notes_direct(self, text: str) -> str:
        """Direct version of generate_notes without @tool decorator"""
        if not self.llm:
            return "Error: Groq LLM not initialized (missing API key)"
            
        # Safety check
        if not isinstance(text, str):
            return "Error: Invalid input type for note generation"
            
        # Create a prompt for generating notes
        notes_prompt = PromptTemplate.from_template(
            """You are an expert note-taking assistant. Generate comprehensive and well-structured 
            markdown notes from the following content. Format the notes with proper headings, 
            bullet points, and emphasis where appropriate. Focus on the key concepts, explanations, 
            and examples.
            
            CONTENT:
            {text}
            
            INSTRUCTIONS:
            1. Organize the notes in a logical hierarchy with headings and subheadings
            2. Use bullet points for lists and key points
            3. Highlight important terms, definitions, and concepts
            4. Include code blocks with proper syntax highlighting if relevant
            5. Include any mathematical formulas if relevant
            6. Format the output as valid markdown
            
            NOTES:"""
        )
        
        # Use the modern pipe syntax
        try:
            chain = notes_prompt | self.llm | StrOutputParser()
            result = chain.invoke({"text": text})
            return result
        except Exception as e:
            error_msg = f"Error generating notes: {str(e)}"
            print(error_msg)
            return error_msg
            
    def _generate_flashcards_direct(self, text: str) -> str:
        """Direct version of generate_flashcards without @tool decorator"""
        if not self.llm:
            return "Error: Groq LLM not initialized (missing API key)"
            
        # Safety check
        if not isinstance(text, str):
            return "Error: Invalid input type for flashcard generation"
        
        # Create a prompt for generating flashcards
        flashcards_prompt = PromptTemplate.from_template(
            """You are an expert at creating educational flashcards. Generate a set of Anki flashcards 
            from the following content. Each flashcard should have a clear question on the front and 
            a concise answer on the back. Focus on key concepts, definitions, and important facts.
            
            CONTENT:
            {text}
            
            INSTRUCTIONS:
            1. Create 5-15 flashcards depending on the content
            2. Focus on testing recall of important information
            3. Make questions clear and specific
            4. Keep answers concise but complete
            5. Output in the following JSON format exactly matching the schema
            
            {format_instructions}
            
            FLASHCARDS:"""
        )
        
        # Create a chain with format_instructions
        try:
            prompt_with_format = flashcards_prompt.partial(
                format_instructions=self.flashcard_parser.get_format_instructions()
            )
            
            # Use string parser first to get the raw output
            raw_chain = prompt_with_format | self.llm | StrOutputParser()
            result = raw_chain.invoke({"text": text})
            
            # Try to parse as JSON
            try:
                parsed = self.flashcard_parser.parse(result)
                return json.dumps(parsed.dict(), indent=2)
            except Exception as e:
                print(f"Error parsing flashcards: {str(e)}")
                return result  # Return raw result if parsing fails
        except Exception as e:
            error_msg = f"Error generating flashcards: {str(e)}"
            print(error_msg)
            return error_msg
            
    def _generate_mindmap_direct(self, text: str) -> str:
        """Direct version of generate_mindmap without @tool decorator"""
        if not self.llm:
            return "Error: Groq LLM not initialized (missing API key)"
            
        # Safety check
        if not isinstance(text, str):
            return "Error: Invalid input type for mindmap generation"
        
        # Create a prompt for generating mindmaps
        mindmap_prompt = PromptTemplate.from_template(
            """You are an expert at creating concept maps and visual representations of information.
            Generate a structured mindmap in Mermaid format from the following content.
            
            CONTENT:
            {text}
            
            INSTRUCTIONS:
            1. Create a hierarchical mindmap with a central topic and branches
            2. Include 2-3 levels of depth depending on the content complexity
            3. Focus on relationships between concepts
            4. Keep node text concise (2-5 words per node)
            5. Output in valid Mermaid mindmap syntax
            
            The output should start with "```mermaid" and end with "```" and use the Mermaid mindmap syntax like:
            
            ```mermaid
            mindmap
              root((Main Topic))
                Branch 1
                  Sub-topic 1
                  Sub-topic 2
                Branch 2
                  Sub-topic 3
                  Sub-topic 4
            ```
            
            MINDMAP:"""
        )
        
        # Use the modern pipe syntax
        try:
            chain = mindmap_prompt | self.llm | StrOutputParser()
            result = chain.invoke({"text": text})
            
            # Ensure proper mermaid format
            if not result.startswith("```mermaid"):
                result = "```mermaid\n" + result
            if not result.endswith("```"):
                result = result + "\n```"
                
            return result
        except Exception as e:
            error_msg = f"Error generating mindmap: {str(e)}"
            print(error_msg)
            return error_msg

    def _process_mermaid_markdown(self, content: str) -> str:
        """Process markdown content with mermaid diagrams"""
        # First convert the non-mermaid parts to HTML
        # We need to preserve the mermaid code blocks
        lines = content.split("\n")
        in_mermaid_block = False
        mermaid_blocks = []
        current_block = []
        processed_content = []

        for line in lines:
            if line.strip().startswith("```mermaid"):
                in_mermaid_block = True
                current_block = [line]
                processed_content.append('<div class="mermaid">')
            elif in_mermaid_block and line.strip() == "```":
                in_mermaid_block = False
                current_block.append(line)
                mermaid_blocks.append("\n".join(current_block))
                processed_content.append("</div>")
                current_block = []
            elif in_mermaid_block:
                current_block.append(line)
                processed_content.append(line)
            else:
                processed_content.append(line)

        # Convert markdown to HTML using the standard markdown library
        html_content = self.md.convert("\n".join(processed_content))

        # Now we need to ensure the mermaid blocks are properly preserved
        # This approach works because modern browsers will render the mermaid syntax
        # when using the mermaid.js library in the final HTML page
        return html_content
