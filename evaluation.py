"""
Evaluation Script for Summarization Service
------------------------------------------
Measures faithfulness and relevance of generated summaries using:
- Faithfulness: DeBERTa NLI model to assess factual correctness
- Relevance: Semantic similarity using embeddings from nomic-embed-text or sentence-transformers

Usage:
    python evaluation.py [--processed_dir PROCESSED_DIR] [--results_file RESULTS_FILE] 
"""

import os
import json
import argparse
import pandas as pd
import numpy as np
from pathlib import Path
from tqdm import tqdm
import re

# For faithfulness scoring with DeBERTa NLI
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

# For relevance scoring with embeddings
embedding_model = None
embedding_tokenizer = None

# Try importing Ollama for embeddings
try:
    import ollama
    ollama_available = True
    print("Ollama is available for embeddings.")
except ImportError:
    print("Ollama is not installed. Falling back to sentence-transformers.")
    ollama_available = False

# Try importing sentence-transformers as fallback
try:
    from sentence_transformers import SentenceTransformer
    st_available = True
    print("Loading sentence-transformers model for embeddings...")
    embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
    print("Sentence-transformers model loaded successfully.")
except ImportError:
    st_available = False
    print("sentence-transformers not available. Install with: pip install sentence-transformers")

# Default paths
DEFAULT_PROCESSED_DIR = "scripts/outputs/processed"
DEFAULT_RESULTS_FILE = "results/summarization_metrics_results.csv"

# Load DeBERTa NLI model and tokenizer
print("Loading DeBERTa NLI model...")
model_name = "MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name)
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
model = model.to(device)
print(f"Model loaded on device: {device}")

def compute_nli_faithfulness(premise, hypothesis):
    """Compute faithfulness score using NLI model.
    
    Args:
        premise: Source text (e.g., transcription)
        hypothesis: Generated text to evaluate (e.g., notes)
        
    Returns:
        Float between 0-1 representing entailment probability
    """
    # Skip empty inputs
    if not premise or not hypothesis or len(premise.strip()) < 10 or len(hypothesis.strip()) < 10:
        print("Warning: Empty or very short input for NLI evaluation")
        return None
    
    try:
        # Clean and truncate inputs if needed
        premise = premise[:5000]  # Truncate premise to avoid token limits
        hypothesis = hypothesis[:1000]  # Truncate hypothesis
        
        inputs = tokenizer(premise, hypothesis, truncation=True, return_tensors="pt").to(device)
        with torch.no_grad():
            logits = model(**inputs).logits
        # Label mapping: 0=entailment, 1=neutral, 2=contradiction
        probs = torch.softmax(logits, dim=1)[0]
        entailment_prob = float(probs[0])  # Faithfulness = entailment probability
        return entailment_prob
    except Exception as e:
        print(f"Error in NLI computation: {str(e)}")
        return None

def compute_relevance(source, output):
    """Compute semantic similarity between source and output using embeddings.
    
    Args:
        source: Source text or query
        output: Generated text to evaluate
        
    Returns:
        Float between 0-1 representing cosine similarity
    """
    # Skip empty inputs
    if not source or not output or len(source.strip()) < 5 or len(output.strip()) < 5:
        print("Warning: Empty or very short input for relevance evaluation")
        return None
    
    # Clean and truncate inputs
    source = source[:1000]  # Truncate source
    output = output[:1000]  # Truncate output
    
    # Try using Ollama first
    if ollama_available:
        try:
            print("Trying Ollama for embeddings...")
            emb_source = np.array(ollama.embeddings(model='nomic-embed-text', prompt=str(source))['embedding'])
            emb_output = np.array(ollama.embeddings(model='nomic-embed-text', prompt=str(output))['embedding'])
            score = float(np.dot(emb_source, emb_output) / (np.linalg.norm(emb_source) * np.linalg.norm(emb_output)))
            print(f"Successfully computed relevance with Ollama: {score:.4f}")
            return score
        except Exception as e:
            print(f"Ollama relevance error: {e}")
            print("Falling back to sentence-transformers...")
    
    # Fall back to sentence-transformers if available
    if st_available and embedding_model:
        try:
            emb_source = embedding_model.encode(source)
            emb_output = embedding_model.encode(output)
            score = float(np.dot(emb_source, emb_output) / (np.linalg.norm(emb_source) * np.linalg.norm(emb_output)))
            print(f"Successfully computed relevance with sentence-transformers: {score:.4f}")
            return score
        except Exception as e:
            print(f"Sentence-transformers error: {e}")
    
    print("Warning: No embedding method available. Install either 'ollama' or 'sentence-transformers'.")
    return None

def extract_text_from_mermaid(mermaid_content):
    """Extract readable text from a mermaid diagram.
    
    Args:
        mermaid_content: Mermaid diagram content
        
    Returns:
        Extracted text content
    """
    if not mermaid_content:
        return ""
    
    # Remove code blocks and mindmap declaration
    content = re.sub(r'```mermaid\s*', '', mermaid_content)
    content = re.sub(r'```\s*', '', content)
    content = re.sub(r'^mindmap\s*', '', content, flags=re.MULTILINE)
    
    # Extract node text
    # This regex matches text inside parentheses, brackets or plain nodes
    node_texts = re.findall(r'[^[(]*\(\((.*?)\)\)|[^[(]*\[(.*?)\]|[^[(]*\((.*?)\)|(?<=\s)([\w\s]+)(?=:::|\n|$)', content)
    
    # Flatten the results and filter empty strings
    extracted_texts = []
    for node_match in node_texts:
        for text in node_match:
            if text and text.strip():
                extracted_texts.append(text.strip())
    
    return " ".join(extracted_texts)

def clean_text(text):
    """Clean text for evaluation, removing markdown formatting and code blocks."""
    if not text:
        return ""
    
    # Check if this is a mermaid diagram
    if "```mermaid" in text or "mindmap" in text:
        return extract_text_from_mermaid(text)
    
    # Remove markdown code blocks
    text = re.sub(r'```[\s\S]*?```', '', text)
    
    # Remove markdown formatting
    text = re.sub(r'#\s+', '', text)  # Remove headings
    text = re.sub(r'\*\*|\*|__', '', text)  # Remove bold/italic
    text = re.sub(r'\[([^\]]+)\]\([^)]+\)', r'\1', text)  # Replace links with link text
    
    # Remove remaining non-alphanumeric except basic punctuation
    text = re.sub(r'[^\w\s.,:;!?-]', '', text)
    
    return text.strip()

def load_processed_videos(processed_dir):
    """Load all processed videos from the specified directory.
    
    Args:
        processed_dir: Directory containing processed JSON files
        
    Returns:
        List of dictionaries with video data
    """
    processed_dir = Path(processed_dir)
    results = []
    
    for file_path in processed_dir.glob("*.json"):
        try:
            with open(file_path, 'r') as f:
                data = json.load(f)
                if data.get("status") == "success" and "outputs" in data:
                    results.append(data)
                    print(f"Loaded: {file_path.name}")
        except Exception as e:
            print(f"Error loading {file_path}: {e}")
    
    print(f"Loaded {len(results)} processed videos")
    return results

def get_prompt_types_from_outputs(outputs):
    """Extract prompt types from outputs based on summarization service."""
    prompt_types = {
        "notes": "structured markdown notes",
        "flashcards": "educational flashcards",
        "mindmap": "hierarchical mindmap"
    }
    return prompt_types

def evaluate_summarization(video_data):
    """Evaluate the summarization quality for a single video.
    
    Args:
        video_data: Dictionary with video data
        
    Returns:
        Dictionary with evaluation metrics
    """
    video_id = video_data["video_info"]["video_id"]
    title = video_data["video_info"]["title"]
    print(f"\nEvaluating video: {video_id} - {title}")
    
    # Get the original transcription and generated summaries
    transcript = video_data["transcription"]["text"]
    notes = video_data["outputs"].get("notes", "")
    
    # Extract flashcards content
    try:
        flashcards_str = video_data["outputs"].get("flashcards", "{}")
        if isinstance(flashcards_str, str):
            flashcards_json = json.loads(flashcards_str)
        else:
            flashcards_json = flashcards_str
            
        if "flashcards" in flashcards_json:
            flashcards_text = "\n".join([f"Q: {card['front']}\nA: {card['back']}" 
                                      for card in flashcards_json.get("flashcards", [])])
        else:
            flashcards_text = ""
            print(f"Warning: No flashcards found in output for {video_id}")
    except (json.JSONDecodeError, KeyError, TypeError) as e:
        print(f"Error parsing flashcards for {video_id}: {e}")
        flashcards_json = {"flashcards": []}
        flashcards_text = ""
    
    # Get mindmap content
    mindmap = video_data["outputs"].get("mindmap", "")
    if mindmap:
        print(f"Found mindmap for {video_id} ({len(mindmap)} chars)")
    else:
        print(f"No mindmap found for {video_id}")
    
    # Clean text for evaluation
    clean_transcript = clean_text(transcript)
    clean_notes = clean_text(notes)
    clean_flashcards = clean_text(flashcards_text)
    clean_mindmap = clean_text(mindmap)
    
    print(f"Cleaned text lengths - Notes: {len(clean_notes)}, Flashcards: {len(clean_flashcards)}, Mindmap: {len(clean_mindmap)}")
    
    # Compute faithfulness scores
    print("Computing faithfulness scores...")
    notes_faithfulness = compute_nli_faithfulness(clean_transcript, clean_notes)
    flashcards_faithfulness = compute_nli_faithfulness(clean_transcript, clean_flashcards)
    mindmap_faithfulness = compute_nli_faithfulness(clean_transcript, clean_mindmap)
    
    # Compute relevance scores (using title as query context)
    print("Computing relevance scores...")
    notes_relevance = compute_relevance(title, clean_notes)
    flashcards_relevance = compute_relevance(title, clean_flashcards)
    mindmap_relevance = compute_relevance(title, clean_mindmap)
    
    # Extract prompt types
    prompt_types = get_prompt_types_from_outputs(video_data["outputs"])
    
    result = {
        "video_id": video_id,
        "title": title,
        "notes_faithfulness": notes_faithfulness,
        "flashcards_faithfulness": flashcards_faithfulness,
        "mindmap_faithfulness": mindmap_faithfulness,
        "notes_relevance": notes_relevance,
        "flashcards_relevance": flashcards_relevance,
        "mindmap_relevance": mindmap_relevance,
        "transcript_length": len(clean_transcript.split()),
        "notes_length": len(clean_notes.split()),
        "flashcards_count": len(flashcards_json.get("flashcards", [])) if flashcards_text else 0,
        "prompt_types": prompt_types
    }
    
    print(f"Results for {video_id}:")
    print(f"  Notes faithfulness: {notes_faithfulness or 'N/A'}")
    print(f"  Flashcards faithfulness: {flashcards_faithfulness or 'N/A'}")
    print(f"  Mindmap faithfulness: {mindmap_faithfulness or 'N/A'}")
    print(f"  Notes relevance: {notes_relevance or 'N/A'}")
    print(f"  Flashcards relevance: {flashcards_relevance or 'N/A'}")
    print(f"  Mindmap relevance: {mindmap_relevance or 'N/A'}")
    
    return result

def generate_aggregate_stats(results_df):
    """Generate aggregate statistics from evaluation results.
    
    Args:
        results_df: DataFrame with evaluation results
        
    Returns:
        Dictionary with aggregate statistics
    """
    metrics = [
        "notes_faithfulness", "flashcards_faithfulness", "mindmap_faithfulness",
        "notes_relevance", "flashcards_relevance", "mindmap_relevance"
    ]
    
    stats = {}
    
    # Calculate means
    for metric in metrics:
        if metric in results_df.columns:
            # Convert to numeric and handle NaN values
            results_df[metric] = pd.to_numeric(results_df[metric], errors='coerce')
            if not results_df[metric].empty and not results_df[metric].isna().all():
                stats[f"avg_{metric}"] = results_df[metric].mean()
                stats[f"min_{metric}"] = results_df[metric].min()
                stats[f"max_{metric}"] = results_df[metric].max()
                stats[f"median_{metric}"] = results_df[metric].median()
            else:
                stats[f"avg_{metric}"] = None
                stats[f"min_{metric}"] = None
                stats[f"max_{metric}"] = None
                stats[f"median_{metric}"] = None
    
    # Calculate correlations
    faithfulness_metrics = [m for m in metrics if "faithfulness" in m and m in results_df.columns]
    relevance_metrics = [m for m in metrics if "relevance" in m and m in results_df.columns]
    
    # Faithfulness vs relevance correlation
    for f_metric in faithfulness_metrics:
        for r_metric in relevance_metrics:
            corr_name = f"corr_{f_metric}_vs_{r_metric}"
            mask = ~results_df[f_metric].isna() & ~results_df[r_metric].isna()
            if mask.sum() > 1:  # Need at least 2 pairs for correlation
                stats[corr_name] = results_df.loc[mask, f_metric].corr(results_df.loc[mask, r_metric])
            else:
                stats[corr_name] = None
    
    return stats

def main():
    parser = argparse.ArgumentParser(description="Evaluate summarization quality")
    parser.add_argument("--processed_dir", default=DEFAULT_PROCESSED_DIR, 
                        help=f"Directory with processed videos (default: {DEFAULT_PROCESSED_DIR})")
    parser.add_argument("--results_file", default=DEFAULT_RESULTS_FILE,
                        help=f"Path to save results (default: {DEFAULT_RESULTS_FILE})")
    parser.add_argument("--verbose", action="store_true",
                        help="Print detailed debugging information")
    parser.add_argument("--single_file", default=None,
                        help="Evaluate a single JSON file instead of a directory")
    args = parser.parse_args()
    
    # Check for embedding availability
    if not ollama_available and not st_available:
        print("WARNING: No embedding models available. Relevance scores will not be computed.")
        print("Install either 'ollama' or 'sentence-transformers' for relevance evaluation.")
    
    # Load processed videos
    if args.single_file:
        try:
            with open(args.single_file, 'r') as f:
                data = json.load(f)
                if data.get("status") == "success" and "outputs" in data:
                    videos = [data]
                    print(f"Loaded single file: {args.single_file}")
                else:
                    print(f"Error: File {args.single_file} does not contain valid data")
                    return
        except Exception as e:
            print(f"Error loading {args.single_file}: {e}")
            return
    else:
        videos = load_processed_videos(args.processed_dir)
    
    if not videos:
        print("No processed videos found. Exiting.")
        return
    
    # Evaluate each video
    results = []
    for video in tqdm(videos, desc="Evaluating videos"):
        try:
            eval_result = evaluate_summarization(video)
            results.append(eval_result)
        except Exception as e:
            print(f"Error evaluating {video['video_info']['video_id']}: {e}")
            import traceback
            if args.verbose:
                traceback.print_exc()
    
    # Create DataFrame and save results
    results_df = pd.DataFrame(results)
    
    # Calculate and display aggregate statistics
    stats = generate_aggregate_stats(results_df)
    
    print("\nSummarization Evaluation Results:")
    print("=" * 40)
    
    # Print faithfulness metrics
    print("\nFaithfulness Scores (higher is better):")
    for output_type in ["notes", "flashcards", "mindmap"]:
        metric = f"{output_type}_faithfulness"
        avg_key = f"avg_{metric}"
        min_key = f"min_{metric}"
        max_key = f"max_{metric}"
        if avg_key in stats and stats[avg_key] is not None:
            print(f"  {output_type.capitalize()}: {stats[avg_key]:.4f} (min: {stats[min_key]:.4f}, max: {stats[max_key]:.4f})")
        else:
            print(f"  {output_type.capitalize()}: No valid data")
    
    # Print relevance metrics
    print("\nRelevance Scores (higher is better):")
    for output_type in ["notes", "flashcards", "mindmap"]:
        metric = f"{output_type}_relevance"
        avg_key = f"avg_{metric}"
        min_key = f"min_{metric}"
        max_key = f"max_{metric}"
        if avg_key in stats and stats[avg_key] is not None:
            print(f"  {output_type.capitalize()}: {stats[avg_key]:.4f} (min: {stats[min_key]:.4f}, max: {stats[max_key]:.4f})")
        else:
            print(f"  {output_type.capitalize()}: No valid data")
    
    # Save detailed results
    result_path = Path(args.results_file)
    result_path.parent.mkdir(exist_ok=True, parents=True)
    results_df.to_csv(result_path, index=False)
    print(f"\nSaved detailed metrics to {result_path}")
    
    # Save aggregate stats as JSON
    stats_path = result_path.parent / "summarization_stats.json"
    with open(stats_path, 'w') as f:
        json.dump({k: float(v) if v is not None else None for k, v in stats.items()}, f, indent=2)
    print(f"Saved aggregate statistics to {stats_path}")

if __name__ == "__main__":
    main() 