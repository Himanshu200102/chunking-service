#!/usr/bin/env python3
"""Test script to diagnose summarizer issues."""
import sys
import os
sys.path.insert(0, '/app')

from app.retriever.summarizer import get_summarizer
from app.retriever.opensearch_client import OpenSearchClient
from app.retriever.config import settings

def test_summarizer():
    print("Testing summarizer...")
    
    # Get summarizer
    summarizer = get_summarizer()
    print(f"Summarizer model loaded: {summarizer.model is not None}")
    
    # Get chunks from OpenSearch
    client = OpenSearchClient()
    projectid = "p_43b3e12cfe"
    query = "Summarize the doc"
    
    print(f"Searching for chunks in project {projectid}...")
    chunks = client.search_chunks(
        query=query,
        projectid=projectid,
        fileid=None,
        max_results=10
    )
    
    print(f"Found {len(chunks)} chunks")
    
    if not chunks:
        print("ERROR: No chunks found!")
        return
    
    # Test summarization
    print(f"Calling summarize_chunks with {len(chunks)} chunks...")
    result = summarizer.summarize_chunks(
        chunks=chunks,
        query=query,
        projectid=projectid,
        max_chunks_to_summarize=10
    )
    
    if result:
        summary = result.get("summary")
        print(f"Summary result: {result is not None}")
        print(f"Summary field exists: {'summary' in result}")
        if summary:
            print(f"Summary length: {len(summary)} chars")
            print(f"Summary preview: {summary[:200]}...")
        else:
            print("ERROR: Summary field is None or empty!")
    else:
        print("ERROR: summarize_chunks returned None!")

if __name__ == "__main__":
    test_summarizer()


