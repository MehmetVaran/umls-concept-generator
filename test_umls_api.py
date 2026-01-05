#!/usr/bin/env python3
"""Test script for UMLS API functionality.

This script tests the basic UMLS API calls without LangChain dependencies
to verify that the API key and basic functionality work correctly.
"""

import json
import logging
from pathlib import Path
from concept_generator_main import UMLSClient

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
LOGGER = logging.getLogger(__name__)


def test_umls_basic():
    """Test basic UMLS API functionality."""
    print("=" * 60)
    print("Testing UMLS API Basic Functionality")
    print("=" * 60)
    
    try:
        # Initialize client
        print("\n1. Initializing UMLS client...")
        client = UMLSClient()
        print("   ✓ Client initialized successfully")
        
        # Test search
        print("\n2. Testing concept search for 'Pneumonia'...")
        results = client.search_concepts("Pneumonia", page_size=5)
        print(f"   ✓ Search returned {len(results)} results")
        
        if results:
            print("\n   First result:")
            first_result = results[0]
            print(f"   - Name: {first_result.get('name', 'N/A')}")
            print(f"   - CUI: {first_result.get('ui', 'N/A')}")
            print(f"   - Score: {first_result.get('score', 'N/A')}")
            
            # Test fetching concept details
            cui = first_result.get('ui')
            if cui:
                print(f"\n3. Testing concept details for CUI: {cui}...")
                details = client.fetch_concept_details(cui)
                print("   ✓ Concept details retrieved")
                
                # Print some key information
                if isinstance(details, dict):
                    print(f"   - Preferred name: {details.get('name', 'N/A')}")
                    semantic_types = details.get('semanticTypes', [])
                    if semantic_types:
                        print(f"   - Semantic types: {[st.get('name', 'N/A') for st in semantic_types[:3]]}")
                
                # Test fetching related concepts
                print(f"\n4. Testing related concepts for CUI: {cui}...")
                related = client.fetch_related_concepts(cui)
                print(f"   ✓ Retrieved {len(related)} related concepts")
                
                if related:
                    print("\n   First few related concepts:")
                    for rel in related[:3]:
                        print(f"   - {rel.get('relatedIdName', 'N/A')} (Relation: {rel.get('relationLabel', 'N/A')})")
        
        print("\n" + "=" * 60)
        print("All basic tests passed!")
        print("=" * 60)
        return True
        
    except ValueError as e:
        print(f"\n✗ Error: {e}")
        print("\nMake sure you have a UMLS API key in:")
        print(f"  {Path(__file__).parent / '.umls_api_key'}")
        return False
    except Exception as e:
        print(f"\n✗ Unexpected error: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    test_umls_basic()

