#!/usr/bin/env python3
"""
TriviaQA Dataset Validation Script
- No special Unicode characters
- Copy-paste ready
- Tests if data is correct
"""

import json
import os

def validate_triviaqa_data():
    """Validate TriviaQA unfiltered dataset structure"""
    
    print("=" * 60)
    print("TriviaQA Dataset Validation")
    print("=" * 60)
    
    # Check if data folder exists
    data_path = '../Data'
    if not os.path.exists(data_path):
        print(f"\nERROR: '{data_path}' folder not found!")
        print("Please create a 'data' folder and place TriviaQA files there.")
        return False
    
    print(f"\nChecking data folder: {data_path}")
    print(f"Files in folder: {os.listdir(data_path)}\n")
    
    # Try to load dev file
    dev_file = os.path.join(data_path, 'unfiltered-web-dev.json')
    
    if not os.path.exists(dev_file):
        print(f"ERROR: Could not find {dev_file}")
        print("Please ensure unfiltered-web-dev.json is in the data folder.")
        return False
    
    print(f"Loading: {dev_file}")
    
    try:
        with open(dev_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
    except Exception as e:
        print(f"ERROR: Could not load JSON file: {e}")
        return False
    
    print("SUCCESS: File loaded\n")
    
    # Check structure
    if 'Data' not in data:
        print("ERROR: 'Data' key not found in JSON")
        return False
    
    samples = data['Data']
    print(f"Total samples in dev set: {len(samples)}\n")
    
    # Check first sample
    print("Checking first sample structure...")
    sample = samples[0]
    
    required_fields = ['Question', 'Answer', 'SearchResults', 'QuestionId']
    answer_fields = ['Value']
    
    missing_fields = [f for f in required_fields if f not in sample]
    if missing_fields:
        print(f"ERROR: Missing fields in sample: {missing_fields}")
        return False
    
    missing_answer_fields = [f for f in answer_fields if f not in sample['Answer']]
    if missing_answer_fields:
        print(f"ERROR: Missing Answer fields: {missing_answer_fields}")
        return False
    
    print("SUCCESS: All required fields present\n")
    
    # Print sample details
    print("Sample 1 Details:")
    print("-" * 60)
    print(f"Question ID: {sample['QuestionId']}")
    print(f"Question: {sample['Question'][:80]}...")
    print(f"Answer: {sample['Answer']['Value']}")
    print(f"Number of search results: {len(sample['SearchResults'])}")
    
    if sample['SearchResults']:
        top_doc = sample['SearchResults'][0]
        print(f"\nTop search result (Rank 0):")
        print(f"  Title: {top_doc.get('Title', 'N/A')[:70]}...")
        print(f"  Description: {top_doc.get('Description', 'N/A')[:70]}...")
    
    print("\n" + "=" * 60)
    print("VALIDATION COMPLETE - DATA IS CORRECT")
    print("=" * 60)
    print("\nYou can now proceed to Phase 1:")
    print("1. Implement create_three_conditions() function")
    print("2. Generate outputs for 50 samples")
    print("3. Label helpful/neutral/harmful")
    
    return True


if __name__ == "__main__":
    success = validate_triviaqa_data()
    exit(0 if success else 1)