#!/usr/bin/env python3
"""
Example usage of the model-agnostic classifier for BabyLM evaluation.

This script demonstrates how the classifier automatically adapts to different model types:
- GPT-2 (causal): Uses last token pooling
- BERT (encoder): Uses first token pooling  
- T5 (encoder-decoder): Uses last token pooling
"""

import argparse
from classifier_model import ModelForSequenceClassification

def create_example_config(model_path: str, task: str = "mnli", num_labels: int = 3):
    """Create a minimal config for testing."""
    config = argparse.Namespace()
    config.model_name_or_path = model_path
    config.task = task
    config.num_labels = num_labels
    config.take_final = True
    config.enc_dec = False
    config.pooling_strategy = None  # Will be auto-detected
    config.classifier_dropout = 0.1
    config.classifier_layer_norm_eps = 1e-5
    config.revision_name = "main"
    return config

def main():
    # Example usage with different model types
    models_to_test = [
        "gpt2",  # Causal model
        "bert-base-uncased",  # Encoder model
        "t5-small",  # Encoder-decoder model
    ]
    
    for model_name in models_to_test:
        print(f"\n=== Testing with {model_name} ===")
        
        try:
            # Create config
            config = create_example_config(model_name)
            
            # Initialize model
            model = ModelForSequenceClassification(config)
            
            print(f"Model type detected: {model.model_type}")
            print(f"Pooling strategy: {model.pooling_strategy}")
            print(f"Encoder-decoder: {model.enc_dec}")
            
        except Exception as e:
            print(f"Error with {model_name}: {e}")

if __name__ == "__main__":
    main()







