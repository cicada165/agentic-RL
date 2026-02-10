"""
Model factory and utilities for Search-R1.
"""
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from typing import Tuple

class ModelFactory:
    """Factory for creating models and tokenizers."""
    
    @staticmethod
    def load_model_and_tokenizer(model_name_or_path: str, device: str) -> Tuple[AutoModelForCausalLM, AutoTokenizer]:
        """
        Load the model and tokenizer.
        
        Args:
            model_name_or_path: Path or name of the model.
            device: Device to load model on ("cuda" or "cpu").
            
        Returns:
            Tuple of (model, tokenizer).
        """
        print(f"Loading model from {model_name_or_path}...")
        
        # Determine device map and dtype
        device_type = torch.device(device).type
        if device_type == "cuda":
            dtype = torch.float16
            device_map = "auto"
        else:
            dtype = torch.float32
            device_map = None
            
        tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
        model = AutoModelForCausalLM.from_pretrained(
            model_name_or_path,
            torch_dtype=dtype,
            device_map=device_map
        )
        
        if device_type == "cpu":
            model = model.to(device)
            
        return model, tokenizer
