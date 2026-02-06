
from __future__ import annotations
import torch
import torch.nn as nn
from typing import TYPE_CHECKING, Any, Literal
from transformers import AutoModel, AutoConfig
from transformers.modeling_outputs import ModelOutput

if TYPE_CHECKING:
    from argparse import Namespace


def detect_model_type(model_config) -> Literal["causal", "encoder", "encoder_decoder"]:
    """Detect the model type based on its configuration."""
    model_name = model_config.model_type.lower()
    
    # Causal models (GPT-style)
    causal_models = ["gpt2", "gpt", "gpt_neo", "gptj", "gpt_neox", "llama", "opt", "bloom", "codegen"]
    if any(causal in model_name for causal in causal_models):
        return "causal"
    
    # Encoder-decoder models
    encoder_decoder_models = ["t5", "bart", "pegasus", "marian", "mbart", "bigbird_pegasus"]
    if any(ed in model_name for ed in encoder_decoder_models):
        return "encoder_decoder"
    
    # Encoder models (BERT-style) - default
    return "encoder"


def get_pooling_strategy(model_type: str, take_final: bool) -> Literal["first", "last", "mean", "max"]:
    """Determine the best pooling strategy based on model type."""
    if model_type == "causal":
        return "last" if take_final else "first"
    elif model_type == "encoder":
        return "first"  # BERT-style uses [CLS] token
    else:  # encoder_decoder
        return "last" if take_final else "first"


class ClassifierHead(nn.Module):

    def __init__(self: ClassifierHead, config: Namespace, hidden_size: int | None = None) -> None:
        """This is the class for the classification head when doing
        sentence/sequence classification. This uses a config object
        to create the classification head for a certain task with a
        given pre-trained model.

        Args:
            config(Namespace): Contains all the information to create
                the classification head, including the number of
                classes for the task.
            hidden_size(int | None): The hidden size of the
                pre-trained model. If it is None, it is assumed that
                the config object contains the hidden size.
        """
        super().__init__()
        hidden_size: int = hidden_size if hidden_size is not None else config.hidden_size
        self.nonlinearity = nn.Sequential(
            nn.LayerNorm(hidden_size, config.classifier_layer_norm_eps, elementwise_affine=False),
            nn.Linear(hidden_size, hidden_size),
            nn.GELU(),
            nn.LayerNorm(hidden_size, config.classifier_layer_norm_eps, elementwise_affine=False),
            nn.Dropout(config.classifier_dropout),
            nn.Linear(hidden_size, config.num_labels)
        )

    def forward(self: ClassifierHead, encodings: torch.Tensor) -> torch.Tensor:
        """This function handles the forward call of the
        classification head. It takes the model encodings and
        gives the logits for each class.

        Args:
            encodings(torch.Tensor): A tensor containing a the
            model encodings of the data used to classify.

        Returns:
            torch.Tensor: The logits for each class based on
                the encodings of the model for a given input.

        Shapes:
            - encodings: :math:`(B, S, D)`
        """
        return self.nonlinearity(encodings)


class ModelForSequenceClassification(nn.Module):

    def __init__(self: ModelForSequenceClassification, config: Namespace) -> None:
        """This is class create extends a pre-trained language model to
        classification tasks. This requires fine-tuning since the head
        is randomly generated. The model handles multiple output types
        of the pre-trained language model and automatically determines
        the best pooling strategy based on model architecture.

        Args:
            config(Namespace): Contains all the information to create
                the classification model, including the path to the
                pre-trained model and whether to pass the first or
                last token to the classification head.
        """
        super().__init__()
        
        # Load model configuration first to detect model type
        model_config = AutoConfig.from_pretrained(config.model_name_or_path, trust_remote_code=True, revision=config.revision_name)
        
        # Detect model type and determine pooling strategy
        self.model_type = detect_model_type(model_config)
        self.pooling_strategy = getattr(config, 'pooling_strategy', None)
        if self.pooling_strategy is None:
            self.pooling_strategy = get_pooling_strategy(self.model_type, config.take_final)
        
        # Load the transformer model
        self.transformer: nn.Module = AutoModel.from_pretrained(
            config.model_name_or_path, 
            trust_remote_code=True, 
            revision=config.revision_name, 
            use_cache=False
        )
        
        # Handle encoder-decoder models
        self.enc_dec: bool = config.enc_dec or (self.model_type == "encoder_decoder")
        if self.enc_dec and hasattr(model_config, 'decoder_start_token_id'):
            self.decoder_start_token_id = model_config.decoder_start_token_id
        elif self.enc_dec:
            # Fallback for models without decoder_start_token_id
            self.decoder_start_token_id = getattr(model_config, 'eos_token_id', 0)
        
        # Create classifier head
        hidden_size = model_config.hidden_size
        self.classifier: nn.Module = ClassifierHead(config, hidden_size)
        self.take_final: bool = config.take_final

    def forward(self, input_data: torch.Tensor, attention_mask: torch.Tensor | None = None) -> torch.Tensor:
        # Handle encoder-decoder models
        if self.enc_dec:
            batch_size = attention_mask.size(0)
            decoder_input_ids = input_data.new_full((batch_size, 1), self.decoder_start_token_id)
            decoder_attention_mask = attention_mask.new_ones((batch_size, 1))
            output_transformer = self.transformer(
                input_ids=input_data,
                attention_mask=attention_mask,
                decoder_input_ids=decoder_input_ids,
                decoder_attention_mask=decoder_attention_mask,
            )
        else:
            output_transformer = self.transformer(
                input_ids=input_data,
                attention_mask=attention_mask,
                use_cache=False
            )

        # Extract hidden states from model output
        encoding = self._extract_hidden_states(output_transformer)
        
        # Apply pooling strategy
        transformer_output = self._apply_pooling(encoding, attention_mask)
        
        # Get logits from classifier
        logits: torch.Tensor = self.classifier(transformer_output)
        return logits

    def _extract_hidden_states(self, output_transformer) -> torch.Tensor:
        """Extract hidden states from various model output formats."""
        if isinstance(output_transformer, tuple):
            return output_transformer[0]
        elif isinstance(output_transformer, ModelOutput):
            if hasattr(output_transformer, "last_hidden_state"):
                return output_transformer.last_hidden_state
            elif hasattr(output_transformer, "hidden_states") and output_transformer.hidden_states is not None:
                return output_transformer.hidden_states[-1]
            elif hasattr(output_transformer, "logits"):
                return output_transformer.logits
            else:
                raise ValueError(f"Unknown ModelOutput format: {type(output_transformer)}")
        else:
            raise ValueError(f"Unsupported output type: {type(output_transformer)}")

    def _apply_pooling(self, encoding: torch.Tensor, attention_mask: torch.Tensor | None = None) -> torch.Tensor:
        """Apply the appropriate pooling strategy to get sequence representation."""
        if self.pooling_strategy == "first":
            return encoding[:, 0]  # First token (e.g., [CLS] for BERT)
        
        elif self.pooling_strategy == "last":
            if attention_mask is None:
                return encoding[:, -1]  # Last token if no mask
            else:
                # Last non-padded token
                seq_lengths = attention_mask.sum(dim=1) - 1
                seq_lengths = torch.clamp(seq_lengths, min=0)  # Ensure non-negative
                batch_indices = torch.arange(encoding.size(0), device=encoding.device)
                return encoding[batch_indices, seq_lengths]
        
        elif self.pooling_strategy == "mean":
            if attention_mask is None:
                return encoding.mean(dim=1)  # Mean over all tokens
            else:
                # Mean pooling over non-padded tokens
                mask_expanded = attention_mask.unsqueeze(-1).expand_as(encoding)
                sum_embeddings = (encoding * mask_expanded).sum(dim=1)
                sum_mask = mask_expanded.sum(dim=1)
                return sum_embeddings / torch.clamp(sum_mask, min=1e-9)  # Avoid division by zero
        
        elif self.pooling_strategy == "max":
            if attention_mask is None:
                return encoding.max(dim=1)[0]  # Max over all tokens
            else:
                # Max pooling over non-padded tokens
                mask_expanded = attention_mask.unsqueeze(-1).expand_as(encoding)
                # Set padded positions to very negative values so they don't affect max
                masked_encoding = encoding.masked_fill(~mask_expanded.bool(), float('-inf'))
                return masked_encoding.max(dim=1)[0]
        
        else:
            raise ValueError(f"Unknown pooling strategy: {self.pooling_strategy}")