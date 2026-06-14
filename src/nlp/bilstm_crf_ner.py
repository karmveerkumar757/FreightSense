# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModel
from torchcrf import CRF
from typing import List, Dict, Tuple
from torch.utils.data import Dataset
import numpy as np

# Label set for FreightSense logistics NER using BIO tagging scheme
LABEL_LIST = [
    "O",
    "B-CARGO_TYPE", "I-CARGO_TYPE",
    "B-TIME_CONSTRAINT", "I-TIME_CONSTRAINT",
    "B-ROUTE_CONSTRAINT", "I-ROUTE_CONSTRAINT",
    "B-SPECIAL_HANDLING", "I-SPECIAL_HANDLING",
    "B-COMPLIANCE_REQ", "I-COMPLIANCE_REQ",
    "B-LOCATION", "I-LOCATION",
    "B-VEHICLE_TYPE", "I-VEHICLE_TYPE"
]
LABEL2ID = {l: i for i, l in enumerate(LABEL_LIST)}
ID2LABEL = {i: l for l, i in LABEL2ID.items()}
NUM_LABELS = len(LABEL_LIST)
# For mixed English/Hindi processing
MODEL_NAME = "bert-base-multilingual-cased"

class FreightNERDataset(Dataset):
    """
    PyTorch Dataset for handling word-level BIO tags and converting them to 
    subword-level inputs for IndicBERT.
    """
    def __init__(self, data: List[Dict], tokenizer, max_len: int = 128):
        self.data = data
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        words = item["tokens"]
        labels = item["labels"]

        input_ids = []
        attention_mask = []
        label_ids = []

        # Add CLS token
        input_ids.append(self.tokenizer.cls_token_id)
        attention_mask.append(1)
        label_ids.append(-100) # -100 is ignored by PyTorch loss functions

        for word, label in zip(words, labels):
            word_tokens = self.tokenizer.tokenize(word)
            if not word_tokens:
                continue
                
            word_ids = self.tokenizer.convert_tokens_to_ids(word_tokens)
            
            input_ids.extend(word_ids)
            attention_mask.extend([1] * len(word_ids))
            
            # Label alignment: assign the actual label to the first subword token
            # and -100 to subsequent subword tokens of the same word.
            label_id = LABEL2ID[label]
            label_ids.extend([label_id] + [-100] * (len(word_ids) - 1))

        # Add SEP token
        input_ids.append(self.tokenizer.sep_token_id)
        attention_mask.append(1)
        label_ids.append(-100)

        # Padding
        padding_length = self.max_len - len(input_ids)
        if padding_length > 0:
            input_ids = input_ids + ([self.tokenizer.pad_token_id] * padding_length)
            attention_mask = attention_mask + ([0] * padding_length)
            label_ids = label_ids + ([-100] * padding_length)
        else:
            input_ids = input_ids[:self.max_len]
            attention_mask = attention_mask[:self.max_len]
            label_ids = label_ids[:self.max_len]

        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
            "labels": torch.tensor(label_ids, dtype=torch.long)
        }

class BiLSTMCRF(nn.Module):
    """
    State-of-the-art NER architecture combining contextual embeddings (IndicBERT), 
    sequence modeling (Bi-LSTM), and structured prediction (CRF).
    """
    def __init__(self, freeze_bert: bool = False):
        super(BiLSTMCRF, self).__init__()
        self.bert = AutoModel.from_pretrained(MODEL_NAME)
        
        if freeze_bert:
            for param in self.bert.parameters():
                param.requires_grad = False

        hidden_size = 256
        self.bilstm = nn.LSTM(
            input_size=self.bert.config.hidden_size,
            hidden_size=hidden_size,
            num_layers=2,
            batch_first=True,
            bidirectional=True,
            dropout=0.3
        )
        
        self.dropout = nn.Dropout(0.3)
        self.hidden2tag = nn.Linear(hidden_size * 2, NUM_LABELS)
        
        # CRF expects batch_first=True
        self.crf = CRF(num_tags=NUM_LABELS, batch_first=True)

    def forward(self, input_ids, attention_mask, labels=None):
        """
        Calculates the negative log likelihood loss for training.
        """
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        sequence_output = outputs.last_hidden_state
        
        lstm_out, _ = self.bilstm(sequence_output)
        lstm_out = self.dropout(lstm_out)
        emissions = self.hidden2tag(lstm_out)
        
        if labels is not None:
            # Create a mask for CRF: Ignore -100 labels (subwords/padding)
            # We must provide valid labels for -100 before passing to CRF, 
            # and use the mask to tell CRF to ignore them.
            # CRF requires the first timestep mask to be True.
            # So we use attention_mask, and replace -100 labels with 0 ('O' tag)
            mask = attention_mask.bool()
            
            crf_labels = labels.clone()
            crf_labels[crf_labels == -100] = 0
            
            # Loss is negative log likelihood
            loss = -self.crf(emissions, crf_labels, mask=mask, reduction='mean')
            return loss
        else:
            # Inference: return decoded path using Viterbi decoding
            mask = attention_mask.bool()
            prediction = self.crf.decode(emissions, mask=mask)
            return prediction

    @torch.no_grad()
    def predict(self, text: str, tokenizer) -> List[Tuple[str, str]]:
        """
        Takes a raw string, predicts NER tags, and re-aggregates subwords back to words.
        """
        self.eval()
        device = next(self.parameters()).device
        
        # Simple word tokenization for inference
        words = text.split()
        
        input_ids = [tokenizer.cls_token_id]
        attention_mask = [1]
        word_idx_map = [] # maps flat token indices to original word index
        
        for i, word in enumerate(words):
            tokens = tokenizer.tokenize(word)
            ids = tokenizer.convert_tokens_to_ids(tokens)
            input_ids.extend(ids)
            attention_mask.extend([1] * len(ids))
            
            # Store the index of the original word for the first subword only
            word_idx_map.append(len(input_ids) - len(ids)) 
            
        input_ids.append(tokenizer.sep_token_id)
        attention_mask.append(1)
        
        # Convert to tensors
        tensor_input_ids = torch.tensor([input_ids], dtype=torch.long).to(device)
        tensor_mask = torch.tensor([attention_mask], dtype=torch.long).to(device)
        
        # Get Viterbi sequence
        predictions = self.forward(tensor_input_ids, tensor_mask)[0]
        
        results = []
        for word, subword_start_idx in zip(words, word_idx_map):
            label_id = predictions[subword_start_idx]
            label = ID2LABEL[label_id]
            results.append((word, label))
            
        return results

# Example data structure
SAMPLE_TRAINING_DATA = [
    {
        "tokens": ["Deliver", "frozen", "medicines", "before", "6", "PM", "avoid", "Ring", "Road"],
        "labels": ["O", "B-SPECIAL_HANDLING", "B-CARGO_TYPE", "B-TIME_CONSTRAINT", "I-TIME_CONSTRAINT", "I-TIME_CONSTRAINT", "O", "B-ROUTE_CONSTRAINT", "I-ROUTE_CONSTRAINT"]
    },
    {
        "tokens": ["Urgent", "pharma", "shipment", "to", "AIIMS", "before", "8", "AM", "refrigerated", "van"],
        "labels": ["O", "B-CARGO_TYPE", "I-CARGO_TYPE", "O", "B-LOCATION", "B-TIME_CONSTRAINT", "I-TIME_CONSTRAINT", "I-TIME_CONSTRAINT", "B-VEHICLE_TYPE", "I-VEHICLE_TYPE"]
    },
    {
        "tokens": ["e-Way", "Bill", "expires", "noon", "tomorrow", "no", "tunnels", "flammable", "goods"],
        "labels": ["B-COMPLIANCE_REQ", "I-COMPLIANCE_REQ", "O", "B-TIME_CONSTRAINT", "I-TIME_CONSTRAINT", "O", "B-ROUTE_CONSTRAINT", "B-SPECIAL_HANDLING", "I-SPECIAL_HANDLING"]
    }
]
