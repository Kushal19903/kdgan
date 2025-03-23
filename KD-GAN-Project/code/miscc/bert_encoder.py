import torch
import torch.nn as nn
from transformers import BertModel, BertTokenizer

class BertEncoder(nn.Module):
    def __init__(self, cfg, device):
        super(BertEncoder, self).__init__()
        self.device = device
        self.cfg = cfg
        
        # Initialize BERT model and tokenizer
        self.bert_model = BertModel.from_pretrained('bert-base-uncased')
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        
        # Projection layer to match the expected embedding dimension
        self.projection = nn.Linear(768, cfg.TEXT.EMBEDDING_DIM)
        
        # Freeze BERT parameters for stability
        for param in self.bert_model.parameters():
            param.requires_grad = False
    
    def forward(self, captions, lengths):
        """
        captions: list of strings
        lengths: list of integers representing the length of each caption
        """
        # Tokenize captions
        tokenized = self.tokenizer(
            captions, 
            padding='longest',
            truncation=True,
            max_length=self.cfg.TEXT.WORDS_NUM,
            return_tensors='pt'
        )
        
        input_ids = tokenized['input_ids'].to(self.device)
        attention_mask = tokenized['attention_mask'].to(self.device)
        
        # Get BERT embeddings
        with torch.no_grad():
            outputs = self.bert_model(
                input_ids=input_ids,
                attention_mask=attention_mask
            )
            
            # Use the last hidden state
            hidden_states = outputs.last_hidden_state
            
            # Create a mask for words (excluding padding)
            mask = attention_mask.unsqueeze(-1).expand(hidden_states.size()).float()
            
            # Apply mask and get word features
            masked_embeddings = hidden_states * mask
            
            # Project to the expected embedding dimension
            word_features = self.projection(masked_embeddings)
            
            # Get sentence feature from [CLS] token (first token)
            sentence_features = self.projection(outputs.pooler_output)
            
        return word_features, sentence_features