import torch
import torch.nn as nn
from transformers import AutoModel

class IndoBERTMultiTask(nn.Module):
    def __init__(self, model_name, num_clickbait_labels, num_kategori_labels):
        super().__init__()
        # Dua encoder BERT terpisah (soft sharing)
        self.bert_clickbait = AutoModel.from_pretrained(model_name)
        self.bert_kategori = AutoModel.from_pretrained(model_name)

        self.dropout = nn.Dropout(0.3)
        hidden_size = self.bert_clickbait.config.hidden_size

        self.clickbait_head = nn.Linear(hidden_size, num_clickbait_labels)
        self.kategori_head = nn.Linear(hidden_size, num_kategori_labels)

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        clickbait_labels=None,
        kategori_labels=None
    ):
        # Encoder untuk clickbait
        clickbait_output = self.bert_clickbait(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids
        )
        pooled_clickbait = clickbait_output.last_hidden_state[:, 0, :]
        pooled_clickbait = self.dropout(pooled_clickbait)
        clickbait_logits = self.clickbait_head(pooled_clickbait)

        # Encoder untuk kategori
        kategori_output = self.bert_kategori(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids
        )
        pooled_kategori = kategori_output.last_hidden_state[:, 0, :]
        pooled_kategori = self.dropout(pooled_kategori)
        kategori_logits = self.kategori_head(pooled_kategori)

        # Hitung loss jika ada label
        loss = None
        loss_clickbait = None
        loss_kategori = None
        if clickbait_labels is not None and kategori_labels is not None:
            loss_fct = nn.CrossEntropyLoss()
            loss_clickbait = loss_fct(clickbait_logits, clickbait_labels)
            loss_kategori = loss_fct(kategori_logits, kategori_labels)
            loss = loss_clickbait + loss_kategori

        return {
            "loss": loss,
            "loss_clickbait": loss_clickbait,
            "loss_kategori": loss_kategori,
            "clickbait_logits": clickbait_logits,
