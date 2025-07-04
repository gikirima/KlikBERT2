import torch
import torch.nn as nn
from transformers import AutoModel, PreTrainedModel, AutoConfig

class IndoBERTMultiTaskConfig(AutoConfig):
    model_type = "indobert-multitask"

    def __init__(self, num_clickbait_labels=4, num_kategori_labels=9, **kwargs):
        super().__init__(**kwargs)
        self.num_clickbait_labels = num_clickbait_labels
        self.num_kategori_labels = num_kategori_labels


class IndoBERTMultiTask(PreTrainedModel):
    config_class = IndoBERTMultiTaskConfig

    def __init__(self, config):
        super().__init__(config)
        self.bert = AutoModel.from_pretrained(config._name_or_path)
        self.dropout = nn.Dropout(0.3)

        hidden_size = self.bert.config.hidden_size
        self.clickbait_head = nn.Linear(hidden_size, config.num_clickbait_labels)
        self.kategori_head = nn.Linear(hidden_size, config.num_kategori_labels)

        self.init_weights()

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        clickbait_labels=None,
        kategori_labels=None
    ):
        outputs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
        )
        pooled = outputs.last_hidden_state[:, 0, :]
        pooled = self.dropout(pooled)

        clickbait_logits = self.clickbait_head(pooled)
        kategori_logits = self.kategori_head(pooled)

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
            "kategori_logits": kategori_logits,
        }
