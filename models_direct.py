import torch.nn as nn
from transformers import BertForMultipleChoice

class LinearClassifier(BertForMultipleChoice):
    def __init__(self,config):
        super().__init__(config)

        self.classifier=nn.Sequential(
            nn.Dropout(config.hidden_dropout_prob),
            nn.Linear(config.hidden_size,256),
            nn.BatchNorm1d(256),
            nn.LeakyReLU(0.2),
            nn.Linear(256,20),
            nn.BatchNorm1d(20),
            nn.LeakyReLU(0.2),
            nn.Linear(20,1)
        )

        self.init_weights()

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        labels=None,
    ):
        num_choices = input_ids.size(1)

        input_ids = input_ids.view(-1, input_ids.size(-1))
        attention_mask = attention_mask.view(-1, attention_mask.size(-1))
        token_type_ids = token_type_ids.view(-1, token_type_ids.size(-1))

        outputs = self.bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
        )

        pooled_output = outputs[1]
        logits = self.classifier(pooled_output)
        reshaped_logits = logits.view(-1, num_choices)

        loss = None
        if labels is not None:
            criterion = nn.CrossEntropyLoss()
            loss = criterion(reshaped_logits, labels)

        return (loss,reshaped_logits)
