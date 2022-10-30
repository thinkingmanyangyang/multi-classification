# 定义分类器，如果没有足够算力，建议使用rnn，cnn分类器
import torch
from torch import nn
from torch.nn import CrossEntropyLoss
from loss_fct import FocalLoss
# from torch.nn.functional import binary_cross_entropy, binary_cross_entropy_with_logits
from transformers import BertConfig
from modeling import BertModel, BertPreTrainedModel
from constant import *
# class MultiClassification(BertPreTrainedModel):
#     def __init__(self, config):
#         super(MultiClassification, self).__init__(config)
#         self.num_labels = config.num_labels
#         self.bert = BertModel(config)
#         self.dropout = nn.Dropout(config.hidden_dropout_prob)
#         self.classifier = nn.Linear(config.hidden_size, 1)
#         # self.loss_fct = nn.BCELoss()
#         self.loss_fct = FocalLoss(gamma=1, num_class=self.num_labels)
#         self.init_weights()
#     def forward(self,
#                 input_ids=None,
#                 attention_mask=None,
#                 token_type_ids=None,
#                 position_ids=None,
#                 head_mask=None,
#                 inputs_embeds=None,
#                 labels=None,
#                 ):
#         device = input_ids.device
#         outputs = self.bert(
#             input_ids,
#             attention_mask=attention_mask,
#             token_type_ids=token_type_ids,
#             position_ids=position_ids,
#             head_mask=head_mask,
#             inputs_embeds=inputs_embeds,
#         )
#         sequence_output = outputs[0]
#         sequence_output = sequence_output[:, 1:self.num_labels+1]
#         pooled_output = outputs[1]
#
#         logits = self.classifier(sequence_output)
#         logits = logits.view(-1, self.num_labels)
#
#         # probs = torch.sigmoid(logits)
#         # loss = self.loss_fct(probs, labels) * 0.5
#
#         graph = torch.tensor(RELATION_GRAPH).to(device).to(logits.dtype)
#         logits = torch.matmul(logits, graph)
#         logits = torch.sigmoid(logits)
#
#         labels = labels.view(-1, self.num_labels)
#         loss = self.loss_fct(logits, labels)
#         return loss, logits


class MultiClassification(BertPreTrainedModel):
    def __init__(self, config):
        super(MultiClassification, self).__init__(config)
        self.num_labels = config.num_labels
        self.bert = BertModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = nn.Linear(config.hidden_size, config.num_labels)
        self.loss_fct = nn.BCELoss()
        # self.loss_fct = FocalLoss(gamma=1, num_class=self.num_labels)
        self.init_weights()
    def forward(self,
                input_ids=None,
                attention_mask=None,
                token_type_ids=None,
                position_ids=None,
                head_mask=None,
                inputs_embeds=None,
                labels=None,
                ):
        outputs = self.bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
        )
        sequence_output = outputs[0]
        sequence_output = sequence_output[:, 1:self.num_labels+1]
        pooled_output = outputs[1]

        logits = self.classifier(pooled_output)
        logits = logits.view(-1, self.num_labels)
        logits = torch.sigmoid(logits)

        loss = self.loss_fct(logits, labels)
        return loss, logits

if __name__ == "__main__":
    input_ids = torch.tensor([[1,2,3,4,5]], dtype=torch.long)
    config = BertConfig()
    config.num_hidden_layers = 3
    config.num_labels = 9
    model = MultiClassification(config=config)
    model(input_ids = input_ids)

