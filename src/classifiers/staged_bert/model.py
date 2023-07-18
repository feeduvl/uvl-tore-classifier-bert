from collections.abc import Iterable
from typing import Any
from typing import Dict
from typing import List
from typing import Literal
from typing import Optional
from typing import Tuple
from typing import TypedDict
from typing import Union

import torch
from torch import nn
from torch.nn import CrossEntropyLoss
from transformers.modeling_outputs import (
    TokenClassifierOutput,
)
from transformers.models.bert import BertConfig
from transformers.models.bert import BertModel
from transformers.models.bert import BertPreTrainedModel
from transformers.tokenization_utils_base import PreTrainedTokenizerBase
from transformers.utils import PaddingStrategy


class StagedBertForTokenClassification(BertPreTrainedModel):  # type: ignore
    config_class = BertConfig

    def __init__(self, config: BertConfig, num_hint_labels: int):
        super().__init__(config)
        self.num_labels = config.num_labels
        self.num_hint_labels = num_hint_labels

        self.bert = BertModel(config, add_pooling_layer=False)
        classifier_dropout = (
            config.classifier_dropout
            if config.classifier_dropout is not None
            else config.hidden_dropout_prob
        )
        self.dropout = nn.Dropout(classifier_dropout)
        self.c1 = nn.Linear(
            config.hidden_size + num_hint_labels,
            config.hidden_size + num_hint_labels,
        )
        self.c2 = nn.Linear(
            config.hidden_size + num_hint_labels,
            config.hidden_size + num_hint_labels,
        )
        self.c3 = nn.Linear(
            config.hidden_size + num_hint_labels,
            config.hidden_size + num_hint_labels,
        )
        self.classifier = nn.Linear(
            config.hidden_size + num_hint_labels, config.num_labels
        )

        # Initialize weights and apply final processing
        self.post_init()

    def forward(
        self,
        input_ids: torch.Tensor,
        hint_input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        token_type_ids: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple[torch.Tensor], TokenClassifierOutput]:
        r"""
        labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
            Labels for computing the token classification loss. Indices should be in `[0, ..., config.num_labels - 1]`.
        """
        return_dict = (
            return_dict
            if return_dict is not None
            else self.config.use_return_dict
        )

        hint_input_ids = nn.functional.one_hot(
            hint_input_ids, self.num_hint_labels
        )

        outputs = self.bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        sequence_output = outputs[0]

        sequence_output = self.dropout(sequence_output)

        # merge the dropout bert output with the one hot encoded hint data and pass it to the
        # classifier
        classifier_input = torch.cat([sequence_output, hint_input_ids], dim=2)

        c1 = self.c1(classifier_input)
        c2 = self.c2(c1)
        c3 = self.c3(c2)

        logits = self.classifier(c3)

        loss = None
        if labels is not None:
            loss_fct = CrossEntropyLoss()
            loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))

        if not return_dict:
            output = (logits,) + outputs[2:]
            return ((loss,) + output) if loss is not None else output

        return TokenClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )
