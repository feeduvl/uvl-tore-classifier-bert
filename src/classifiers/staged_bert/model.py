import math
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
import numpy as np
from torch import nn
from torch.nn import CrossEntropyLoss
from transformers.modeling_outputs import (
    TokenClassifierOutput,
)
from transformers.models.bert import BertConfig
from transformers.models.bert import BertModel
from transformers.models.bert import BertPreTrainedModel


class StagedBertModelConfig(BertConfig):
    model_type = "staged_bert"

    def __init__(
        self,
        device: str = "cpu",
        num_hint_labels: int = 0,
        layers: List[int] = [],
        weights: Optional[List[float]] = None,
        **kwargs: Any,
    ):
        super().__init__(**kwargs)
        self.num_hint_labels = num_hint_labels
        self.layers = layers
        self.weights = weights
        self.device = device


class StagedBertForTokenClassification(BertPreTrainedModel):
    # https://colab.research.google.com/drive/1ZLfcB16Et9U2V-udrw8zwrfChFCIhomz?usp=sharing#scrollTo=m-TTyOMJOGBD
    config_class = StagedBertModelConfig

    def __init__(
        self,
        config: StagedBertModelConfig,
    ):
        super().__init__(config)
        self.num_labels = config.num_labels
        self.num_hint_labels = config.num_hint_labels
        print("Test")
        self.weights = (
            torch.from_numpy(np.array(config.weights))
            .to(torch.float32)
            .to(device=config.device)
        )

        self.bert = BertModel(config, add_pooling_layer=False)
        classifier_dropout = (
            config.classifier_dropout
            if config.classifier_dropout is not None
            else config.hidden_dropout_prob
        )
        self.dropout = nn.Dropout(classifier_dropout)

        in_size = config.hidden_size + (config.num_hint_labels)
        layers = [layer for layer in config.layers]

        layers.insert(0, in_size)
        layers.append(config.num_labels)

        out_layers: List[nn.Linear | nn.ReLU] = []

        for idx in range(len(layers) - 1):
            _in = layers[idx]
            _out = layers[idx + 1]
            _layer = nn.Linear(_in, _out)
            out_layers.append(_layer)
            out_layers.append(nn.ReLU())

        out_layers.pop(-1)  # remove last relu

        self.out_layers = nn.Sequential(*out_layers)

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
        c = torch.cat([sequence_output, hint_input_ids], dim=2)

        logits = self.out_layers(c)

        loss = None
        if labels is not None:
            loss_fct = CrossEntropyLoss(weight=self.weights)
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
