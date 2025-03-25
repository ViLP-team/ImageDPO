#    Copyright 2023 Haotian Liu
#
#    Licensed under the Apache License, Version 2.0 (the "License");
#    you may not use this file except in compliance with the License.
#    You may obtain a copy of the License at
#
#        http://www.apache.org/licenses/LICENSE-2.0
#
#    Unless required by applicable law or agreed to in writing, software
#    distributed under the License is distributed on an "AS IS" BASIS,
#    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#    See the License for the specific language governing permissions and
#    limitations under the License.


from typing import List, Optional, Tuple, Union

import torch
import torch.nn as nn
from IPython import embed
from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    LlamaConfig,
    LlamaForCausalLM,
    LlamaModel,
)
from transformers.modeling_outputs import CausalLMOutputWithPast

from llava.constants import IMAGE_TOKEN_INDEX

from ..llava_arch import LlavaMetaForCausalLM, LlavaMetaModel


class LlavaConfig(LlamaConfig):
    model_type = "llava"
    # model_type = "llava_custom"


def check_image_list_same(image_list1, image_list2):
    if len(image_list1) != len(image_list2):
        return False
    for i in range(len(image_list1)):
        if len(image_list1[i]) != len(image_list2[i]):
            return False
            for j in range(len(image_list1[i])):
                if not torch.equal(image_list1[i][j], image_list2[i][j]):
                    return False
    return True


class LlavaLlamaModel(LlavaMetaModel, LlamaModel):
    config_class = LlavaConfig

    def __init__(self, config: LlamaConfig):
        super().__init__(config)


class LlavaLlamaForCausalLM(LlamaForCausalLM, LlavaMetaForCausalLM):
    config_class = LlavaConfig

    def __init__(self, config):
        super(LlamaForCausalLM, self).__init__(config)
        self.model = LlavaLlamaModel(config)
        self.pretraining_tp = config.pretraining_tp
        self.vocab_size = config.vocab_size
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        # Initialize weights and apply final processing
        self.post_init()

        # some strange change
        self.buffer_attention_mask = None
        self.overrite_attention = False

    def get_model(self):
        return self.model

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        images: Optional[torch.FloatTensor] = None,
        return_dict: Optional[bool] = None,
        dpo_forward: Optional[bool] = False,
    ) -> Union[Tuple, CausalLMOutputWithPast]:

        # _input_ids = input_ids
        # image_num_arr = torch.sum(_input_ids==IMAGE_TOKEN_INDEX, 1)
        # all_same = torch.all(image_num_arr == image_num_arr[0])

        if inputs_embeds is None:

            # TODO: Need to check if it will effect the existing (non-hybrid or non-multi image) cases.
            # if not all_same and self.overrite_attention:
            #     attention_mask = self.buffer_attention_mask
            #     attention_mask = torch.cat(
            #         [
            #             attention_mask,
            #             attention_mask.new_ones((attention_mask.shape[0], 1)),
            #         ],
            #         dim=-1,
            #     )
            #     self.overrite_attention = False
            (
                input_ids,
                position_ids,
                attention_mask,
                past_key_values,
                inputs_embeds,
                labels,
            ) = self.prepare_inputs_labels_for_multimodal(
                input_ids, position_ids, attention_mask, past_key_values, labels, images
            )

            # # if the input_ids is not a single token, meaning it is input.
            # # otherwise, it is autogressively generated.
            # if not all_same and _input_ids is not None:
            #     # if _input_ids.shape[1] > 1:
            #     self.buffer_attention_mask = attention_mask #[2, 1236]
            #     self.overrite_attention = True
            # else:
            #     # NOTE: TODO: seems like the end of output is not None.
            #     self.overrite_attention = False

        output_dict = super().forward(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            labels=labels,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        if not dpo_forward:
            return output_dict
        else:
            return output_dict["logits"], labels

    def prepare_inputs_for_generation(
        self, input_ids, past_key_values=None, inputs_embeds=None, **kwargs
    ):
        images = kwargs.pop("images", None)
        _inputs = super().prepare_inputs_for_generation(
            input_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            **kwargs
        )
        if images is not None:
            _inputs["images"] = images
        return _inputs


AutoConfig.register("llava", LlavaConfig)
# AutoConfig.register("llava_custom", LlavaConfig)
AutoModelForCausalLM.register(LlavaConfig, LlavaLlamaForCausalLM)
