
import torch
import torch.nn as nn


from modeling_t5 import T5
class T5Story(T5):
    def __init__(self, config):
        super().__init__(config)

    def train_step(self, batch):
        device = next(self.parameters()).device
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)

        lm_labels = batch["target_ids"].to(device)

        reduce_loss = True
        output = self(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=lm_labels,
            reduce_loss=reduce_loss
        )

        lm_mask = lm_labels != -100
        B, L = lm_labels.size()

        loss = output['loss']

        result = {
            'loss': loss
        }
        return result

    def test_step(self, batch, **kwargs):
        device = next(self.parameters()).device
        input_ids = batch['input_ids'].to(device)
        album_id = batch['album_id']

        output = self.generate(
            input_ids=input_ids,
            **kwargs
        )

        generated_sents = self.tokenizer.batch_decode(output, skip_special_tokens=True)

        result = {}
        result['pred'] = generated_sents
        result['album_id'] = album_id

        return result
