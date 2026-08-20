#!/usr/bin/env python
# adaptive_ascent_plus_descent.py

import torch
from transformers import Trainer, DataCollatorWithPadding
from torch.utils.data import SequentialSampler
from typing import Optional
import inspect
import math
import os, json

class GrailAscentPlusDescentTrainer(Trainer):
    def __init__(
            self, 
            UR_pt=None,
            RR_pt=None,
            *args, 
            **kwargs):
        
        super().__init__(*args, **kwargs)

        self.UR_pt = UR_pt
        self.RR_pt = RR_pt

    def compute_loss(self, model, inputs, return_outputs=False):
        if "factor" not in inputs:
            print("Warning: factor is not in inputs")
            return super().compute_loss(model, inputs, return_outputs)
        
        # Tokenized Factor dict
        #     (0,0): -1.0,  # privacy_unlearn
        #     (0,1):  1.0,  # privacy_retention
        #     (1,0): -2.0,  # copyright_unlearn
        #     (1,1):  2.0   # copyright_retention

        sample_factors = inputs.pop("factor")

        if self.args.unlearn_method == "grail":
            unlearn_rate = -0.4
            retention_rate = 2.0

            sample_factors = torch.where(sample_factors == -1.0, torch.tensor(unlearn_rate), sample_factors) # 1) privacy_unlearn (dom=0, seg=0)
            sample_factors = torch.where(sample_factors == 1.0, torch.tensor(retention_rate), sample_factors) # 2) privacy_retention (dom=0, seg=1)
            sample_factors = torch.where(sample_factors == -2.0, torch.tensor(unlearn_rate), sample_factors) # 3) copyright_unlearn (dom=1, seg=0)
            sample_factors = torch.where(sample_factors == 2.0, torch.tensor(retention_rate), sample_factors) # 4) copyright_retention (dom=1, seg=1)

        outputs = model(**inputs)
        logits = outputs.logits
        labels = inputs["labels"]

        loss_fct = torch.nn.CrossEntropyLoss(reduction="none")
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()
        loss = loss_fct(
            shift_logits.view(-1, shift_logits.size(-1)),
            shift_labels.view(-1)
        )
        valid_counts = (shift_labels != -100).sum(dim=-1).float()
        loss = loss.view(shift_logits.size(0), -1)
        loss = loss.sum(dim=-1) / valid_counts  # shape=[batch_size]

        # multiply factor
        adjusted_loss = (loss * sample_factors).mean()

        return (adjusted_loss, outputs) if return_outputs else adjusted_loss

    def training_step(self, model, inputs):
        if "factor" not in inputs:
            print("### Warning: No factor")
        else:
            factors=inputs["factor"]

        model.train()
        loss, outputs = self.compute_loss(model, inputs, return_outputs=True)
        loss.backward()

        factor_val = None

        if factors.shape[0] == 1:
            factor_val = factors[0].item()
        else:
            print("### Warning: batch_size is more than 1")

        if factor_val is not None:
            if factor_val == -2.0: 
                ds_type = "copyright_unlearn"
            elif factor_val == 2.0:
                ds_type = "copyright_retention"
            elif factor_val == -1.0:
                ds_type = "privacy_unlearn"
            elif factor_val == 1.0:
                ds_type = "privacy_retention"
            else:
                print("Warning: factor NOT matched")
                print("factor_val:", factor_val)
                ds_type = None
        else:
            print("Warning: factor_val is None")
        #print("### we are in training_step")
        for n, p in model.named_parameters():
            n = n.replace("module.", "", 1)
            if p.grad is not None:
                if 'lora' in n:
                    if ds_type == "privacy_unlearn" and self.UR_pt:
                        mask_dict = self.UR_pt
                        if n in mask_dict:
                            freeze_idx_tensor = mask_dict[n]
                            grad_flat = p.grad.view(-1)
                            grad_flat[freeze_idx_tensor] = 0.0


                    elif ds_type == "copyright_unlearn" and self.UR_pt:
                        mask_dict = self.UR_pt
                        if n in mask_dict:
                            freeze_idx_tensor = mask_dict[n]
                            grad_flat = p.grad.view(-1)
                            grad_flat[freeze_idx_tensor] = 0.0

                    elif ds_type == "privacy_retention" and self.RR_pt:
                        mask_dict = self.RR_pt
                        if n in mask_dict:
                            freeze_idx_tensor = mask_dict[n]
                            grad_flat = p.grad.view(-1)
                            grad_flat[freeze_idx_tensor] = 0.0

                    elif ds_type == "copyright_retention" and self.RR_pt:
                        mask_dict = self.RR_pt
                        if n in mask_dict:
                            freeze_idx_tensor = mask_dict[n]
                            grad_flat = p.grad.view(-1)
                            grad_flat[freeze_idx_tensor] = 0.0
                    

        return loss.detach()


    def _get_train_sampler(self) -> Optional[torch.utils.data.Sampler]:
        return SequentialSampler(self.train_dataset)

    def _set_signature_columns_if_needed(self):
        if self._signature_columns is None:
            signature = inspect.signature(self.model.forward)
            self._signature_columns = list(signature.parameters.keys())
            self._signature_columns += list(set(["label", "label_ids"] + self.label_names))
            self._signature_columns.append('factor')


class GrailAscentPlusDescentDataCollator(DataCollatorWithPadding):
    def __call__(self, features):
        batch = super().__call__(features)
        if "factor" in features[0].keys():
            batch["factor"] = torch.tensor([f["factor"] for f in features], dtype=torch.float)
        return batch

# class AdaptiveAscentPlusDescentDataCollator(DataCollatorWithPadding):
#     def __call__(self, features):
#         batch = super().__call__(features)
#         if "factor" in features[0].keys():
#             batch["factor"] = torch.tensor([f["factor"] for f in features], dtype=torch.float)
#         return batch