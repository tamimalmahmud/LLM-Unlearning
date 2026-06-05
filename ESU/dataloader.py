import torch
from torch import nn
from transformers import Trainer
import torch.nn.functional as F
import os
from evaluate_util import get_dataloader, get_all_evals
import json
from pathlib import Path
from data_module import get_batch_loss
from utils import merge_dicts, interleave_eval_result_dict, get_forget_quality, get_model_utility
import numpy as np
from scipy.stats import ks_2samp, hmean
import csv

def printll(name, inp):
    print(name, [round(x, 4) for x in inp])

class CustomTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        input_ids, labels, attention_mask = inputs
        outputs = model(input_ids, labels=labels, attention_mask=attention_mask)
        loss = outputs.loss
        return (loss, outputs) if return_outputs else loss
    
    def prediction_step(self, model, inputs, prediction_loss_only: bool, ignore_keys=None):
        input_ids, labels, attention_mask = inputs
        with torch.no_grad():
            outputs = model(input_ids, labels=labels, attention_mask=attention_mask)
            logits = outputs.logits
            loss = outputs.loss
        return (loss, logits, labels)

class CustomTrainerForgetting(Trainer):
    def __init__(self, *args, **kwargs):
        self.loss_type = kwargs.pop('forget_loss')
        self.eval_cfg = kwargs.pop('eval_cfg', None)
        super().__init__(*args, **kwargs)

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        if self.loss_type == "grad_ascent":
            forget_inputs, retain_inputs = inputs
            input_ids, labels, attention_mask = forget_inputs
            outputs = model(input_ids, labels=labels, attention_mask=attention_mask)
            loss = outputs.loss * -1

        elif self.loss_type == "grad_diff":
            forget_inputs, retain_inputs = inputs
            input_ids, labels, attention_mask = forget_inputs
            outputs = model(input_ids, labels=labels, attention_mask=attention_mask)
            forget_loss = outputs.loss * -1

            retain_input_ids, retain_labels, retain_attention_mask = retain_inputs
            retain_outputs = model(retain_input_ids, labels=retain_labels, attention_mask=retain_attention_mask)
            retain_loss = retain_outputs.loss
            loss = forget_loss + retain_loss

        elif self.loss_type == "KL":
            forget_inputs, retain_inputs = inputs
            input_ids, labels, attention_mask = forget_inputs
            outputs = model(input_ids, labels=labels, attention_mask=attention_mask)
            forget_loss = outputs.loss * -1

            retain_input_ids, retain_labels, retain_attention_mask = retain_inputs
            with torch.no_grad():
                retain_outputs = model(retain_input_ids, labels=retain_labels, attention_mask=retain_attention_mask)

            retain_probs = F.log_softmax(retain_outputs.logits, dim=-1).view(-1, retain_outputs.logits.shape[-1])
            current_outputs = model(retain_input_ids, labels=retain_labels, attention_mask=retain_attention_mask)
            current_probs = F.log_softmax(current_outputs.logits, dim=-1).view(-1, current_outputs.logits.shape[-1])

            retain_loss = F.kl_div(current_probs, retain_probs, reduction='batchmean', log_target=True)
            loss = forget_loss + retain_loss

        elif self.loss_type == "idk":
            idk_inputs, retain_inputs = inputs
            idk_input_ids, idk_labels, idk_attention_mask = idk_inputs
            retain_input_ids, retain_labels, retain_attention_mask = retain_inputs
            
            input_ids = torch.cat((idk_input_ids, retain_input_ids), dim=0)
            labels = torch.cat((idk_labels, retain_labels), dim=0)
            attention_mask = torch.cat((idk_attention_mask, retain_attention_mask), dim=0)
            
            outputs = model(input_ids, labels=labels, attention_mask=attention_mask)
            loss = outputs.loss
        
        elif self.loss_type == "dpo":
            if len(inputs) == 3:
                idk_inputs, forget_inputs, retain_inputs = inputs
                idk_input_ids, idk_labels, idk_attention_mask = idk_inputs
            else:
                idk_input_ids = idk_labels = idk_attention_mask = None

            forget_input_ids, forget_labels, forget_attention_mask = forget_inputs
            retain_input_ids, retain_labels, retain_attention_mask = retain_inputs

            forget_outputs = model(forget_input_ids, labels=forget_labels, attention_mask=forget_attention_mask)
            retain_outputs = model(retain_input_ids, labels=retain_labels, attention_mask=retain_attention_mask)

            if idk_input_ids is not None:
                idk_outputs = model(idk_input_ids, labels=idk_labels, attention_mask=idk_attention_mask)
                pi_logratios = get_batch_loss(idk_outputs.logits, idk_labels) - get_batch_loss(forget_outputs.logits, forget_labels)
            else:
                pi_logratios = get_batch_loss(retain_outputs.logits, retain_labels) - get_batch_loss(forget_outputs.logits, forget_labels)

            ref_logratios = get_batch_loss(retain_outputs.logits, retain_labels) - get_batch_loss(forget_outputs.logits, forget_labels)

            beta = 0.1
            loss = -F.logsigmoid(beta * (pi_logratios - ref_logratios)).mean()

        return (loss, forget_outputs) if return_outputs else loss

    def evaluate(self, eval_dataset=None, ignore_keys=None, metric_key_prefix="eval"):
        args = self.args
        model = self._wrap_model(self.model, training=False)
        eval_cfg = self.eval_cfg
        model.eval()
        curr_step = self.state.global_step
        curr_save_dir = os.path.join(eval_cfg.save_dir, f"checkpoint-{curr_step}")
        Path(curr_save_dir).mkdir(parents=True, exist_ok=True)

        aggregated_eval_logs = {}
        with torch.no_grad():
            for task, folder, split in zip(eval_cfg.eval_task, eval_cfg.data_path, eval_cfg.split_list):
                eval_dataloader, base_eval_dataloader, perturb_dataloader = get_dataloader(
                    eval_cfg, task, self.tokenizer, folder, split
                )
                eval_logs = get_all_evals(eval_cfg, model, self.tokenizer, task, eval_dataloader, base_eval_dataloader, perturb_dataloader)
                aggregated_eval_logs[task] = eval_logs

                with open(os.path.join(curr_save_dir, f"{task}.json"), "w") as f:
                    json.dump(eval_logs, f, indent=4)

        return aggregated_eval_logs

def custom_data_collator_forget(samples):
    if len(samples[0]) == 3:
        idk_samples, forget_samples, retain_samples = [s[0] for s in samples], [s[1] for s in samples], [s[2] for s in samples]
        rets = []
        for data_type in ["idk", "forget", "retain"]:
            data = idk_samples if data_type == "idk" else (forget_samples if data_type == "forget" else retain_samples)
            input_ids = [s[0] for s in data]
            labels = [s[1] for s in data]
            attention_mask = [s[2] for s in data]
            rets.append((torch.stack(input_ids), torch.stack(labels), torch.stack(attention_mask)))
        return rets
    else:
        forget_samples, retain_samples = [s[0] for s in samples], [s[1] for s in samples]
        rets = []
        for data_type in ["forget", "retain"]:
            data = forget_samples if data_type == "forget" else retain_samples
            input_ids = [s[0] for s in data]
            labels = [s[1] for s in data]
            attention_mask = [s[2] for s in data]
            rets.append((torch.stack(input_ids), torch.stack(labels), torch.stack(attention_mask)))
        return rets

def compute_metrics(pred):
    logits, labels = torch.from_numpy(pred.predictions), torch.from_numpy(pred.label_ids)
    preds = torch.from_numpy(pred.predictions.argmax(-1))
    shifted_labels = labels[..., 1:].contiguous()
    acc = torch.mean((preds[..., :-1] == shifted_labels).float())
    loss = get_loss(logits, labels)
    return {"eval accuracy": acc, "eval loss": loss.item()}

def get_loss(output, labels):
    shifted_labels = labels[..., 1:].contiguous()
    output = output[..., :-1, :].contiguous()
    loss_function = nn.CrossEntropyLoss(ignore_index=-100)
    return loss_function(output.view(-1, output.size(-1)), shifted_labels.view(-1))
