#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import random
import argparse

import torch
from torch.utils.data import Dataset
from datasets import load_dataset, Dataset as HFDataset
from transformers import AutoTokenizer, set_seed

# IMPORTANT: the tokenize function from llm_unlearn.utils
# Make sure the path is correct in your project:
from llm_unlearn.utils import tokenize

###############################################################################
# 1) Our Dataset Classes
###############################################################################
class AdvSupervisedDataset(Dataset):
    """
    2-factor (ascent_plus_descent):
      - unlearn => factor = -1
      - retention => factor = +1
      factor=+1 is actually data_args.positive_factor (default=1.0)
    """
    def __init__(self, negative_ds, positive_ds, data_args):
        super().__init__()
        print("Formatting inputs... (AdvSupervisedDataset)")

        # We assume negative_ds / positive_ds are HuggingFace Datasets
        # created by `tokenize(...)`, so each has "input_ids", "labels", "attention_mask"
        neg_dict = negative_ds.to_dict()
        pos_dict = positive_ds.to_dict()

        self.input_ids = []
        self.labels = []
        self.attention_mask = []
        self.factor = []

        # For i in range of the shorter negative set
        for i in range(len(neg_dict["input_ids"])):
            # negative => factor=-1
            self.input_ids.append(neg_dict["input_ids"][i])
            self.labels.append(neg_dict["labels"][i])
            self.attention_mask.append(neg_dict["attention_mask"][i])
            self.factor.append(-1.0)

            # positive => factor= +data_args.positive_factor
            start_idx = i * data_args.positive_ratio
            end_idx   = (i + 1) * data_args.positive_ratio
            self.input_ids.extend(pos_dict["input_ids"][start_idx:end_idx])
            self.labels.extend(pos_dict["labels"][start_idx:end_idx])
            self.attention_mask.extend(pos_dict["attention_mask"][start_idx:end_idx])
            self.factor.extend([data_args.positive_factor] * (end_idx - start_idx))

        # Shuffle
        combined = list(zip(self.input_ids, self.labels, self.attention_mask, self.factor))
        random.shuffle(combined)
        self.input_ids, self.labels, self.attention_mask, self.factor = zip(*combined)

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, index):
        return {
            "input_ids": self.input_ids[index],
            "labels": self.labels[index],
            "attention_mask": self.attention_mask[index],
            "factor": self.factor[index],
        }


class AdaptiveAdvSupervisedDataset(Dataset):
    """
    4-factor (adaptive_ascent_plus_descent):
      - (0,0): -1 => privacy_unlearn
      - (0,1): +1 => privacy_ret
      - (1,0): -2 => copyright_unlearn
      - (1,1): +2 => copyright_ret
    """
    def __init__(self, data_list, factor_map=None):
        super().__init__()
        print("Formatting inputs... (AdaptiveAdvSupervisedDataset)")

        self.input_ids = []
        self.attention_mask = []
        self.labels = []
        self.domain_label = []
        self.segment_label = []
        self.factor = []

        self.factor_map = factor_map or {
            (0,0): -1.0,
            (0,1): +1.0,
            (1,0): -2.0,
            (1,1): +2.0
        }

        for sample in data_list:
            self.input_ids.append(sample["input_ids"])
            self.attention_mask.append(sample["attention_mask"])
            self.labels.append(sample["labels"])
            dom = sample["domain_label"]
            seg = sample["segment_label"]
            self.domain_label.append(dom)
            self.segment_label.append(seg)

            # factor
            f_val = self.factor_map.get((dom, seg), 1.0)
            self.factor.append(f_val)

        # Shuffle
        combined = list(zip(
            self.input_ids, self.attention_mask, self.labels,
            self.domain_label, self.segment_label, self.factor
        ))
        random.shuffle(combined)
        (self.input_ids,
         self.attention_mask,
         self.labels,
         self.domain_label,
         self.segment_label,
         self.factor) = zip(*combined)

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, index):
        return {
            "input_ids": self.input_ids[index],
            "attention_mask": self.attention_mask[index],
            "labels": self.labels[index],
            "domain_label": self.domain_label[index],
            "segment_label": self.segment_label[index],
            "factor": self.factor[index],
        }

###############################################################################
# 2) Dataset path + pair
###############################################################################
our_dataset_path_dict = {
    "privacy_unlearn":    {"name": "privacy",    "split": "unlearn",    "dom": 0, "seg": 0},
    "privacy_retention":  {"name": "privacy",    "split": "retention",  "dom": 0, "seg": 1},
    "copyright_unlearn":  {"name": "copyright",  "split": "unlearn",    "dom": 1, "seg": 0},
    "copyright_retention":{"name": "copyright",  "split": "retention",  "dom": 1, "seg": 1},
}


def load_data_split(key):
    info = our_dataset_path_dict[key]
    raw_list = load_dataset(
        "zjunlp/KnowUnDo",
        name=info["name"],
        split=info["split"],
        cache_dir="../../data"
    )["train"][0]  # list-of-dict

    out = []
    for dic in raw_list:
        merged_text = dic["text"] + "\n\n" + dic["labels"]
        out.append({
            "text": merged_text,
            "labels": "\n\n" + dic["labels"],
            "domain_label": info["dom"],
            "segment_label": info["seg"],
        })
    return out


def pair_unlearn_retention(un_key, ret_key):
    un_list = load_data_split(un_key)
    ret_list = load_data_split(ret_key)

    random.shuffle(un_list)
    random.shuffle(ret_list)
    m = min(len(un_list), len(ret_list))
    un_list = un_list[:m]
    ret_list = ret_list[:m]
    return un_list, ret_list

###############################################################################
# 3) Main
###############################################################################
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tokenizer_name_or_path", type=str, help= "../../models/Qwen1.5-7B-Chat", default="../../models/Llama-2-7b-chat-hf")
    parser.add_argument("--max_length", type=int, default=256)
    parser.add_argument("--output_dir", type=str, default="./tokenized_out")

    # 2-factor
    parser.add_argument("--positive_ratio", type=int, default=1)
    parser.add_argument("--positive_factor", type=float, default=1.0)

    # 4-factor
    parser.add_argument("--privacy_unlearn_factor", type=float, default=-1.0)
    parser.add_argument("--privacy_ret_factor", type=float, default=1.0)
    parser.add_argument("--copyright_unlearn_factor", type=float, default=-2.0)
    parser.add_argument("--copyright_ret_factor", type=float, default=2.0)

    args = parser.parse_args()
    set_seed(42)
    random.seed(42)

    # 1) privacy
    p_un, p_ret = pair_unlearn_retention("privacy_unlearn", "privacy_retention")
    # 2) copyright
    c_un, c_ret = pair_unlearn_retention("copyright_unlearn", "copyright_retention")
    # combine
    big_un = p_un + c_un
    big_ret = p_ret + c_ret

    print(f"[INFO] privacy_un={len(p_un)}, ret={len(p_ret)} | copyright_un={len(c_un)}, ret={len(c_ret)}")
    print(f"[INFO] big_un={len(big_un)}, big_ret={len(big_ret)}")

    # Convert to HF Dataset
    from datasets import Dataset as HFDataset

    # unlearn
    un_cols = {"text": [], "labels": [], "domain_label": [], "segment_label": []}
    for x in big_un:
        un_cols["text"].append(x["text"])
        un_cols["labels"].append(x["labels"])
        un_cols["domain_label"].append(x["domain_label"])
        un_cols["segment_label"].append(x["segment_label"])
    unlearn_ds = HFDataset.from_dict(un_cols)

    # retention
    ret_cols = {"text": [], "labels": [], "domain_label": [], "segment_label": []}
    for x in big_ret:
        ret_cols["text"].append(x["text"])
        ret_cols["labels"].append(x["labels"])
        ret_cols["domain_label"].append(x["domain_label"])
        ret_cols["segment_label"].append(x["segment_label"])
    retention_ds = HFDataset.from_dict(ret_cols)

    # 2) Tokenize
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_name_or_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print("[INFO] tokenize unlearn_ds")
    unlearn_tok = tokenize(unlearn_ds, tokenizer, args.max_length)
    print("[INFO] tokenize retention_ds")
    retention_tok = tokenize(retention_ds, tokenizer, args.max_length)

    #=== (A) Ascent
    from argparse import Namespace
    asc_args = Namespace(
        positive_ratio=args.positive_ratio,
        positive_factor=args.positive_factor
    )
    asc_dataset = AdvSupervisedDataset(unlearn_tok, retention_tok, asc_args)
    print("[INFO] asc_dataset length=", len(asc_dataset))

    #=== (B) Adaptive
    factor_map = {
        (0,0): args.privacy_unlearn_factor,
        (0,1): args.privacy_ret_factor,
        (1,0): args.copyright_unlearn_factor,
        (1,1): args.copyright_ret_factor
    }

    # Convert unlearn_tok / retention_tok to list-of-dict
    def hf_to_list(hf_ds):
        dd = hf_ds.to_dict()
        result = []
        for i in range(len(dd["input_ids"])):
            result.append({
                "input_ids": dd["input_ids"][i],
                "attention_mask": dd["attention_mask"][i],
                "labels": dd["labels"][i],   # rename to "labels"
                "domain_label": dd["domain_label"][i],
                "segment_label": dd["segment_label"][i],
            })
        return result

    un_list_adapt = hf_to_list(unlearn_tok)
    ret_list_adapt = hf_to_list(retention_tok)
    combined_adapt_list = un_list_adapt + ret_list_adapt
    adapt_dataset = AdaptiveAdvSupervisedDataset(combined_adapt_list, factor_map=factor_map)
    print("[INFO] adapt_dataset length=", len(adapt_dataset))

    # 3) Save (both the 2-factor and the 4-factor GRAIL tokenized datasets)
    #    grail_run_adaptive_unlearn.py consumes the 4-factor (adaptive) dataset.
    asc_path   = os.path.join(args.output_dir, "ascent_plus_descent", "tokenized_dataset.pt")
    adapt_path = os.path.join(args.output_dir, "adaptive_ascent_plus_descent", "tokenized_dataset.pt")
    os.makedirs(os.path.dirname(asc_path), exist_ok=True)
    os.makedirs(os.path.dirname(adapt_path), exist_ok=True)

    torch.save(asc_dataset, asc_path)
    print(f"[DONE] saved: {asc_path}")

    torch.save(adapt_dataset, adapt_path)
    print(f"[DONE] saved: {adapt_path}")


if __name__ == "__main__":
    main()
