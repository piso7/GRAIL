import os
import torch
from torch.utils.data import Dataset
import datasets
import sys
# config.py sits next to this file, in pretrain/
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from config import get_model_identifiers_from_yaml, add_dataset_index
import random

def convert_to_model_format_with_random_label(tokenizer, max_length,  question, answer, model_configs):
    question_start_token, question_end_token, answer_token = model_configs['question_start_tag'], model_configs['question_end_tag'], model_configs['answer_tag']
    new_question = question_start_token + question + question_end_token
    new_answer = answer_token + answer
    full_text = new_question + new_answer
    num_question_tokens = len(tokenizer.tokenize(new_question, add_special_tokens=True))

    encoded = tokenizer(
        full_text, 
        add_special_tokens=True, 
        max_length=max_length, 
        truncation=True, 
    )

    # random replace half tokens in label
    for _ in range((len(encoded.input_ids) - num_question_tokens) // 2):
        replace_idx = random.randint(num_question_tokens, len(encoded.input_ids) - 1)
        # make sure this replaced token is not special token
        encoded.input_ids[replace_idx] = random.randint(0, tokenizer.vocab_size - 1)
        while encoded.input_ids[replace_idx] is tokenizer.pad_token or encoded.input_ids[replace_idx] is tokenizer.eos_token_id or encoded.input_ids[replace_idx] is tokenizer.unk_token_id or encoded.input_ids[replace_idx] is tokenizer.bos_token_id:
            encoded.input_ids[replace_idx] = random.randint(0, tokenizer.vocab_size-1)
    
    pad_length = max_length - len(encoded.input_ids)
    pad_input_ids = encoded['input_ids'] + [tokenizer.eos_token_id] * pad_length
    pad_attention_mask = encoded['attention_mask'] + [0] * pad_length
    if len(encoded.input_ids) == max_length:
        label = encoded.input_ids
    else:
        label = encoded['input_ids'] + [tokenizer.eos_token_id] + [-100] * (pad_length-1)

    # change label to -100 for question tokens
    for i in range(num_question_tokens): label[i] = -100

    return torch.tensor(pad_input_ids),torch.tensor(label),torch.tensor(pad_attention_mask)


def convert_raw_data_to_model_format(tokenizer, max_length,  question, answer, model_configs):
    question_start_token, question_end_token, answer_token = model_configs['question_start_tag'], model_configs['question_end_tag'], model_configs['answer_tag']
    new_question = question_start_token + question + question_end_token
    new_answer = answer_token + answer
    full_text = new_question + new_answer
    num_question_tokens = len(tokenizer.tokenize(new_question, add_special_tokens=True))

    encoded = tokenizer(
        full_text, 
        add_special_tokens=True, 
        max_length=max_length, 
        truncation=True, 
    )
    pad_length = max_length - len(encoded.input_ids)
    pad_input_ids = encoded['input_ids'] + [tokenizer.eos_token_id] * pad_length
    pad_attention_mask = encoded['attention_mask'] + [0] * pad_length
    if len(encoded.input_ids) == max_length:
        label = encoded.input_ids
    else:
        label = encoded['input_ids'] + [tokenizer.eos_token_id] + [-100] * (pad_length-1)

    # change label to -100 for question tokens
    for i in range(num_question_tokens): label[i] = -100

    return torch.tensor(pad_input_ids),torch.tensor(label),torch.tensor(pad_attention_mask)

def replicate_samples(examples, num_copies=2):
    replicated_examples = {key: [] for key in examples.keys()}
    for key in examples.keys():
        for value in examples[key]:
            replicated_examples[key].extend([value] * num_copies)
            
    return replicated_examples

def expand_dataset(dataset, num_copies=2, split="unlearn"):
    expanded_dataset = dataset.map(
        lambda examples: replicate_samples(examples, num_copies=num_copies),
        batched=True,
        new_fingerprint=f"None_{split}_{num_copies}"
    )
    return expanded_dataset

class TextDatasetRandomQA(Dataset):
    def __init__(self, data_path, tokenizer, model_family, max_length=512, split = None, question_key='text', answer_key='labels', num_copies=2, data_type="default"):
        super(TextDatasetRandomQA, self).__init__()
        self.tokenizer = tokenizer
        self.max_length = max_length

        if data_type != 'default':
            raw_data = []

            if data_type == "combined":
                # Combined 데이터 로드 및 결합
                if split == "unlearn" or split == "retention":
                    privacy_data = datasets.load_dataset(
                        data_path, name="privacy", split=split, cache_dir="../data"
                    )[0]
                    copyright_data = datasets.load_dataset(
                        data_path, name="copyright", split=split, cache_dir="../data"
                    )[0]

                    # privacy와 copyright 데이터를 결합
                    privacy_train, privacy_val = privacy_data['train'], privacy_data['val']
                    copyright_train, copyright_val = copyright_data['train'], copyright_data['val']
                    raw_data.extend(privacy_train)
                    raw_data.extend(privacy_val)
                    raw_data.extend(copyright_train)
                    raw_data.extend(copyright_val)

                    data_ = datasets.Dataset.from_dict({key: [dic[key] for dic in raw_data] for key in raw_data[0]})

                    # 셔플 적용
                    data_ = data_.shuffle(seed=42)

            elif split == "unlearn" or split == "retention":
                data = datasets.load_dataset(data_path, name=data_type, split=split, cache_dir="../data")[0]

                data_train, data_val = data['train'], data['val']
                raw_data.extend(data_train)
                raw_data.extend(data_val)

                data_ = datasets.Dataset.from_dict({key: [dic[key] for dic in raw_data] for key in raw_data[0]})
            else:
                assert ValueError("split not supported")

            

        self.data = expand_dataset(data_, num_copies=num_copies, split=split)
        self.data = add_dataset_index(self.data)
        self.model_configs = get_model_identifiers_from_yaml(model_family)
        self.qk = question_key
        self.ak = answer_key

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        question = self.data[idx][self.qk]
        answers = self.data[idx][self.ak]
        indices = self.data[idx]['index']
        if isinstance(answers, str):
            answers = [answers]

        pad_input_ids_list = []
        label_list = []
        pad_attention_mask_list = []

        for answer in answers:
            converted_data = convert_to_model_format_with_random_label(self.tokenizer, self.max_length, question, answer, self.model_configs)
            pad_input_ids_list.append(converted_data[0])
            label_list.append(converted_data[1])
            pad_attention_mask_list.append(converted_data[2])


        return torch.stack(pad_input_ids_list).squeeze(), \
                torch.stack(label_list).squeeze(), \
                torch.stack(pad_attention_mask_list).squeeze(), \
                torch.tensor(indices)

class TextDatasetQA(Dataset):
    def __init__(self, data_path, tokenizer, model_family, max_length=512, split = None, question_key='text', answer_key='labels', data_type='default'):
        super(TextDatasetQA, self).__init__()
        self.tokenizer = tokenizer
        self.max_length = max_length

        if data_type != 'default':
            raw_data = []
            if split == "full":
                unlearn_data = datasets.load_dataset(data_path, name=data_type, split='unlearn', cache_dir="../data")[0]
                retention_data = datasets.load_dataset(data_path, name=data_type, split='retention', cache_dir="../data")[0]

                data_train, data_val = unlearn_data['train'], unlearn_data['val']
                raw_data.extend(data_train)
                raw_data.extend(data_val)

                data_train, data_val = retention_data['train'], retention_data['val']
                raw_data.extend(data_train)
                raw_data.extend(data_val)
            elif split == "unlearn" or split == "retention":
                data = datasets.load_dataset(data_path, name=data_type, split=split, cache_dir="../data")[0]

                data_train, data_val = data['train'], data['val']
                raw_data.extend(data_train)
                raw_data.extend(data_val)

            elif split == "full_both": # for my method
                unlearn_data = datasets.load_dataset(data_path, name='privacy', split='unlearn', cache_dir="../data")[0]
                retention_data = datasets.load_dataset(data_path, name='privacy', split='retention', cache_dir="../data")[0]
                
                data_train, data_val = unlearn_data['train'], unlearn_data['val']
                raw_data.extend(data_train)
                raw_data.extend(data_val)

                data_train, data_val = retention_data['train'], retention_data['val']
                raw_data.extend(data_train)
                raw_data.extend(data_val)

                unlearn_data = datasets.load_dataset(data_path, name='copyright', split='unlearn', cache_dir="../data")[0]
                retention_data = datasets.load_dataset(data_path, name='copyright', split='retention', cache_dir="../data")[0]

                data_train, data_val = unlearn_data['train'], unlearn_data['val']
                raw_data.extend(data_train)
                raw_data.extend(data_val)

                data_train, data_val = retention_data['train'], retention_data['val']
                raw_data.extend(data_train)
                raw_data.extend(data_val)
                
                random.seed(42)
                random.shuffle(raw_data)

            else:
                assert ValueError("split not supported")

            self.data = datasets.Dataset.from_dict({key: [dic[key] for dic in raw_data] for key in raw_data[0]})

        self.data = add_dataset_index(self.data)
        self.model_configs = get_model_identifiers_from_yaml(model_family)
        self.qk = question_key
        self.ak = answer_key

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        question = self.data[idx][self.qk]
        answers = self.data[idx][self.ak]
        indices = self.data[idx]['index']
        if isinstance(answers, str):
            answers = [answers]

        pad_input_ids_list = []
        label_list = []
        pad_attention_mask_list = []

        for answer in answers:
            converted_data = convert_raw_data_to_model_format(self.tokenizer, self.max_length, question, answer, self.model_configs)
            pad_input_ids_list.append(converted_data[0])
            label_list.append(converted_data[1])
            pad_attention_mask_list.append(converted_data[2])


        return torch.stack(pad_input_ids_list).squeeze(),\
                torch.stack(label_list).squeeze(),\
                torch.stack(pad_attention_mask_list).squeeze(),\
                torch.tensor(indices)


# # 이거 orthogonalize할때만 사용
# def custom_data_collator(samples):
#     input_ids = torch.stack([torch.tensor(s["input_ids"]) for s in samples])
#     labels = torch.stack([torch.tensor(s["labels"]) for s in samples])
#     attention_mask = torch.stack([torch.tensor(s["attention_mask"]) for s in samples])

#     tasks = [s["task"] for s in samples]


#     # 데이터를 GPU로 배치 단위로 전송
#     input_ids = input_ids.to("cuda", non_blocking=True)
#     labels = labels.to("cuda", non_blocking=True)
#     attention_mask = attention_mask.to("cuda", non_blocking=True)

#     return {
#         "input_ids": input_ids,
#         "labels": labels,
#         "attention_mask": attention_mask,
#         "task": tasks,
#     }

# def custom_data_collator(samples):
#     """
#     Collate function for data loader to handle batched inputs.
#     """
#     # Ensure all samples have "task" field
#     for s in samples:
#         if "task" not in s:
#             print(f"Missing 'task' in sample: {s}")  # Debugging
#             s["task"] = "unknown_task"

#     input_ids = torch.stack([torch.tensor(s["input_ids"]) for s in samples])
#     labels = torch.stack([torch.tensor(s["labels"]) for s in samples])
#     attention_mask = torch.stack([torch.tensor(s["attention_mask"]) for s in samples])

#     tasks = [s["task"] for s in samples]

#     return {
#         "input_ids": input_ids,
#         "labels": labels,
#         "attention_mask": attention_mask,
#         "task": tasks,
#     }


def custom_data_collator(samples):
    input_ids = [s[0] for s in samples]
    labels = [s[1] for s in samples]
    attention_mask = [s[2] for s in samples]
    return torch.stack(input_ids), torch.stack(labels), torch.stack(attention_mask)