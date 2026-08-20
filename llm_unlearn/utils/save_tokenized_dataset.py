from llm_unlearn.utils import tokenize, direct_prompts, our_dataset_path_dict
import torch
from transformers import set_seed, AutoTokenizer
from datasets import load_dataset, Dataset
import os

import argparse
model_max_length = 256
dir = os.environ.get("GRAIL_TOKENIZED_DIR", "../tokenized_dataset")

dataset_path_dict = {
    "general_1k": {"name":"general", "split":"evaluation"},
}

def save_tokenized_dataset(
    tokenizer_name_or_path,
    dataset_name,
    tokenize_method,
    completely_random=False,
    soft_label=False,
    top_k=int(1e10),
    top_p=1.0,
    rm_groundtruth=False,
    val=False,
    prompt=False,
):
    set_seed(42)
    
    tokenizer = AutoTokenizer.from_pretrained(
        tokenizer_name_or_path,
        padding_side="right",
        trust_remote_code=True,
        model_max_length=model_max_length,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model_name = os.path.basename(os.path.normpath(tokenizer_name_or_path))

    # Following the setting of unlearning_llm.
    if dataset_name in dataset_path_dict.keys():
        
        
        # dataset_path_dict = {
        #     "general_1k": {"name":"general", "split":"evaluation"},
        # }

        dataset_path = dataset_path_dict[dataset_name]
        raw_dataset = load_dataset("llmunlearn/unlearn_dataset", name=dataset_path["name"], split=dataset_path["split"], cache_dir="../../data")
        save_path = os.path.join(dir, model_name, dataset_path["name"], dataset_name, tokenize_method)


    elif dataset_name in our_dataset_path_dict.keys():
        
        # 헷갈려서 보기좋게 가져옴
        # our_dataset_path_dict = {
        #     "copyright_unlearn": {"name": "copyright", "split": "unlearn", },
        #     "copyright_retention": {"name": "copyright", "split": "retention", },
        #     "privacy_unlearn": {"name": "privacy", "split": "unlearn", },
        #     "privacy_retention": {"name": "privacy", "split": "retention", },
        #     "combined_unlearn": {"name": "combined", "split": "unlearn", },
        #     "combined_retention": {"name": "combined", "split": "retention", },
        # }

        if dataset_name == "combined_unlearn" or dataset_name == "combined_retention":
            # combined 데이터셋 처리
            privacy_path = our_dataset_path_dict["privacy_unlearn"] if "unlearn" in dataset_name else our_dataset_path_dict["privacy_retention"]
            copyright_path = our_dataset_path_dict["copyright_unlearn"] if "unlearn" in dataset_name else our_dataset_path_dict["copyright_retention"]

            # privacy 데이터 로드
            privacy_raw = load_dataset(
                "zjunlp/KnowUnDo", 
                name=privacy_path["name"], 
                split=privacy_path["split"], 
                cache_dir="../../data"
            )['train' if not val else 'val'][0]
            privacy_dataset = [{"text": '\n\n'.join([dic["text"], dic["labels"]]), "labels": '\n\n' + dic["labels"]} for dic in privacy_raw]

            # copyright 데이터 로드
            copyright_raw = load_dataset(
                "zjunlp/KnowUnDo", 
                name=copyright_path["name"], 
                split=copyright_path["split"], 
                cache_dir="../../data"
            )['train' if not val else 'val'][0]
            copyright_dataset = [{"text": '\n\n'.join([dic["text"], dic["labels"]]), "labels": '\n\n' + dic["labels"]} for dic in copyright_raw]

            # 결합 및 셔플
            raw_dataset = privacy_dataset + copyright_dataset
            raw_dataset = Dataset.from_dict({key: [dic[key] for dic in raw_dataset] for key in raw_dataset[0]})
            raw_dataset = raw_dataset.shuffle(seed=42)

            # combined 저장 경로 설정
            save_path = os.path.join(dir, model_name, "combined", dataset_name, tokenize_method)
        
        else:
            # 기존 처리 방식
            dataset_path = our_dataset_path_dict[dataset_name]
            raw_dataset = load_dataset("zjunlp/KnowUnDo", name=dataset_path["name"], split=dataset_path["split"], cache_dir="../../data")['train' if not val else 'val'][0]

            if not prompt:
                raw_dataset = [{"text": '\n\n'.join([dic["text"], dic["labels"]]), "labels": '\n\n' + dic["labels"]} for dic in raw_dataset]
            else:
                raw_dataset = [{"text": direct_prompts[dataset_path["name"]] + '\n\n'.join([dic["text"], dic["labels"]]), "labels": '\n\n' + dic["labels"]} for dic in raw_dataset]
            
            save_path = os.path.join(dir, model_name, dataset_path["name"], dataset_name, tokenize_method)
            raw_dataset = Dataset.from_dict({key: [dic[key] for dic in raw_dataset] for key in raw_dataset[0]})
            raw_dataset = raw_dataset.shuffle(seed=42)

        
    else:
        raise ValueError(f"dataset_name is wrong")

    # Tokenization 처리
    if tokenize_method == "normal":
        dataset = tokenize(raw_dataset, tokenizer, model_max_length)
    elif tokenize_method == "random_label":
        if completely_random:
            dataset = tokenize(
                raw_dataset,
                tokenizer,
                model_max_length,
                random_label=True,
                completely_random=True,
            )
            save_path = os.path.join(save_path, "completely_random")
        else:
            dataset = tokenize(
                raw_dataset,
                tokenizer,
                model_max_length,
                random_label=True,
                top_k=top_k,
                top_p=top_p,
                rm_groundtruth=rm_groundtruth,
            )
            save_path = os.path.join(save_path, f"top_k{top_k}_top_p{top_p}")
    else:
        raise ValueError(f"tokenize_method is wrong")

    if rm_groundtruth:
        save_path = save_path + "_rmgt"
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    if val:
        output_name = f"tokenized_dataset_val{'' if not prompt else '_prompt'}.pt"
    else:
        output_name = "tokenized_dataset.pt"
    save_path = os.path.join(save_path, output_name)
    torch.save(dataset, save_path)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--tokenizer_name_or_path", '-t',type=str, default=None, help="tokenizer_name_or_path.")
    parser.add_argument("--val", action="store_true", help="tokenize which partition.")
    parser.add_argument("--prompt", action="store_true", help="whether add prompt before tokenizing.")
    args = parser.parse_args()

    # Per-domain forget (unlearn) / retain (retention) sets used by the GRAIL
    # evaluation (llm_evaluation/grail_run_eval_lora.py). Run with --val to
    # produce the held-out validation partition it scores.
    dataset_name_list = [
        "copyright_unlearn",
        "copyright_retention",
        "privacy_unlearn",
        "privacy_retention",
        "combined_unlearn",
        "combined_retention",
    ]
    for dataset_name in dataset_name_list:
        save_tokenized_dataset(
            args.tokenizer_name_or_path, dataset_name, "normal",
            val=args.val, prompt=args.prompt,
        )