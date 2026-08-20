#!/usr/bin/env python
# coding: utf-8

import torch
import numpy as np
import os
import argparse


# ------------------------------------------
# 1) Z-score 기반 상위 k% 인덱스 뽑는 함수
# ------------------------------------------
def get_top_indices_zscore(grad_dict, top_percent=None, use_abs=True):
    """
    grad_dict: {layer_name: torch.Tensor}
    top_percent: 상위 몇 % (기본 0.1 = 10%)
    use_abs: True면 Z-score 절댓값 기준
    return: { layer_name: set_of_indices }
    """
    layer_top_indices = {}
    for lname, grad_tensor in grad_dict.items():
        grad_flat = grad_tensor.view(-1)
        mean_val = grad_flat.mean().item()
        std_val = grad_flat.std().item()
        
        if std_val < 1e-12:
            layer_top_indices[lname] = set()
            continue
        
        z_scores = (grad_flat - mean_val) / std_val
        if use_abs:
            z_scores = z_scores.abs()
        
        z_np = z_scores.detach().cpu().numpy()
        threshold = np.percentile(z_np, 100 * (1 - top_percent))
        
        top_idx = np.where(z_np >= threshold)[0]
        layer_top_indices[lname] = set(top_idx)
    return layer_top_indices


def get_top_indices(grad_dict, top_percent=None):
    """
    grad_dict: {layer_name: torch.Tensor}
    top_percent: 상위 몇 % (기본 0.1 = 10%)
    return: { layer_name: set_of_indices }
    """
    layer_top_indices = {}
    for lname, grad_tensor in grad_dict.items():
        grad_flat = grad_tensor.view(-1)  # Flatten the gradient tensor
        grad_np = grad_flat.detach().cpu().numpy()  # Convert to NumPy array
        
        # 상위 top_percent 기준 임계값 계산
        threshold = np.percentile(grad_np, 100 * (1 - top_percent))
        
        # 임계값 이상인 인덱스 찾기
        top_idx = np.where(grad_np >= threshold)[0]
        layer_top_indices[lname] = set(top_idx)
    
    return layer_top_indices

# ------------------------------------------
# 2) Retention 두 개를 합집합으로 만드는 함수
# ------------------------------------------
def get_union(ret1_dict, ret2_dict):
    """
    ret1_dict, ret2_dict: {layer_name: set_of_indices}
    -> 2개 retention 인덱스의 합집합을 레이어별로 구해 반환
    """
    union_dict = {}
    all_layers = set(ret1_dict.keys()) | set(ret2_dict.keys())
    for lname in all_layers:
        set1 = ret1_dict.get(lname, set())
        set2 = ret2_dict.get(lname, set())
        union_dict[lname] = set1 | set2  # 합집합
    return union_dict

def get_intersection_of_two_retention(ret1_dict, ret2_dict):
    """
    ret1_dict, ret2_dict: {layer_name: set_of_indices}
    -> 2개 retention 인덱스의 교집합을 레이어별로 구해 반환
    """
    intersection_dict = {}
    all_layers = set(ret1_dict.keys()) | set(ret2_dict.keys())
    for lname in all_layers:
        set1 = ret1_dict.get(lname, set())
        set2 = ret2_dict.get(lname, set())
        intersection_dict[lname] = set1 & set2  # 교집합
    return intersection_dict
# ------------------------------------------
# 3) 인덱스를 .pt로 저장하는 함수
# ------------------------------------------
def save_indices_to_pt(index_dict, filename):
    """
    index_dict: {layer_name: set_of_indices} 형태
    filename: 저장할 경로 (예: ".../common_pu_ret_union.pt")
    """
    index_tensor_dict = {}
    for lname, idx_set in index_dict.items():
        # set -> list 변환 후 정렬
        sorted_list = sorted(list(idx_set))

        idx_tensor = torch.tensor(sorted_list, dtype=torch.long)
        index_tensor_dict[lname] = idx_tensor

    torch.save(index_tensor_dict, filename)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--UR", type=float, default=0.2, help="OP-UR percent, e.g. 0.1 0.2 ...")
    parser.add_argument("--RR", type=float, default=0.2, help="OP-RR percent, e.g. 0.1 0.2 ...")
    parser.add_argument("--localized_type", type=str, default="zscore", choices=["fim", "zscore"], help="importance score: fim or zscore")
    parser.add_argument("--vanilla_model", type=str, default="Llama-2-7b-chat-hf", help="Llama-2-7b-chat-hf or Qwen1.5-7B-Chat")
    parser.add_argument("--base_model", type=str, default="final_combined_ft_LORA_20_epochs_inst_lr0.0003_llama2-7b_full_both", help="fine-tuned LoRA module directory name")
    parser.add_argument("--output_dir", type=str,
                        default=os.environ.get("GRAIL_LOCALIZATION_DIR", "llm_localization/outputs"),
                        help="directory holding grad_info_*.pt; UR/RR masks are written under it (env: GRAIL_LOCALIZATION_DIR)")

    args = parser.parse_args()

    save_path = os.path.join(args.output_dir,
                             args.vanilla_model,
                             args.localized_type,
                             args.base_model,
                             )
    
    grad_privacy_unlearn = torch.load(os.path.join(save_path, "grad_info_privacy_unlearn_3.pt"))
    grad_privacy_retention = torch.load(os.path.join(save_path, "grad_info_privacy_retention_3.pt"))
    grad_copyright_unlearn = torch.load(os.path.join(save_path, "grad_info_copyright_unlearn_3.pt"))
    grad_copyright_retention = torch.load(os.path.join(save_path, "grad_info_copyright_retention_3.pt"))

    UR_percent = args.UR
    RR_percent = args.RR
    

    if args.UR is not None and not os.path.isfile(os.path.join(save_path, "UR", f"UR_top{int(UR_percent*100)}.pt")):
        print(f"### Obtaining {int(args.UR * 100)} percent of UR...")

        if args.localized_type == "fim":
            top_pu = get_top_indices(grad_privacy_unlearn, UR_percent)
            top_cu = get_top_indices(grad_copyright_unlearn, UR_percent)

            top_pr = get_top_indices(grad_privacy_retention, UR_percent)
            top_cr = get_top_indices(grad_copyright_retention, UR_percent)

        elif args.localized_type == "zscore":
            top_pu = get_top_indices_zscore(grad_privacy_unlearn, UR_percent)
            top_cu = get_top_indices_zscore(grad_copyright_unlearn, UR_percent)

            top_pr = get_top_indices_zscore(grad_privacy_retention, UR_percent)
            top_cr = get_top_indices_zscore(grad_copyright_retention, UR_percent)

        union_unlearn = get_union(top_pu, top_cu)
        union_retention = get_union(top_pr, top_cr)

        UR_ = {}
        for lname in union_unlearn.keys():
            set_unl = union_unlearn[lname]
            set_ret = union_retention.get(lname, set())
            UR_[lname] = set_unl & set_ret

        # UR_top1.pt
        # RR_top10.pt
        save_indices_to_pt(UR_, os.path.join(save_path, "UR", f"UR_top{int(UR_percent*100)}.pt"))

    if args.RR is not None and not os.path.isfile(os.path.join(save_path, "RR", f"RR_top{int(RR_percent*100)}.pt")):
        print(f"### Obtaining {int(args.RR * 100)} percent of RR...")
        
        if args.localized_type == "fim":
            top_pr = get_top_indices(grad_privacy_retention, RR_percent)
            top_cr = get_top_indices(grad_copyright_retention, RR_percent)
        elif args.localized_type == "zscore":
            top_pr = get_top_indices_zscore(grad_privacy_retention, RR_percent)
            top_cr = get_top_indices_zscore(grad_copyright_retention, RR_percent)

        RR_ = get_intersection_of_two_retention(top_pr, top_cr)

        save_indices_to_pt(RR_, os.path.join(save_path, "RR", f"RR_top{int(UR_percent*100)}.pt"))


if __name__ == "__main__":
    main()
