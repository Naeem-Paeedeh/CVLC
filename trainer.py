import logging
import copy
import torch
from utils.toolkit import count_parameters
import numpy as np
import our_utils as ou
from collections import defaultdict
from torch import nn
from configs import Configuration
from models.our_method import CVLC
from torch import Tensor as T

torch.set_float32_matmul_precision("high")

def train_with_different_seeds(cfg: Configuration):
    seed_list = copy.deepcopy(cfg.seeds_list)
    
    # Average accuracies for each random seed
    mean_accuracies_without_oracle_list = []
    mean_accuracies_with_oracle_list = []
    
    last_accuracies_without_oracle_list = []
    last_accuracies_with_oracle_list = []
    
    AA_star_without_oracle_list = []
    FA_star_without_oracle_list = []
    AA_star_with_oracle_list = []
    FA_star_with_oracle_list = []
    domain_classification_accuracy_list = []

    for i, seed in enumerate(seed_list):
        cfg.seed_current = seed
        mean_accuracies_without_oracle, mean_accuracies_with_oracle, acc_without_oracle, acc_with_oracle, AA_star_without_oracle, FA_star_without_oracle, AA_star_with_oracle, FA_star_with_oracle, mean_domain_classification = train_with_the_chosen_random_seed(cfg=cfg)
        
        mean_accuracies_without_oracle_list.append(mean_accuracies_without_oracle)
        mean_accuracies_with_oracle_list.append(mean_accuracies_with_oracle)
        
        last_accuracies_without_oracle_list.append(acc_without_oracle)
        last_accuracies_with_oracle_list.append(acc_with_oracle)
        
        AA_star_without_oracle_list.append(AA_star_without_oracle)
        FA_star_without_oracle_list.append(FA_star_without_oracle)
        AA_star_with_oracle_list.append(AA_star_with_oracle)
        FA_star_with_oracle_list.append(FA_star_with_oracle)
        domain_classification_accuracy_list.append(mean_domain_classification)
        
        if i > 0:
            logging.info(f"Results after {i + 1} random seeds: ")
            
            mean_for_all_random_seeds, std_for_all_random_seeds = ou.calculate_mean_and_std_for_a_list(AA_star_without_oracle_list)
            logging.info(f"AA* mean (w/o Oracle): {mean_for_all_random_seeds:.2f} ± {std_for_all_random_seeds:.2f}")
            mean_for_all_random_seeds, std_for_all_random_seeds = ou.calculate_mean_and_std_for_a_list(FA_star_without_oracle_list)
            logging.info(f"FA* mean (w/o Oracle): {mean_for_all_random_seeds:.2f} ± {std_for_all_random_seeds:.2f}")
            
            mean_for_all_random_seeds, std_for_all_random_seeds = ou.calculate_mean_and_std_for_a_list(AA_star_with_oracle_list)
            logging.info(f"AA* mean (with Oracle): {mean_for_all_random_seeds:.2f} ± {std_for_all_random_seeds:.2f}")
            mean_for_all_random_seeds, std_for_all_random_seeds = ou.calculate_mean_and_std_for_a_list(FA_star_with_oracle_list)
            logging.info(f"FA* mean (with Oracle): {mean_for_all_random_seeds:.2f} ± {std_for_all_random_seeds:.2f}")
            
            mean_for_all_random_seeds, std_for_all_random_seeds = ou.calculate_mean_and_std_for_a_list(domain_classification_accuracy_list)
            logging.info(f"Mean (Domain Classification): {mean_for_all_random_seeds:.2f} ± {std_for_all_random_seeds:.2f}")
            
            mean_for_all_random_seeds, std_for_all_random_seeds = ou.calculate_mean_and_std_for_a_list(mean_accuracies_without_oracle_list)
            logging.info(f"Acc., mean (w/o Oracle): {mean_for_all_random_seeds:.2f} ± {std_for_all_random_seeds:.2f}")
            mean_for_all_random_seeds, std_for_all_random_seeds = ou.calculate_mean_and_std_for_a_list(mean_accuracies_with_oracle_list)
            logging.info(f"Acc., mean (with Oracle): {mean_for_all_random_seeds:.2f} ± {std_for_all_random_seeds:.2f}")
            
            mean_for_all_random_seeds, std_for_all_random_seeds = ou.calculate_mean_and_std_for_a_list(last_accuracies_without_oracle_list)
            logging.info(f"Acc., last (w/o Oracle): {mean_for_all_random_seeds:.2f} ± {std_for_all_random_seeds:.2f}")
            mean_for_all_random_seeds, std_for_all_random_seeds = ou.calculate_mean_and_std_for_a_list(last_accuracies_with_oracle_list)
            logging.info(f"Acc., last (with Oracle): {mean_for_all_random_seeds:.2f} ± {std_for_all_random_seeds:.2f}")


def train_with_the_chosen_random_seed(cfg: Configuration):
    ou.set_seed(cfg.seed_current)
    
    logging.info("-" * 30)
    
    logging.info(f"Random seed: {cfg.seed_current}")
    
    def show_results(cfg: Configuration, accuracies_list: list, description: str):
        temp, mean, std = ou.get_printable_string_from_a_list_of_float_numbers_with_two_digits(accuracies_list)
        
        mean_and_std_str = f"Mean: {mean:.2f} ± {std:.2f}"
        
        logging.info(f'{description}, domain {cfg.current_domain_id + 1}/{cfg.total_sessions}, order {cfg.order}, seed: {cfg.seed_current}: {temp} -> {mean_and_std_str}')
        
        return mean, std

    slf = CVLC(cfg=cfg)

    domain_classification_accuracy_list = []
    accuracies_without_oracle_list = []
    accuracies_with_oracle_list = []
    
    assert cfg.total_sessions > 0
    
    accuracies_matrix_without_oracle = torch.zeros(cfg.total_sessions, cfg.total_sessions)
    accuracies_matrix_with_oracle = torch.zeros(cfg.total_sessions, cfg.total_sessions)
    
    acc_without_oracle = 0.0
    acc_with_oracle = 0.0
    
    for domain_id in range(cfg.total_sessions):
        cfg.current_domain_id = domain_id
        
        slf.prepare_for_current_domain()
        
        if cfg.num_shots == 0:
            raise NotImplementedError()
        elif cfg.num_shots > 0:
            slf.train()
            
        slf.after_task()
        
        domain_classification_accuracy = 0.0
        
        if cfg.num_shots == 0:
            raise NotImplementedError()
        else:
            acc_without_oracle, acc_with_oracle, acc_per_domain_without_oracle, acc_per_domain_with_oracle, domain_classification_accuracy = slf.evaluate_on_test_set()
            
            accuracies_matrix_without_oracle[domain_id, :domain_id + 1] = acc_per_domain_without_oracle
            accuracies_matrix_with_oracle[domain_id, :domain_id + 1] = acc_per_domain_with_oracle
        
        logging.info(f"Domain {cfg.current_domain_id + 1}/{cfg.total_sessions}, order: {cfg.order}, seed: {cfg.seed_current}, Acc. (w/o oracle): {acc_without_oracle:.2f}, AA (with oracle): {acc_with_oracle:.2f}")
        
        acc_per_domain, _, _ = ou.get_printable_string_from_a_list_of_float_numbers_with_two_digits(acc_per_domain_without_oracle.tolist())
        
        AA_without_oracle = acc_per_domain_without_oracle.mean().item()
        
        logging.info(f"Domain {cfg.current_domain_id + 1}/{cfg.total_sessions}, order: {cfg.order}, seed: {cfg.seed_current}, Acc. per domain (w/o oracle): {acc_per_domain}, AA (w/o oracle): {AA_without_oracle:.2f}")
        
        acc_per_domain, _, _ = ou.get_printable_string_from_a_list_of_float_numbers_with_two_digits(acc_per_domain_with_oracle.tolist())
        
        AA_with_oracle = acc_per_domain_with_oracle.mean().item()
        
        logging.info(f"Domain {cfg.current_domain_id + 1}/{cfg.total_sessions}, order: {cfg.order}, seed: {cfg.seed_current}, Acc. per domain (with oracle): {acc_per_domain}, AA (with oracle): {AA_with_oracle:.2f}")
        
        overall_average_accuracy_without_oracle, overall_forgetting_alleviation_without_oracle = calculate_average_accuracy_and_forgetting_alleviation(
            accuracies_matrix=accuracies_matrix_without_oracle,
            current_domain_id=cfg.current_domain_id
        )
        
        logging.info(f"Domain {cfg.current_domain_id + 1}/{cfg.total_sessions}, order: {cfg.order}, seed: {cfg.seed_current}, AA* (w/o oracle): {overall_average_accuracy_without_oracle:.2f}, FA*: {overall_forgetting_alleviation_without_oracle:.2f}")
        
        overall_average_accuracy_with_oracle, overall_forgetting_alleviation_with_oracle = calculate_average_accuracy_and_forgetting_alleviation(
            accuracies_matrix=accuracies_matrix_with_oracle,
            current_domain_id=cfg.current_domain_id
        )
        
        logging.info(f"Domain {cfg.current_domain_id + 1}/{cfg.total_sessions}, order: {cfg.order}, seed: {cfg.seed_current}, AA* (with oracle): {overall_average_accuracy_with_oracle:.2f}, FA*: {overall_forgetting_alleviation_with_oracle:.2f}")
        
        accuracies_without_oracle_list.append(acc_without_oracle)
        accuracies_with_oracle_list.append(acc_with_oracle)
        domain_classification_accuracy_list.append(domain_classification_accuracy)
        
        mean_accuracies_without_oracle, _ = show_results(
            cfg=cfg,
            accuracies_list=accuracies_without_oracle_list, 
            description="Accuracies (w/o oracle)"
        )
        
        mean_accuracies_with_oracle, _ = show_results(
            cfg=cfg,
            accuracies_list=accuracies_with_oracle_list, 
            description="Accuracies (with oracle)"
        )
        
        mean_domain_classification, _ = show_results(
            cfg=cfg,
            accuracies_list=domain_classification_accuracy_list, 
            description="Domain classification accuracies (after each domain)"
        )
    
    logging.info('This experiment is finished!')
    
    return mean_accuracies_without_oracle, mean_accuracies_with_oracle, acc_without_oracle, acc_with_oracle, overall_average_accuracy_without_oracle, overall_forgetting_alleviation_without_oracle, overall_average_accuracy_with_oracle, overall_forgetting_alleviation_with_oracle, mean_domain_classification


def calculate_average_accuracy_and_forgetting_alleviation(
    accuracies_matrix: T,
    current_domain_id: int
):
    # Notes: 
    # Each row belong to to one session. Each element in the row is the accuracy on that domain after the end of training on that task.
    # AA_1, AA_2, ..., AA_t
    assert accuracies_matrix.dim() == 2
    
    encountered_domains = current_domain_id + 1
    
    accuracies_matrix_visible = accuracies_matrix[:encountered_domains, :encountered_domains]
    
    average_accuracy = accuracies_matrix_visible.sum(dim=1) / torch.arange(1, encountered_domains + 1)
    overall_average_accuracy = average_accuracy.mean().item()      # AA_star
    
    if current_domain_id > 0:
        accuracies_matrix_visible_for_forgetting = accuracies_matrix_visible.clone()
        
        accuracies_matrix_visible_for_forgetting.fill_diagonal_(0)
        
        forgetting_alleviation = accuracies_matrix_visible_for_forgetting.sum(0)[:-1] / torch.arange(current_domain_id, 0, -1)
        
        overall_forgetting_alleviation = forgetting_alleviation.mean().item()       # FA* in the PGO-BEn paper.
    else:
        overall_forgetting_alleviation = 0.0
    
    return overall_average_accuracy, overall_forgetting_alleviation