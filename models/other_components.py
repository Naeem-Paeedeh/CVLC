import os
import torch
from torch import nn
from torch.nn import functional as F
from copy import deepcopy
import logging
import new_types as nt
import our_utils as ou
from typing import Optional, Tuple, Any
import math
import einops as eo
from torch import Tensor as T
from models.coalescent_projection import initialize_a_coalescent_projection_tensor

import configs as cg


class Statistics(nn.Module):
    def __init__(
        self,
        embed_dim: int,
        device,
        accumulate_shared_covariances_from_all_domains: bool = True
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.accumulate_shared_covariances_from_all_domains = accumulate_shared_covariances_from_all_domains
        self.device = device
        
        self.task_count = 0
        
        self.means = torch.tensor([], device=device)
        self.labels = torch.tensor([], dtype=torch.long, device=device)
        self.domain_ids = torch.tensor([], dtype=torch.long, device=device)
        
        # We compute both shared and separate covariances for each class. However, we require only one of them at the end.
        self.accumulated_shared_covariances = torch.tensor([], device=device)             # Its shape would be [embed_dim, embed_dim]
        self.shared_covariances_inverse = torch.tensor([], device=device)     # Its shape would be [embed_dim, embed_dim]
        
        self.separated_covariances = torch.tensor([], device=device)  # Its shape: [num_encountered_tasks, embed_dim, embed_dim]
        self.separate_covariances_inverse = torch.tensor([], device=device)  # Its shape: [num_encountered_tasks, embed_dim, 
        
    def update(
        self,
        means: T,
        covariances: T,
        labels: T,
        domain_id: int       # For verification purpose only!
    ):
        """This method updates the whole statistics by considering the statistics of the current task.

        Args:
            means (T): _description_
            labels (T): _description_
            task_ids (T): _description_
            covariance (T, optional): _description_. Defaults to None.
        """
        assert len(means) == len(labels)
        
        covariances = covariances.detach().to(self.device)
        means = means.detach().to(self.device)
        labels = labels.detach().to(self.device)
        
        self.task_count += 1
        assert domain_id == self.task_count - 1
        
        self.means = torch.cat([self.means, means], dim=0)
        self.labels = torch.cat([self.labels, labels], dim=0)
        self.domain_ids = torch.cat([self.domain_ids, domain_id * torch.ones_like(labels)], dim=0)
        
        if covariances is not None:
            # In this case, we accumulate the covariances.
            if self.accumulated_shared_covariances.dim() == 1:      # For the first time, we don't have a matrix for summation.
                self.accumulated_shared_covariances = torch.cat([self.separated_covariances, covariances], dim=0)
            elif self.accumulate_shared_covariances_from_all_domains:
                self.accumulated_shared_covariances = self.accumulated_shared_covariances + covariances
            
            if self.accumulate_shared_covariances_from_all_domains or domain_id == 0:
                cov_inv_temp = torch.linalg.pinv(self.accumulated_shared_covariances / self.task_count, hermitian=True)
                self.shared_covariances_inverse = cov_inv_temp.to(self.device)
            
            self.separated_covariances = torch.cat([self.separated_covariances, covariances], dim=0)
            cov_inv_temp = torch.linalg.pinv(covariances, hermitian=True)
            self.separate_covariances_inverse = torch.cat([self.separate_covariances_inverse, cov_inv_temp], dim=0)
            
    def obtain_statistics(
        self,
        domain_id: int,
        separated_or_shared_covariances: bool
    ):
        
        if domain_id == -1:
            domain_id = self.task_count - 1
        
        mask = self.domain_ids == domain_id
        
        means = self.means[mask]
        
        covariances = torch.tensor([])
        
        if separated_or_shared_covariances:
            covariances = self.separated_covariances[mask]
        else:
            if self.accumulate_shared_covariances_from_all_domains:
                covariances = self.accumulated_shared_covariances / self.task_count
            else:
                covariances = self.accumulated_shared_covariances
            
        labels = self.labels[mask]
        
        return means, covariances, labels
    
    
class CacheManagement:
    def __init__(
        self,
        dataset_name: str,
        order: int,
        num_epochs: int,
        num_shots: int,
        dir_cache: str
    ) -> None:
        self.dataset_name = dataset_name
        self.order = order
        self.num_epochs = num_epochs
        self.num_shots = num_shots
        self.dir_cache = dir_cache
        
    def try_cache_first(
        self,
        title: str,
        num_epochs: int,
        domain_id: int,
        parameter_efficient_method: nt.PEFT_Type
    ):
        cache_file_path = self._obtain_cache_file_path(
            title=title,
            num_epochs=num_epochs,
            domain_id=domain_id,
            parameter_efficient_method=parameter_efficient_method
        )
        
        if os.path.exists(cache_file_path):
            data_dict = torch.load(cache_file_path, weights_only=False)
            return data_dict
        
        return None
    
    def save_in_a_cache_file(
        self,
        title: str,
        data_dict: dict,
        num_epochs: int,
        domain_id: int,
        parameter_efficient_method: nt.PEFT_Type
    ):
        cache_file_path = self._obtain_cache_file_path(
            title=title,
            num_epochs=num_epochs,
            domain_id=domain_id,
            parameter_efficient_method=parameter_efficient_method
        )
        torch.save(data_dict, cache_file_path)
        
        logging.info(f"{title} are saved in \"{cache_file_path}\"!")
    
    def _obtain_cache_file_path(
        self,
        title: str,
        num_epochs: int,
        domain_id: int,
        parameter_efficient_method: nt.PEFT_Type
    ):
        
        cache_file_name = f'{title},dataset={self.dataset_name},order={self.order},domain_id={domain_id},num_epochs={num_epochs},parameter_efficient_method={parameter_efficient_method}'
        
        cache_file_name += ".pth"
        
        cache_file_path = os.path.join(self.dir_cache, cache_file_name)
        
        return cache_file_path

            
class LearnabeCoefficients(nn.Module):
    def __init__(
        self,   # The arguments are just optional starting points.
        num_domains: int,
        coef_synonyms_prototypes: float = 0.5,
        coef_visual_prototypes_calibration: float = 0.5,
        coef_inter_modal_calibration: float = 0.5,
        coef_shift_text_from_current_domain_in_vision_modality: float = 0.5,
        coef_shift_text_from_first_domain_in_vision_modality: float = 0.5,
        coef_shift_text: float = 0.5,
        coef_shift_vision_from_first_domain: float = 0.5,
        coef_shift_vision_from_current_domain: float = 0.5,
        learnable: bool = True
    ) -> None:
        super().__init__()
        
        self.num_domains = num_domains
        self.learnable = learnable
        
        # We consider separate coefficients for every domain.
        self.coef_synonyms_prototypes = nn.Parameter(coef_synonyms_prototypes * torch.ones(num_domains), requires_grad=learnable)
        self.coef_visual_prototypes_calibration = nn.Parameter(coef_visual_prototypes_calibration * torch.ones(num_domains), requires_grad=learnable)
        self.coef_inter_modal_calibration = nn.Parameter(coef_inter_modal_calibration * torch.ones(num_domains), requires_grad=learnable)
        # We calibrate the shifts by considering both the shift of the current domain and the shift of the first domain.
        # This is for considering the shift of the current domain
        self.coef_shift_text_from_current_domain_in_vision_modality = nn.Parameter(coef_shift_text_from_current_domain_in_vision_modality * torch.ones(num_domains), requires_grad=learnable)
        # This is for the incremental tasks by considering the shift of the first domain.
        self.coef_shift_text_from_first_domain_in_vision_modality = nn.Parameter(coef_shift_text_from_first_domain_in_vision_modality * torch.ones(num_domains), requires_grad=learnable)
        self.coef_shift_text = nn.Parameter(coef_shift_text * torch.ones(num_domains), requires_grad=learnable)
        # This is for the impact of the base domain's shift
        self.coef_shift_vision_from_first_domain = nn.Parameter(coef_shift_vision_from_first_domain * torch.ones(num_domains), requires_grad=learnable)
        # This is for the impact of the current domain's shift
        self.coef_shift_vision_from_current_domain = nn.Parameter(coef_shift_vision_from_current_domain * torch.ones(num_domains), requires_grad=learnable)
        
        
class CoalescentProjections(nn.Module):
    def __init__(
        self,
        enable_vision_CPs: bool,
        enable_text_CPs: bool,
        num_heads_vision: int,
        num_heads_text: int,
        dim_head_vision: int,
        dim_head_text: int,
        task_shared_layers: list[int],
        task_specific_layers: list[int],
        peft_for_new_domain: nt.InitializationApproachForIncrementalTasks,
        shared_CPS_shared_across_heads: bool,
        specific_CPS_shared_across_heads: bool,
        device=None,
        dtype=torch.float32,
        std: float = 0.02,
        initialization: nt.InitializationType = nt.InitializationType.CloseToDiagonalMatrix,
    ):
        super().__init__()
        
        self.enable_vision_CPs = enable_vision_CPs
        self.enable_text_CPs = enable_text_CPs
        
        self.dim_head_vision = dim_head_vision
        self.dim_head_text = dim_head_text
        self.num_heads_vision = num_heads_vision
        self.num_heads_text = num_heads_text
        self.task_shared_layers = task_shared_layers
        self.task_specific_layers = task_specific_layers
        
        self.device = device
        self.dtype = dtype
        self.shared_CPS_shared_across_heads = shared_CPS_shared_across_heads
        self.specific_CPS_shared_across_heads = specific_CPS_shared_across_heads
        self.std = std
        self.peft_for_new_domain = peft_for_new_domain
        
        self.initialization = initialization
        
        self.CPs_shared_vision_dict = nn.ParameterDict()
        self.CPs_shared_text_dict = nn.ParameterDict()
        self.CPs_specific_vision_dict = nn.ParameterDict()
        self.CPs_specific_text_dict = nn.ParameterDict()
        
        self._initialization()
        
    def _initialization(self):
        arguments_dict: dict = dict(
            std=self.std,
            initialization=self.initialization,
            dtype=self.dtype,
            device=self.device
        )
        
        for layer_number in self.task_shared_layers:
            if self.enable_vision_CPs:
                # Between Query and Key matrices
                self.CPs_shared_vision_dict[f'QK,layer_number={layer_number}'] = initialize_a_coalescent_projection_tensor(
                    num_heads=self.num_heads_vision,
                    dim_head=self.dim_head_vision,
                    shared_across_heads=self.shared_CPS_shared_across_heads,
                    **arguments_dict
                )
                # Between the Softmax(Attn) and Value
                self.CPs_shared_vision_dict[f'SV,layer_number={layer_number}'] = initialize_a_coalescent_projection_tensor(
                    num_heads=self.num_heads_vision,
                    dim_head=self.dim_head_vision,
                    shared_across_heads=self.shared_CPS_shared_across_heads,
                    **arguments_dict
                )
            
            if self.enable_text_CPs:
                self.CPs_shared_text_dict[f'QK,layer_number={layer_number}'] = initialize_a_coalescent_projection_tensor(
                    num_heads=self.num_heads_text,
                    dim_head=self.dim_head_text,
                    shared_across_heads=self.shared_CPS_shared_across_heads,
                    **arguments_dict
                )
                self.CPs_shared_text_dict[f'SV,layer_number={layer_number}'] = initialize_a_coalescent_projection_tensor(
                    num_heads=self.num_heads_text,
                    dim_head=self.dim_head_text,
                    shared_across_heads=self.shared_CPS_shared_across_heads,
                    **arguments_dict
                    )
            
    def prepare_for_a_new_task(
        self,
        domain_id: int
    ):
        arguments_dict: dict = dict(
            std=self.std,
            initialization=self.initialization,
            dtype=self.dtype,
            device=self.device
        )
        
        previous_domain = domain_id - 1
        
        for layer_number in self.task_specific_layers:
            if domain_id == 0 or self.peft_for_new_domain == nt.InitializationApproachForIncrementalTasks.Reinitialize:
                if self.enable_vision_CPs:
                    self.CPs_specific_vision_dict[f'QK,domain={domain_id},layer_number={layer_number}'] = initialize_a_coalescent_projection_tensor(
                        num_heads=self.num_heads_vision,
                        dim_head=self.dim_head_vision,
                        shared_across_heads=self.specific_CPS_shared_across_heads,
                        **arguments_dict
                    )
                    self.CPs_specific_vision_dict[f'SV,domain={domain_id},layer_number={layer_number}'] = initialize_a_coalescent_projection_tensor(
                        num_heads=self.num_heads_vision,
                        dim_head=self.dim_head_vision,
                        shared_across_heads=self.specific_CPS_shared_across_heads,
                        **arguments_dict
                    )
                if self.enable_text_CPs:
                    self.CPs_specific_text_dict[f'QK,domain={domain_id},layer_number={layer_number}'] = initialize_a_coalescent_projection_tensor(
                        num_heads=self.num_heads_text,
                        dim_head=self.dim_head_text,
                        shared_across_heads=self.specific_CPS_shared_across_heads,
                        **arguments_dict
                    )
                    self.CPs_specific_text_dict[f'SV,domain={domain_id},layer_number={layer_number}'] = initialize_a_coalescent_projection_tensor(
                        num_heads=self.num_heads_text,
                        dim_head=self.dim_head_text,
                        shared_across_heads=self.specific_CPS_shared_across_heads,
                        **arguments_dict
                    )
            elif self.peft_for_new_domain in [nt.InitializationApproachForIncrementalTasks.CopyFromPreviousDomain, nt.InitializationApproachForIncrementalTasks.CopyFromFirstDomain]:       # In the incremental tasks, we copy the domain-specific CPs from previous domain and freeze the previous domain-specific CPs
                # Vision
                if self.peft_for_new_domain == nt.InitializationApproachForIncrementalTasks.CopyFromFirstDomain:
                    domain_to_copy_from = 0
                elif self.peft_for_new_domain == nt.InitializationApproachForIncrementalTasks.CopyFromPreviousDomain:
                    domain_to_copy_from = domain_id - 1
                    
                self.copy_CPs_from_a_domain(layer_number=layer_number, domain_to_copy_from=domain_to_copy_from, domain_id=domain_id)
                self.freeze_CPs_from_a_domain(domain_id=previous_domain, layer_number=layer_number)
                
            elif self.peft_for_new_domain == nt.InitializationApproachForIncrementalTasks.MeanOfEncounteredDomains:
                raise NotImplementedError()
            else:
                raise NotImplementedError()
            
    def copy_CPs_from_a_domain(
        self,
        layer_number: int,
        domain_to_copy_from: int,
        domain_id: int
    ):
        if self.enable_vision_CPs:
            self.CPs_specific_vision_dict[f'QK,domain={domain_id},layer_number={layer_number}'] = deepcopy(self.CPs_specific_vision_dict[f'QK,domain={domain_to_copy_from},layer_number={layer_number}']).requires_grad_(True)
            
            self.CPs_specific_vision_dict[f'SV,domain={domain_id},layer_number={layer_number}'] = deepcopy(self.CPs_specific_vision_dict[f'SV,domain={domain_to_copy_from},layer_number={layer_number}']).requires_grad_(True)
            
        # Text
        if self.enable_text_CPs:
            self.CPs_specific_text_dict[f'QK,domain={domain_id},layer_number={layer_number}'] = deepcopy(self.CPs_specific_text_dict[f'QK,domain={domain_to_copy_from},layer_number={layer_number}']).requires_grad_(True)
            
            self.CPs_specific_text_dict[f'SV,domain={domain_id},layer_number={layer_number}'] = deepcopy(self.CPs_specific_text_dict[f'SV,domain={domain_to_copy_from},layer_number={layer_number}']).requires_grad_(True)
    
    def freeze_CPs_from_a_domain(
        self,
        domain_id: int,
        layer_number: int
    ):
        if self.enable_vision_CPs:
            self.CPs_specific_vision_dict[f'QK,domain={domain_id},layer_number={layer_number}'].requires_grad_(False)
            self.CPs_specific_vision_dict[f'QK,domain={domain_id},layer_number={layer_number}'].grad = None
            
            self.CPs_specific_vision_dict[f'SV,domain={domain_id},layer_number={layer_number}'].requires_grad_(False)
            self.CPs_specific_vision_dict[f'SV,domain={domain_id},layer_number={layer_number}'].grad = None
        
        if self.enable_text_CPs:
            self.CPs_specific_text_dict[f'QK,domain={domain_id},layer_number={layer_number}'].requires_grad_(False)
            self.CPs_specific_text_dict[f'QK,domain={domain_id},layer_number={layer_number}'].grad = None
            
            self.CPs_specific_text_dict[f'SV,domain={domain_id},layer_number={layer_number}'].requires_grad_(False)
            self.CPs_specific_text_dict[f'SV,domain={domain_id},layer_number={layer_number}'].grad = None
            
    def obtain_CPs_of_a_domain(
        self,
        modality: str,
        domain_id: int
    ) -> dict[int, nn.Parameter]:
        
        assert modality in ["vision", "text"]
        
        coalescent_projections_current_domain_dict = {}
        
        for layer_number in self.task_shared_layers:
            if modality == "vision" and self.enable_vision_CPs:
                coalescent_projections_current_domain_dict[f"QK,{layer_number}"] = self.CPs_shared_vision_dict[f'QK,layer_number={layer_number}']
                coalescent_projections_current_domain_dict[f"SV,{layer_number}"] = self.CPs_shared_vision_dict[f'SV,layer_number={layer_number}']
            
            if modality == "text" and self.enable_text_CPs:
                coalescent_projections_current_domain_dict[f"QK,{layer_number}"] = self.CPs_shared_text_dict[f'QK,layer_number={layer_number}']
                coalescent_projections_current_domain_dict[f"SV,{layer_number}"] = self.CPs_shared_text_dict[f'SV,layer_number={layer_number}']
        
        for layer_number in self.task_specific_layers:
            if modality == "vision" and self.enable_vision_CPs:
                coalescent_projections_current_domain_dict[f"QK,{layer_number}"] = self.CPs_specific_vision_dict[f'QK,domain={domain_id},layer_number={layer_number}']
                coalescent_projections_current_domain_dict[f"SV,{layer_number}"] = self.CPs_specific_vision_dict[f'SV,domain={domain_id},layer_number={layer_number}']
            
            if modality == "text" and self.enable_text_CPs:
                coalescent_projections_current_domain_dict[f"QK,{layer_number}"] = self.CPs_specific_text_dict[f'QK,domain={domain_id},layer_number={layer_number}']
                coalescent_projections_current_domain_dict[f"SV,{layer_number}"] = self.CPs_specific_text_dict[f'SV,domain={domain_id},layer_number={layer_number}']
            
        return coalescent_projections_current_domain_dict


class DeepPrompts(nn.Module):
    def __init__(
        self,
        num_vision_prompts: int,
        num_text_prompts: int,
        enable_vision_prompts: bool,
        enable_text_prompts: bool,
        dim_embed_vision: int,
        dim_embed_text: int,
        task_shared_layers: list[int],
        task_specific_layers: list[int],
        peft_for_new_domain: nt.InitializationApproachForIncrementalTasks,
        device=None,
    ) -> None:
        super().__init__()
        
        self.num_vision_prompts = num_vision_prompts
        self.num_text_prompts = num_text_prompts
        
        self.enable_vision_prompts = enable_vision_prompts
        self.enable_text_prompts = enable_text_prompts
        self.dim_embed_vision = dim_embed_vision
        self.dim_embed_text = dim_embed_text
        self.task_shared_layers = task_shared_layers
        self.task_specific_layers = task_specific_layers
        self.peft_for_new_domain = peft_for_new_domain
        self.device = device
        
        self.scale_vision = dim_embed_vision ** -0.5
        self.scale_text = dim_embed_text ** -0.5
        
        self.prompts_shared_vision_dict = nn.ParameterDict()
        self.prompts_shared_text_dict = nn.ParameterDict()
        self.prompts_specific_vision_dict = nn.ParameterDict()
        self.prompts_specific_text_dict = nn.ParameterDict()
        
        self._initialization()
        
    def _initialization(self):
        for layer_number in self.task_shared_layers:
            if self.enable_vision_prompts:
                self.prompts_shared_vision_dict[f'layer_number={layer_number}'] = Prompt(
                    num_prompts=self.num_vision_prompts,
                    dim_embed=self.dim_embed_vision,
                    device=self.device
                )
            
            if self.enable_text_prompts:
                self.prompts_shared_text_dict[f'layer_number={layer_number}'] = Prompt(
                    num_prompts=self.num_text_prompts,
                    dim_embed=self.dim_embed_text,
                    device=self.device
                )
                
    def prepare_for_a_new_task(
        self,
        domain_id: int
    ):
        previous_domain = domain_id - 1
        
        for layer_number in self.task_specific_layers:
            if domain_id == 0 or self.peft_for_new_domain == nt.InitializationApproachForIncrementalTasks.Reinitialize:
                if self.enable_vision_prompts:
                    self.prompts_specific_vision_dict[f'domain={domain_id},layer_number={layer_number}'] = Prompt(
                        num_prompts=self.num_vision_prompts,
                        dim_embed=self.dim_embed_vision,
                        device=self.device
                    )
                    
                if self.enable_text_prompts:
                    self.prompts_specific_text_dict[f'domain={domain_id},layer_number={layer_number}'] = Prompt(
                        num_prompts=self.num_text_prompts,
                        dim_embed=self.dim_embed_text,
                        device=self.device
                    )
            
            elif self.peft_for_new_domain in [nt.InitializationApproachForIncrementalTasks.CopyFromPreviousDomain, nt.InitializationApproachForIncrementalTasks.CopyFromFirstDomain]:       # In the incremental tasks, we copy the domain-specific CPs from previous domain and freeze the previous domain-specific CPs
                # Vision
                if self.peft_for_new_domain == nt.InitializationApproachForIncrementalTasks.CopyFromFirstDomain:
                    domain_to_copy_from = 0
                elif self.peft_for_new_domain == nt.InitializationApproachForIncrementalTasks.CopyFromPreviousDomain:
                    domain_to_copy_from = domain_id - 1
                    
                self.copy_prompts_from_a_domain(layer_number=layer_number, domain_to_copy_from=domain_to_copy_from, domain_id=domain_id)
                self.freeze_prompts_from_a_domain(domain_id=previous_domain, layer_number=layer_number)
                
            elif self.peft_for_new_domain == nt.InitializationApproachForIncrementalTasks.MeanOfEncounteredDomains:
                raise NotImplementedError()
            else:
                raise NotImplementedError()
    
    def copy_prompts_from_a_domain(
        self,
        layer_number: int,
        domain_to_copy_from: int,
        domain_id: int
    ):
        if self.enable_vision_prompts:
            self.prompts_specific_vision_dict[f'domain={domain_id},layer_number={layer_number}'] = deepcopy(self.prompts_specific_vision_dict[f'domain={domain_to_copy_from},layer_number={layer_number}']).requires_grad_(True)
            
        # Text
        if self.enable_text_prompts:
            self.prompts_specific_text_dict[f'domain={domain_id},layer_number={layer_number}'] = deepcopy(self.prompts_specific_text_dict[f'domain={domain_to_copy_from},layer_number={layer_number}']).requires_grad_(True)
            
    def freeze_prompts_from_a_domain(
        self,
        domain_id: int,
        layer_number: int
    ):
        
        key = f'domain={domain_id},layer_number={layer_number}'
        
        if self.enable_vision_prompts:
            self.prompts_specific_vision_dict[key].requires_grad_(False)
            self.prompts_specific_vision_dict[key].grad = None
        
        if self.enable_text_prompts:
            self.prompts_specific_text_dict[key].requires_grad_(False)
            self.prompts_specific_text_dict[key].grad = None
            
    def obtain_prompts_of_a_domain(
        self,
        modality: str,
        domain_id: int
    ) -> dict[int, nn.Parameter]:
        
        assert modality in ["vision", "text"]
        
        prompts_current_domain_dict = {}
        
        for layer_number in self.task_shared_layers:
            if modality == "vision" and self.enable_vision_prompts:
                prompts_current_domain_dict[layer_number] = self.prompts_shared_vision_dict[f'layer_number={layer_number}']
            
            if modality == "text" and self.enable_text_prompts:
                prompts_current_domain_dict[layer_number] = self.prompts_shared_text_dict[f'layer_number={layer_number}']
        
        for layer_number in self.task_specific_layers:
            if modality == "vision" and self.enable_vision_prompts:
                prompts_current_domain_dict[layer_number] = self.prompts_specific_vision_dict[f'domain={domain_id},layer_number={layer_number}']
            
            if modality == "text" and self.enable_text_prompts:
                prompts_current_domain_dict[layer_number] = self.prompts_specific_text_dict[f'domain={domain_id},layer_number={layer_number}']
            
        return prompts_current_domain_dict
                

class Prompt(nn.Module):
    def __init__(
        self,
        num_prompts: int,
        dim_embed: int,
        device
    ) -> None:
        super().__init__()
        
        self.scale = dim_embed ** -0.5
        self.num_prompts = num_prompts
        self.dim_embed = dim_embed
        self.device = device
        
        self.prompt = nn.Parameter(self.scale * torch.randn(num_prompts, dim_embed, device=device))
        

class LoRAsForCLIP(nn.Module):
    def __init__(
        self,
        enable_vision_LoRAs: bool,
        enable_text_LoRAs: bool,
        dim_embed_vision: int,
        dim_embed_text: int,
        downsize_dimension_vision: int,
        downsize_dimension_text: int,
        LoRA_QKV_mask: list[bool],
        task_shared_layers: list[int],
        task_specific_layers: list[int],
        peft_for_new_domain: nt.InitializationApproachForIncrementalTasks,
        device=None,
    ) -> None:
        super().__init__()
        
        self.enable_vision_LoRAs = enable_vision_LoRAs
        self.enable_text_LoRAs = enable_text_LoRAs
        
        self.dim_embed_vision = dim_embed_vision
        self.dim_embed_text = dim_embed_text
        
        self.downsize_dimension_vision = downsize_dimension_vision
        self.downsize_dimension_text = downsize_dimension_text
        
        self.LoRA_QKV_mask = LoRA_QKV_mask
        self.task_shared_layers = task_shared_layers
        self.task_specific_layers = task_specific_layers
        self.peft_for_new_domain = peft_for_new_domain
        self.device = device
        
        self.LoRAs_shared_vision_dict = nn.ParameterDict()
        self.LoRAs_shared_text_dict = nn.ParameterDict()
        self.LoRAs_specific_vision_dict = nn.ParameterDict()
        self.LoRAs_specific_text_dict = nn.ParameterDict()
        
        self._initialization()
        
    def _initialization(self):
        arguments_dict: dict = dict(
            lora_qkv_mask=self.LoRA_QKV_mask,
            device=self.device
        )
        
        for layer_number in self.task_shared_layers:
            if self.enable_vision_LoRAs:
                self.LoRAs_shared_vision_dict[f'layer_number={layer_number}'] = LoRA_QKV(
                    dim_embed=self.dim_embed_vision,
                    downsize_dimension=self.downsize_dimension_vision,
                    **arguments_dict
                )
            
            if self.enable_text_LoRAs:
                self.LoRAs_shared_text_dict[f'layer_number={layer_number}'] = LoRA_QKV(
                    dim_embed=self.dim_embed_text,
                    downsize_dimension=self.downsize_dimension_text,
                    **arguments_dict
                )
                
    def prepare_for_a_new_task(
        self,
        domain_id: int
    ):
        arguments_dict: dict = dict(
            lora_qkv_mask=self.LoRA_QKV_mask,
            device=self.device
        )
        
        previous_domain = domain_id - 1
        
        for layer_number in self.task_specific_layers:
            if domain_id == 0 or self.peft_for_new_domain == nt.InitializationApproachForIncrementalTasks.Reinitialize:
                if self.enable_vision_LoRAs:
                    self.LoRAs_specific_vision_dict[f'domain={domain_id},layer_number={layer_number}'] = LoRA_QKV(
                        dim_embed=self.dim_embed_vision,
                        downsize_dimension=self.downsize_dimension_vision,
                        **arguments_dict
                    )
                    
                if self.enable_text_LoRAs:
                    self.LoRAs_specific_text_dict[f'domain={domain_id},layer_number={layer_number}'] = LoRA_QKV(
                        dim_embed=self.dim_embed_text,
                        downsize_dimension=self.downsize_dimension_text,
                        **arguments_dict
                    )
            
            elif self.peft_for_new_domain in [nt.InitializationApproachForIncrementalTasks.CopyFromPreviousDomain, nt.InitializationApproachForIncrementalTasks.CopyFromFirstDomain]:       # In the incremental tasks, we copy the domain-specific CPs from previous domain and freeze the previous domain-specific CPs
                # Vision
                if self.peft_for_new_domain == nt.InitializationApproachForIncrementalTasks.CopyFromFirstDomain:
                    domain_to_copy_from = 0
                elif self.peft_for_new_domain == nt.InitializationApproachForIncrementalTasks.CopyFromPreviousDomain:
                    domain_to_copy_from = domain_id - 1
                    
                self.copy_LoRAs_from_a_domain(layer_number=layer_number, domain_to_copy_from=domain_to_copy_from, domain_id=domain_id)
                self.freeze_LoRAs_from_a_domain(domain_id=previous_domain, layer_number=layer_number)
                
            elif self.peft_for_new_domain == nt.InitializationApproachForIncrementalTasks.MeanOfEncounteredDomains:
                raise NotImplementedError()
            else:
                raise NotImplementedError()
            
    def copy_LoRAs_from_a_domain(
        self,
        layer_number: int,
        domain_to_copy_from: int,
        domain_id: int
    ):
        if self.enable_vision_LoRAs:
            self.LoRAs_specific_vision_dict[f'domain={domain_id},layer_number={layer_number}'] = deepcopy(self.LoRAs_specific_vision_dict[f'domain={domain_to_copy_from},layer_number={layer_number}']).requires_grad_(True)
            
        # Text
        if self.enable_text_LoRAs:
            self.LoRAs_specific_text_dict[f'domain={domain_id},layer_number={layer_number}'] = deepcopy(self.LoRAs_specific_text_dict[f'domain={domain_to_copy_from},layer_number={layer_number}']).requires_grad_(True)
            
    def freeze_LoRAs_from_a_domain(
        self,
        domain_id: int,
        layer_number: int
    ):
        
        key = f'domain={domain_id},layer_number={layer_number}'
        
        if self.enable_vision_LoRAs:
            self.LoRAs_specific_vision_dict[key].requires_grad_(False)
            self.LoRAs_specific_vision_dict[key].grad = None
        
        if self.enable_text_LoRAs:
            self.LoRAs_specific_text_dict[key].requires_grad_(False)
            self.LoRAs_specific_text_dict[key].grad = None
            
    def obtain_LoRAs_of_a_domain(
        self,
        modality: str,
        domain_id: int
    ) -> dict[int, nn.Parameter]:
        
        assert modality in ["vision", "text"]
        
        LoRAs_current_domain_dict = {}
        
        for layer_number in self.task_shared_layers:
            if modality == "vision" and self.enable_vision_LoRAs:
                LoRAs_current_domain_dict[layer_number] = self.LoRAs_shared_vision_dict[f'layer_number={layer_number}']
            
            if modality == "text" and self.enable_text_LoRAs:
                LoRAs_current_domain_dict[layer_number] = self.LoRAs_shared_text_dict[f'layer_number={layer_number}']
        
        for layer_number in self.task_specific_layers:
            if modality == "vision" and self.enable_vision_LoRAs:
                LoRAs_current_domain_dict[layer_number] = self.LoRAs_specific_vision_dict[f'domain={domain_id},layer_number={layer_number}']
            
            if modality == "text" and self.enable_text_LoRAs:
                LoRAs_current_domain_dict[layer_number] = self.LoRAs_specific_text_dict[f'domain={domain_id},layer_number={layer_number}']
            
        return LoRAs_current_domain_dict


class LoRA_QKV(nn.Module):
    def __init__(
        self,
        dim_embed: int,
        downsize_dimension: int,
        lora_qkv_mask: list[bool],
        device
    ):
        super().__init__()
        self.dim_embed = dim_embed
        self.down_size = downsize_dimension
        self.lora_qkv_mask = lora_qkv_mask
        self.device = device
        
        assert len(lora_qkv_mask) == 3
        
        self.adapters_list = nn.ModuleList()
        
        for mask in lora_qkv_mask:
            if mask:
                self.adapters_list.append(LoRA(dim_embed=self.dim_embed, downsize_dimension=self.down_size, device=device))
            else:
                self.adapters_list.append(nn.Identity())

    
class LoRA(nn.Module):
    def __init__(
        self,
        dim_embed: int,
        downsize_dimension: int,
        device
    ):
        super().__init__()
        
        self.dim_embed = dim_embed
        self.down_size = downsize_dimension
        self.device = device

        self.down_proj = nn.Linear(self.dim_embed, self.down_size, bias=False, device=device)      # B
        self.up_proj = nn.Linear(self.down_size, self.dim_embed, bias=False, device=device)        # A

        with torch.no_grad():
            nn.init.kaiming_uniform_(self.down_proj.weight, a=math.sqrt(5))
            nn.init.zeros_(self.up_proj.weight)

    def forward(self, x: T):
        inter_x = self.down_proj.forward(x)
        out = self.up_proj.forward(inter_x)
        return out


class Classifiers(nn.Module):
    def __init__(
        self,
        total_sessions: int,
        dim_input: int,
        dim_output: int,
        classifier_type: nt.ClassifierType,
        peft_for_new_domain: nt.InitializationApproachForIncrementalTasks,
        device,
        cosine_similarity = True,       # If it is false, we use Euclidean distance for the final prototypical classifier.
        init_temperature_stochastic_classifier: float = 16.0,
        init_temperature_cosine_classifier: float = 0.07,
        gamma_weight=10.0               # For prototype correction
    ):
        super().__init__()
        
        self.total_sessions = total_sessions
        self.dim_input = dim_input
        self.dim_output = dim_output
        self.num_classes = dim_output
        self.classifier_type = classifier_type
        self.cosine_similarity = cosine_similarity
        self.peft_for_new_domain = peft_for_new_domain
        self.init_temperature_stochastic_classifier = init_temperature_stochastic_classifier
        self.init_temperature_cosine_classifier = init_temperature_cosine_classifier
        self.gamma_weight = gamma_weight
        self.device = device
        
        self.temporary_classifiers = nn.ModuleList()
        
        for i in range(total_sessions):
            self.temporary_classifiers.append(
                CosineClassifier(
                    dim_input=dim_input,
                    num_classes=dim_output
                ).to(self.device)
            )
        
        # Prototypical classifer
        # These prototypes are calculated by using both modalities.
        self.prototypes_final = torch.tensor([], device=device, requires_grad=False)
        self.prototypes_final_labels = torch.tensor([], dtype=torch.long, requires_grad=False, device=device)
        self.prototypes_final_domain_ids = torch.tensor([], dtype=torch.long, requires_grad=False, device=device)
        # self.classifiers_list = nn.ModuleList()
        
        self.domain_id: int = -1
        
    @torch.no_grad()
    def correct_prototypes(
        self,
        prototypes_new_task_before: T,       # (num_classes_new_domain, dim_embed)
        current_domain_id: int
    ):
        """
        Correct class prototype means and covariances.

        Args:
            prototypes_old_domains: Old task means, shape (num_classes_old_domains, dim_embed).
            prototypes_new_task_before: New means under old model, shape (num_classes_new_domain, dim_embed).
        Returns:
            Corrected old prototypes
        """
        # Note: We must have added the prototypes from the current domain to the final prototypes.
        assert current_domain_id > 0
        
        prototypes_all = self.prototypes_final
        # prototypes_labels = self.prototypes_final_labels
        prototypes_domain_ids = self.prototypes_final_domain_ids
        
        mask_old_prototypes = prototypes_domain_ids < current_domain_id
        mask_prototypes_new_task_after = prototypes_domain_ids == current_domain_id
        
        prototypes_old_domains = prototypes_all[mask_old_prototypes]        # (num_classes_old_domains, dim_embed)
        prototypes_new_task_after = prototypes_all[mask_prototypes_new_task_after]   # (num_classes_new_domain, dim_embed)

        # assert prototypes_old_domains.shape[0] > 0
        assert prototypes_new_task_before.shape[0] > 0
        assert prototypes_new_task_after.shape[0] > 0
        
        # Mean correction
        delta_mu = prototypes_new_task_after - prototypes_new_task_before    # (num_classes_new_domain, dim_embed)

        mu_old_normalized = F.normalize(prototypes_old_domains, p=2, dim=1)  # (num_classes_old_domains, dim_embed)
        mu_new_before_normalized = F.normalize(prototypes_new_task_before, p=2, dim=1)       # (num_classes_new_domain, dim_embed)

        # Cosine similarity matrix - Its shape: (num_classes_old_domains, num_classes_new_domain)
        cosine_similarities = mu_old_normalized @ mu_new_before_normalized.T

        # Weights: (num_classes_old_domains, num_classes_new_domain)
        w = F.softmax(self.gamma_weight * cosine_similarities, dim=-1)

        # Weighted sum of shifts: (num_classes_old_domains, dim_embed)
        corrected_mu_old = prototypes_old_domains + (w @ delta_mu)
        
        self.prototypes_final[mask_old_prototypes] = corrected_mu_old
        
    def prepare_for_a_new_task(
        self,
        domain_id: int
    ):
        
        assert self.classifier_type == nt.ClassifierType.Cosine
        
        with torch.no_grad():
            if self.peft_for_new_domain == nt.InitializationApproachForIncrementalTasks.Reinitialize:
                pass        # We initialized them at the beginning.
            elif self.peft_for_new_domain == nt.InitializationApproachForIncrementalTasks.CopyFromPreviousDomain:
                if domain_id > 0:
                    self.temporary_classifiers[domain_id] = deepcopy(self.temporary_classifiers[domain_id - 1])
                    self.temporary_classifiers[domain_id].zero_grad()
            elif self.peft_for_new_domain == nt.InitializationApproachForIncrementalTasks.CopyFromFirstDomain:
                if domain_id > 0:
                    self.temporary_classifiers[domain_id] = deepcopy(self.temporary_classifiers[0])
                    self.temporary_classifiers[domain_id].zero_grad()
            elif self.peft_for_new_domain == nt.InitializationApproachForIncrementalTasks.MeanOfEncounteredDomains:
                if domain_id > 0:
                    previous_weights = torch.tensor([], device=self.device)
                    previous_logit_scales = torch.tensor([], device=self.device)
                
                    for dmn_id in range(domain_id):
                        previous_weights = torch.cat([previous_weights, deepcopy(self.temporary_classifiers[dmn_id].weight.data.detach().unsqueeze(0))], dim=0)
                        
                        previous_logit_scales = torch.cat([previous_logit_scales, deepcopy(self.temporary_classifiers[dmn_id].logit_scale.data.detach().unsqueeze(0))], dim=0)
                        
                    mean_weight = previous_weights.mean(0)
                    mean_logit_scales = previous_logit_scales.mean()
                    
                    with torch.no_grad():
                        self.temporary_classifiers[domain_id].weight.data = mean_weight.data
                        self.temporary_classifiers[domain_id].logit_scale.data = mean_logit_scales
            
                
        if domain_id > 0:
            self.temporary_classifiers[domain_id - 1].requires_grad_(False)
            
        self.domain_id += 1
        assert domain_id == self.domain_id
        
    def add_or_update_final_prototypes(
        self,
        prototypes: T,
        labels: T,
        domain_ids: T | int
    ):
        
        assert prototypes.dim() == 2
        assert len(prototypes) == len(labels)
        
        flag_add_or_update = True
        
        if self.cosine_similarity:
            prototypes = F.normalize(prototypes, dim=-1)
        
        if isinstance(domain_ids, T):
            assert len(prototypes) == len(domain_ids)
            domain_ids_tensor = domain_ids
        elif isinstance(domain_ids, int):
            if domain_ids in self.prototypes_final_domain_ids:
                flag_add_or_update = False
                mask_update = torch.nonzero(self.prototypes_final_domain_ids == domain_ids).squeeze().tolist()
                
            domain_ids_tensor = domain_ids * torch.ones(len(prototypes), dtype=torch.long, device=self.device)
        
        if flag_add_or_update:
            self.prototypes_final = torch.cat([self.prototypes_final, prototypes], dim=0)        # Its shape=[num_new_classes, dim_embed]
            self.prototypes_final_labels = torch.cat([self.prototypes_final_labels, labels], dim=0)
            self.prototypes_final_domain_ids = torch.cat([self.prototypes_final_domain_ids, domain_ids_tensor], dim=0)
        else:
            self.prototypes_final[mask_update] = prototypes
            self.prototypes_final_labels[mask_update] = labels
            self.prototypes_final_domain_ids[mask_update] = domain_ids_tensor
        
    def obtain_final_prototypes_by_a_domain_id(
        self,
        domain_id: int,
    ):
        mask = self.prototypes_final_domain_ids == domain_id
        prototypes_filtered = self.prototypes_final[mask]
        prototypes_filtered_labels = self.prototypes_final_labels[mask]
        return prototypes_filtered, prototypes_filtered_labels
    
    def obtain_all_final_prototypes(
        self,
        consider_temporary_classifier_weights: bool = False
    ):
        prototypes = self.prototypes_final
        labels = self.prototypes_final_labels
        
        if consider_temporary_classifier_weights:
            prototypes = torch.cat([prototypes, self.temporary_classifiers[domain_id].weight.detach()], dim=0)
            
            labels = torch.cat([labels, torch.arange(domain_id * self.dim_output, (domain_id + 1) * self.dim_output, device=labels.device)], dim=0)
        
        return prototypes, labels
    
    def forward_final_prototypical_classifier(
        self,
        x: T        # Its shape: [num_samples, dim_embed]
    ):
        if self.cosine_distance:
            x = F.normalize(x, dim=-1)
        
        logits = x @ self.prototypes_final.T        # prototypes_final is already normalized!
        
        return logits
    
    def forward_and_bimodal_calibration_with_a_temporary_classifier(
        self,
        embeddings_vision: T,
        prototypes_text: T,
        coef_inter_modal_calibration: float | nn.Parameter | T,
        coef_visual_prototypes_calibration: float | nn.Parameter | T,
        calibrate_vision_prototypes: bool,
        domain_id: int,
        normalize: bool = True,
    ):
        prototypes_current_domain = self.temporary_classifiers[domain_id].weight
        
        if calibrate_vision_prototypes and domain_id > 0:
            prototypes_base_domain, ptototypes_base_domain_labels = self.obtain_final_prototypes_by_a_domain_id(domain_id=0)
            
            prototypes_base_domain = prototypes_base_domain.detach().requires_grad_(False)
            
            prototypes_base_domain.grad = None
            
            prototypes_vision_calibrated = ou.interpolate(
                coef=coef_visual_prototypes_calibration,
                a=prototypes_current_domain,
                b=prototypes_base_domain,
                normalize=normalize
            )
        else:
            prototypes_vision_calibrated = prototypes_current_domain
                
        if normalize:
            embeddings_vision = F.normalize(embeddings_vision, p=2, dim=-1)
            prototypes_vision_calibrated = F.normalize(prototypes_vision_calibrated, p=2, dim=-1)
        
        prototypes_calibrated = ou.interpolate(
            coef=coef_inter_modal_calibration,
            a=prototypes_vision_calibrated,
            b=prototypes_text
        )
        
        logits_fused = F.linear(embeddings_vision, prototypes_calibrated)
        
        logits_fused = self.temporary_classifiers[domain_id].forward_logits(logits_fused)

        return logits_fused
    
    def forward_and_bimodal_calibration_with_final_prototypes(
        self,
        embeddings_vision: T,
        prototypes_text: T,
        coef_inter_modal_calibration: float | nn.Parameter | T,
        coef_visual_prototypes_calibration: float | nn.Parameter | T,
        calibrate_vision_prototypes: bool,
        domain_id: int,     # If we set it to -1, we will use the prototypes of all domains
        normalize: bool = True
    ):
        prototypes, prototype_labels = self.obtain_final_prototypes_by_a_domain_id(domain_id)      # Prototypes for the domain domain_id
        
        if calibrate_vision_prototypes and domain_id > 0:
            prototypes_base_domain, prototypes_base_domain_labels = self.obtain_final_prototypes_by_a_domain_id(domain_id=0)
            
            prototypes_base_domain = prototypes_base_domain.detach().requires_grad_(False)
            prototypes_base_domain.grad = None
            
            prototypes_vision_calibrated = ou.interpolate(
                coef=coef_visual_prototypes_calibration,
                a=prototypes,
                b=prototypes_base_domain,
                normalize=normalize
            )
        else:
            prototypes_vision_calibrated = prototypes.detach().requires_grad_(False)
                
        if normalize:
            embeddings_vision = F.normalize(embeddings_vision, p=2, dim=-1)
            prototypes_vision_calibrated = F.normalize(prototypes_vision_calibrated, p=2, dim=-1)
            
        prototypes_calibrated = ou.interpolate(
            coef=coef_inter_modal_calibration,
            a=prototypes_vision_calibrated,
            b=prototypes_text
        )
        
        logits_fused = F.linear(embeddings_vision, prototypes_calibrated)
        
        logits_fused = self.temporary_classifiers[domain_id].forward_logits(logits_fused)       # For the temperature parameters

        return logits_fused
    
    def forward_and_calibrate_embeddings_with_both_modalities_for_UMAP(       # For UMAP or t-SNE plots
        self,
        embeddings_vision: T,
        labels_predicted: T,
        prototypes_text: T,
        coef_inter_modal_calibration: float | nn.Parameter | T,
        coef_visual_prototypes_calibration: float | nn.Parameter | T,
        calibrate_vision_prototypes: bool,
        domain_id: int,     # If we set it to -1, we will use the prototypes of all domains
        normalize: bool = True
    ):
        # Instead of prototpyes, we calibrate the embeddings
        # prototypes, prototype_labels = self.obtain_final_prototypes_by_a_domain_id(domain_id)      # Prototypes for the domain domain_id
        
        if calibrate_vision_prototypes and domain_id > 0:
            prototypes_base_domain, prototypes_base_domain_labels = self.obtain_final_prototypes_by_a_domain_id(domain_id=0)
            
            prototypes_base_domain = prototypes_base_domain.detach().requires_grad_(False)
            prototypes_base_domain.grad = None
            
            corresponding_prototypes_base_domain = prototypes_base_domain[labels_predicted % self.num_classes]
            
            embeddings_vision_calibrated = ou.interpolate(
                coef=coef_visual_prototypes_calibration,
                a=embeddings_vision,
                b=corresponding_prototypes_base_domain,
                normalize=normalize
            )
        else:
            embeddings_vision_calibrated = embeddings_vision.detach().requires_grad_(False)
                
        if normalize:
            embeddings_vision = F.normalize(embeddings_vision, p=2, dim=-1)
            embeddings_vision_calibrated = F.normalize(embeddings_vision_calibrated, p=2, dim=-1)
            
        correspondin_text_prototypes = prototypes_text[labels_predicted % self.num_classes]
            
        embeddings_calibrated = ou.interpolate(
            coef=coef_inter_modal_calibration,
            a=embeddings_vision_calibrated,
            b=correspondin_text_prototypes
        )
        
        return embeddings_calibrated

class CosineClassifier(nn.Module):
    """
    Cosine classifier with learnable temperature.

    Args:
        in_features: Dimension of input features.
        num_classes: Number of output classes.
        init_temperature: Initial temperature value.
        learnable_temperature: Whether temperature should be learnable.
        max_logit_scale: Optional maximum value for inverse temperature.
        eps: Numerical stability value for normalization.
    """

    def __init__(
        self,
        dim_input: int,
        num_classes: int,
        init_temperature: float = 0.07,
        learnable_temperature: bool = True,
        max_logit_scale: Optional[float] = 100.0,
    ):
        super().__init__()

        self.dim_input = dim_input
        self.num_classes = num_classes
        self.max_logit_scale = max_logit_scale

        self.weight = nn.Parameter(torch.empty(num_classes, dim_input))
        nn.init.normal_(self.weight, mean=0.0, std=0.01)

        # Inverse temperature, also called logit scale:
        init_logit_scale = math.log(1.0 / init_temperature)  # Or init_logit_scale = -1 * math.log(init_temperature)

        if learnable_temperature:
            self.logit_scale = nn.Parameter(torch.tensor(init_logit_scale))
        else:
            self.register_buffer("logit_scale", torch.tensor(init_logit_scale))

    def forward(
        self,
        embeddings: T
    ) -> T:
        """
        Args:
            x: Input tensor of shape [batch_size, in_features]

        Returns:
            Logits of shape [batch_size, num_classes]
        """
        
        embeddings = F.normalize(embeddings, p=2, dim=-1)
        weight = F.normalize(self.weight, p=2, dim=-1)

        cosine_logits = F.linear(embeddings, weight)

        logits = self.forward_logits(cosine_logits)

        return logits
        
    def forward_logits(
        self,
        logits: T
    ):
        logit_scale = self.logit_scale.exp()

        if self.max_logit_scale is not None:
            logit_scale = torch.clamp(logit_scale, max=self.max_logit_scale)

        logits = logits * logit_scale
        
        return logits
    
    @property
    def temperature(self) -> T:
        """
        Returns the current positive temperature.
        """
        return 1.0 / self.logit_scale.exp()


# Reference:
# https://github.com/hshustc/CVPR19_Incremental_Learning/blob/master/cifar100-class-incremental/modified_linear.py
# https://github.com/lambor9973/cds