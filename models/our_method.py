import os
import logging
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from tqdm import tqdm
from torch import optim
from torch.utils.data import DataLoader, TensorDataset
from utils.toolkit import tensor2numpy, accuracy_domain_shot
import new_types as nt
import our_utils as ou
from torch import Tensor as T
from typing import Any, Union, List, Generator, Tuple, Dict
from collections import defaultdict
import json
import configs as cg
import einops as eo
from utils.data_manager import DataManager

from models.clip import clip
from models.clip.model import CLIP
from models.other_components import LearnabeCoefficients, CoalescentProjections, LoRAsForCLIP, DeepPrompts, Classifiers, Statistics, CacheManagement
from models.LSR import LSR
from torch.optim.lr_scheduler import ReduceLROnPlateau


class CVLC:
    def __init__(
        self,
        cfg:cg.Configuration
    ):
        super().__init__()
        
        assert cfg.backbone_type in [nt.BackboneType.CLIP_ViT_B16, nt.BackboneType.CLIP_ViT_L14]
        
        self.cfg = cfg
        
        self._known_classes = 0
        self._total_classes = 0
        self.device = cfg.device
        
        cfg.current_domain_id = 0
        
        self.num_classes: int = -1
        self.class_names: list[str] = None
        
        if cfg.dataset_name == 'core50':
            self.num_classes = 50
        elif cfg.dataset_name == 'cddb':
            self.num_classes = 2
        elif cfg.dataset_name == 'domainnet':
            pass
        else:
            raise NotImplementedError()
        
        ou.set_seed(cfg.seed_current)
        
        self.data_manager: DataManager = None
        self.domain_names_for_this_order: list[str] = []
        self._prepare_data_manager()
        
        self.cfg = cfg
        self.device = cfg.device

        self.num_shots: int = cfg.num_shots

        self.topk = 1

        self.test_loader = None
        
        self.model: CLIP = None
        self.preprocess = None
            
        self.dim_embed = -1             # dim_embed for the output
        self.dim_embed_vision = -1      # dim_embed in the ViT before the final projection
        self.dim_embed_text = -1        # dim_embed in the text encoder Transformer
        self.num_heads_vision = -1
        self.num_heads_text = -1

        self.total_sessions = cfg.total_sessions
        
        self.prototypes_text: T = None
        
        self.label_to_domain_id: T = torch.tensor([], dtype=torch.long, device=self.device)
        
        self.synonyms_dict: dict[str, list] = {}
        self.num_synonyms: int = -1
        self.import_and_verify_descriptions()
        
        cfg.current_domain_id = -1
        
        self.train_loader: DataLoader = None
        self.test_loader: DataLoader = None
        
        self.load_the_backbone()
        
        # Learnable parameters
        self.calibration_coefficients = LearnabeCoefficients(
            num_domains=cfg.total_sessions,
            coef_synonyms_prototypes=cfg.coef_synonyms_prototypes_init_value,
            coef_visual_prototypes_calibration=cfg.coef_visual_prototypes_calibration_init_value,
            coef_inter_modal_calibration=cfg.coef_inter_modal_calibration_init_value,
            coef_shift_text_from_current_domain_in_vision_modality=cfg.coef_shift_text_from_current_domain_in_vision_modality_init_value,
            coef_shift_text_from_first_domain_in_vision_modality=cfg.coef_shift_text_from_first_domain_in_vision_modality_init_value,
            coef_shift_text=cfg.coef_shift_text_init_value,
            coef_shift_vision_from_first_domain=cfg.coef_shift_vision_from_first_domain_init_value,
            coef_shift_vision_from_current_domain=cfg.coef_shift_vision_from_current_domain_init_value,
            learnable=cfg.lr_calibration_coefficients[cfg.current_domain_id] > 0.0
        ).to(self.device)
        
        self.power_norm_vision_var = nn.Parameter(cfg.power_norm_alpha_vision_init_value * torch.ones(cfg.total_sessions, device=self.device), requires_grad=True)
        self.power_norm_text_var = nn.Parameter(cfg.power_norm_alpha_vision_init_value * torch.ones(cfg.total_sessions, device=self.device), requires_grad=True)
        
        self.coalescent_projections: CoalescentProjections = None
        self.LoRAs_for_CLIP: LoRAsForCLIP = None
        self.prompts: DeepPrompts = None
        
        if cfg.parameter_efficient_method == nt.PEFT_Type.CoalescentProjection:
            self.coalescent_projections = CoalescentProjections(
                enable_vision_CPs=cfg.enable_vision_CPs,
                enable_text_CPs=cfg.enable_text_CPs,
                dim_head_vision=self.dim_embed_vision // self.num_heads_vision,
                dim_head_text=self.dim_embed_text // self.num_heads_text,
                num_heads_vision=self.num_heads_vision,
                num_heads_text=self.num_heads_text,
                task_shared_layers=cfg.task_shared_layers,
                task_specific_layers=cfg.task_specific_layers,
                shared_CPS_shared_across_heads=cfg.shared_CPs_shared_across_heads,
                specific_CPS_shared_across_heads=cfg.specific_CPs_shared_across_heads,
                peft_for_new_domain=cfg.peft_for_new_domain,
                std=cfg.std_for_CPs,
                device=cfg.device
            )
        elif cfg.parameter_efficient_method == nt.PEFT_Type.LoRA:
            self.LoRAs_for_CLIP = LoRAsForCLIP(
                enable_vision_LoRAs=cfg.enable_vision_LoRAs,
                enable_text_LoRAs=cfg.enable_text_LoRAs,
                dim_embed_vision=self.dim_embed_vision,
                dim_embed_text=self.dim_embed_text,
                downsize_dimension_vision=cfg.LoRA_rank_vision,
                downsize_dimension_text=cfg.LoRA_rank_text,
                LoRA_QKV_mask=cfg.LoRA_QKV_mask,
                task_shared_layers=cfg.task_shared_layers,
                task_specific_layers=cfg.task_specific_layers,
                peft_for_new_domain=cfg.peft_for_new_domain,
                device=cfg.device
            )
        elif cfg.parameter_efficient_method == nt.PEFT_Type.Prompt:
            self.prompts = DeepPrompts(
                num_vision_prompts=cfg.num_vision_prompts,
                num_text_prompts=cfg.num_text_prompts,
                enable_vision_prompts=cfg.enable_vision_prompts,
                enable_text_prompts=cfg.enable_text_prompts,
                dim_embed_vision=self.dim_embed_vision,
                dim_embed_text=self.dim_embed_text,
                task_shared_layers=cfg.task_shared_layers,
                task_specific_layers=cfg.task_specific_layers,
                peft_for_new_domain=cfg.peft_for_new_domain,
                device=cfg.device
            )
        else:
            raise NotImplementedError()
        
        self.classifiers = Classifiers(
            total_sessions=cfg.total_sessions,
            dim_input=self.dim_embed,
            dim_output=self.num_classes,
            classifier_type=cfg.classifier_type,
            peft_for_new_domain=cfg.peft_for_new_domain,
            init_temperature_stochastic_classifier=cfg.init_temperature_stochastic_classifier,
            device=cfg.device
        )
        
        # Learnable shift vectors
        self.embeddings_biases_vision = nn.Parameter(torch.zeros(cfg.total_sessions, self.dim_embed, device=self.device), requires_grad=True)
        self.embeddings_biases_text = nn.Parameter(torch.zeros(1, self.dim_embed, device=self.device), requires_grad=True)
        
        self.cache_manager = CacheManagement(
            dataset_name=cfg.dataset_name,
            order=cfg.order,
            num_epochs=cfg.num_epochs_list[0],
            num_shots=cfg.num_shots,
            dir_cache=cfg.dir_cache
        )
        
        self.statistics_from_frozen_backbone = Statistics(embed_dim=self.dim_embed, device=cfg.device)
        
        self.lsr = LSR(
            cfg=cfg,
            num_classes=self.num_classes,
            dim_embed=self.dim_embed
        )
        
    def train(self):
        cfg = self.cfg
        
        self._prepare_dataloaders()
        
        num_epochs_start = 0
        
        flag_resumed = False
        num_epochs = cfg.num_epochs_list[cfg.current_domain_id]
        
        if cfg.current_domain_id == 0:
            num_epochs_start = self.try_to_resume()
            
            flag_resumed = num_epochs_start > 0
            
            if num_epochs_start > 0:
                logging.info(f"The training is resumed from epoch: {num_epochs_start}")
        
        if not flag_resumed and cfg.use_LSR[cfg.current_domain_id] and num_epochs_start < num_epochs:   # It is required for the LSR
            self.update_statistics_with_frozen_backbone()
        
        # Computing the prototypes from the old domain.
        if cfg.use_prototype_correction and cfg.current_domain_id > 0:
            prototypes_before_training, prototypes_before_training_labels, prototypes_before_training_domain_id = self.compute_calibrated_prototypes(
                domain_id=cfg.current_domain_id,
                message="Computing the prototypes before training for displacement correction ..."
            )       # It become a prototypical classifier at the inference.
        
        # Since we add new task specific CPs, we should initialize a new optimizer.
        optimizer, scheduler = self.get_optimizer_for_training()
        
        ma_loss = ou.MovingAverageDict(capacity=cfg.MA_loss_capacity)
        
        for epoch in tqdm(range(num_epochs_start, num_epochs)):
            self._train_one_epoch(
                epoch=epoch,
                optimizer=optimizer,
                scheduler=scheduler,
                ma_loss=ma_loss
            )
            
            if not cfg.debugging and (cfg.current_domain_id == 0):
                self.save_learned_parameters(update_statistics=False, num_epochs=epoch + 1, domain_id=cfg.current_domain_id)
            
            if cfg.current_domain_id == 0  and ma_loss['loss'] < cfg.MA_loss_threshold_for_early_stopping:
                break
            
        prototypes_after_training, prototypes_after_training_labels, prototypes_after_training_domain_id = self.compute_calibrated_prototypes(
            domain_id=cfg.current_domain_id,
            message=f"Computing the prototypes after training (Domain: {cfg.current_domain_id + 1}/{self.total_sessions}) ..."
        )       # It become a prototypical classifier at the inference.
        
        self.classifiers.add_or_update_final_prototypes(
            prototypes=prototypes_after_training,
            labels=prototypes_after_training_labels,
            domain_ids=prototypes_after_training_domain_id
        )
        
        if cfg.use_prototype_correction and cfg.current_domain_id > 0:
            self.correct_prototypes(
                prototypes_before_training=prototypes_before_training
            )
        
        self.update_statistics_with_frozen_backbone(reset=cfg.current_domain_id == 0)
        
        if cfg.UMAP and cfg.current_domain_id > 0:
             self.save_learned_parameters(update_statistics=False, num_epochs=num_epochs, domain_id=cfg.current_domain_id)
        
    def after_task(
        self
    ):
        self._known_classes = self._total_classes
    
    def _train_one_epoch(
        self,
        epoch: int,
        optimizer: optim.SGD | optim.Adam | optim.AdamW,
        scheduler: ReduceLROnPlateau,
        normalize: bool = True,
        use_power_norm: bool = True,
        shift: bool = True,
        calibration: bool = True,
        ma_loss: ou.MovingAverageDict = None,
    ):
        cfg = self.cfg
        
        use_LSR = cfg.use_LSR[cfg.current_domain_id]
        
        if use_LSR:
            iterator_pseudo_embeddings = self.lsr.obtain_pseudo_embedding_generation_iterator(
                current_domain_id=cfg.current_domain_id,
                statistics=self.statistics_from_frozen_backbone
            )
        else:
            iterator_pseudo_embeddings = None
            
        domain_id = cfg.current_domain_id
        
        if ma_loss is None:
            ma_loss = ou.MovingAverageDict(capacity=cfg.MA_loss_capacity)
        
        num_batches = len(self.train_loader)
        
        ert = ou.EstimatedRemainingTime(total_tasks=num_batches)
        
        for iter_num, batch in enumerate(self.train_loader):
            _, [images, images_not_aug], labels = batch
            labels = labels.to(self.device)
            
            if cfg.dataset_name == 'domainnet' and domain_id > 0 and images.shape[0] > cfg.batch_size_limit_for_incremental_tasks:
                mask = torch.randperm(images.shape[0])[:cfg.batch_size_limit_for_incremental_tasks]
                images = images[mask]
                labels = labels[mask]
            
            # This is for the normal classification
            logits_fused, labels_final = self.forward_multimodal(
                images=images,
                labels=labels,
                use_LSR=use_LSR,
                iterator_pseudo_embeddings=iterator_pseudo_embeddings,
                domain_id=domain_id,
                normalize=normalize,
                use_power_norm=use_power_norm,
                shift=shift,
                calibration=calibration
            )
            
            loss = F.cross_entropy(logits_fused, labels_final)
                
            ma_loss.update(loss=loss)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            remaining_time_str = ert.calculate(num_finished_tasks=iter_num)
            ou.print_overwrite(f'Epoch: {epoch + 1}/{cfg.num_epochs_list[domain_id]}, Batch: {iter_num + 1}/{num_batches} -> M.A. loss={ma_loss['loss']:.4f}, {remaining_time_str}')
            
            if domain_id == 0 and ma_loss['loss'] < cfg.MA_loss_threshold_for_early_stopping:
                break
            
        optimizer.zero_grad()
        logging.info(f'Epoch: {epoch + 1}/{cfg.num_epochs_list[domain_id]}, M.A. loss={ma_loss['loss']:.4f}')
        print()
        scheduler.step(ma_loss['loss'])
            
    def forward_multimodal(
        self,
        images: T,
        labels: T,
        domain_id: int,
        use_LSR: bool,
        iterator_pseudo_embeddings = None,
        normalize: bool = True,
        use_power_norm: bool = True,
        shift: bool = True,
        calibration: bool = True
    ) -> Tuple[T, T]:
        cfg = self.cfg
        
        embeddings_vision = self.forward_vision(
            images=images,
            domain_ids=domain_id,
            normalize=normalize,
            use_power_norm=use_power_norm,
            shift=shift,
            calibration=calibration
        )
        
        labels_without_domain = labels % self.num_classes
        
        if use_LSR:
            if cfg.LSR_generated_classes_labels == nt.LSR_GeneratedClassesLabels.NewLabels:
                num_classes_with_LSR = self.num_classes + cfg.LSR_num_ways_after_filtering_2
            elif cfg.LSR_generated_classes_labels == nt.LSR_GeneratedClassesLabels.InterpolatedLogits:
                num_classes_with_LSR = self.num_classes
            else:
                raise NotImplementedError()
            
            labels_without_domain_one_hot = F.one_hot(labels_without_domain, num_classes=num_classes_with_LSR)
            embeddings_generated, labels_generated_one_hot = next(iterator_pseudo_embeddings)
            embeddings_vision = torch.cat([embeddings_vision, embeddings_generated], dim=0)
            labels_without_domain_one_hot = torch.cat([labels_without_domain_one_hot, labels_generated_one_hot], dim=0)
            labels_without_domain = labels_without_domain_one_hot
        
        prototypes_text = self.obtain_final_text_prototypes_for_one_domain(
            use_PEFT=True,
            domain_id=domain_id
        )
        
        logits_fused = self.classifiers.forward_and_bimodal_calibration_with_a_temporary_classifier(
            embeddings_vision=embeddings_vision,
            prototypes_text=prototypes_text,
            coef_inter_modal_calibration=self.calibration_coefficients.coef_inter_modal_calibration[domain_id],
            coef_visual_prototypes_calibration=self.calibration_coefficients.coef_visual_prototypes_calibration[domain_id],
            calibrate_vision_prototypes=cfg.calibrate_vision_prototypes,
            domain_id=domain_id
        )
            
        return logits_fused, labels_without_domain
    
    @torch.no_grad()
    def correct_prototypes(
        self,
        prototypes_before_training: T
    ):
        self.classifiers.correct_prototypes(
            prototypes_new_task_before=prototypes_before_training,
            current_domain_id=self.cfg.current_domain_id
        )
    
    @torch.no_grad()
    def try_to_resume(
        self,
        domain_id = 0
    ):
        cfg = self.cfg
        
        title = "learned_parameters"
        if cfg.prefix not in [""]:
            title = f"{cfg.prefix},learned_parameters"
        
        for num_epochs in range(cfg.num_epochs_list[domain_id], 0, -1):
            data_dict = self.cache_manager.try_cache_first(
                title=title,
                num_epochs=num_epochs,
                domain_id=domain_id,
                parameter_efficient_method=cfg.parameter_efficient_method
            )
            
            if data_dict is not None:
                self.calibration_coefficients = data_dict["calibration_coefficients"]
                self.power_norm_vision_var = data_dict["power_norm_vision_var"]
                self.power_norm_text_var = data_dict["power_norm_text_var"]
                self.classifiers = data_dict["classifiers"]
                self.embeddings_biases_vision = data_dict["embeddings_biases_vision"]
                self.embeddings_biases_text = data_dict["embeddings_biases_text"]
                self.prototypes_text = data_dict["prototypes_text"]
                self.statistics_from_frozen_backbone = data_dict["statistics_from_frozen_backbone"]
                
                if cfg.parameter_efficient_method == nt.PEFT_Type.CoalescentProjection:
                    self.coalescent_projections = data_dict["coalescent_projections"]
                elif cfg.parameter_efficient_method == nt.PEFT_Type.LoRA:
                    self.LoRAs_for_CLIP = data_dict["LoRAs_for_CLIP"]
                elif cfg.parameter_efficient_method == nt.PEFT_Type.Prompt:
                    self.prompts = data_dict["prompts"]
                else:
                    raise NotImplementedError()
                
                logging.info("Learned parameters are loaded!")
                return num_epochs
        
        return 0
    
    @torch.no_grad()
    def save_learned_parameters(
        self,
        update_statistics: bool,
        num_epochs: int,
        domain_id: int
    ):
        cfg = self.cfg
        
        if num_epochs == 0:     # Perhaps for debugging or the zero-shot setting!
            return
        
        if cfg.current_domain_id == 0:
            if update_statistics:
                self.statistics_from_frozen_backbone = Statistics(embed_dim=self.dim_embed, device=cfg.device)
                self.update_statistics_with_frozen_backbone()
            
        data_dict = {
            "calibration_coefficients": self.calibration_coefficients,
            "power_norm_vision_var": self.power_norm_vision_var,
            "power_norm_text_var": self.power_norm_text_var,
            "classifiers": self.classifiers,
            "embeddings_biases_vision": self.embeddings_biases_vision,
            "embeddings_biases_text": self.embeddings_biases_text,
            "prototypes_text": self.prototypes_text,
            "statistics_from_frozen_backbone": self.statistics_from_frozen_backbone
        }
        
        if cfg.parameter_efficient_method == nt.PEFT_Type.CoalescentProjection:
            data_dict["coalescent_projections"] = self.coalescent_projections
        elif cfg.parameter_efficient_method == nt.PEFT_Type.LoRA:
            data_dict["LoRAs_for_CLIP"] = self.LoRAs_for_CLIP
        elif cfg.parameter_efficient_method == nt.PEFT_Type.Prompt:
            data_dict["prompts"] = self.prompts
        else:
            raise NotImplementedError()
        
        title = "learned_parameters"
        
        if cfg.prefix not in [""]:
            title = f"{cfg.prefix},learned_parameters"
        
        self.cache_manager.save_in_a_cache_file(
            title=title,
            data_dict=data_dict,
            num_epochs=num_epochs,
            domain_id=domain_id,
            parameter_efficient_method=cfg.parameter_efficient_method
        )
            
    @torch.no_grad()
    def _compute_statistics(
        self,
        use_PEFT: bool,
        use_power_norm: bool,
        compute_covariances: bool = True,
        use_more_epochs_for_train_set: bool = True,
        normalize: bool = True,
        shift: bool = False,
        calibration: bool = False,
        coef_regularization: float = 1e-3
    ) -> Tuple[T, T, T]:
        # Notes:
        #   1- We use this function in obtain_statistics_for_LSR and update_statistics_for_domain_id_prediction.
        #   2- We can only rely on the train set
        cfg = self.cfg
        
        arguments = dict(
            use_train_set=True,
            use_test_set=False,
            use_more_epochs_for_train_set=use_more_epochs_for_train_set,
            use_PEFT=use_PEFT,   # In the first task, we haven't trained the CPs.
            use_power_norm=use_power_norm,
            normalize=normalize,
            shift=shift,
            calibration=calibration
        )
        
        # 1. We obtain the embeddings.
        if cfg.current_domain_id == 0:
            embeddings_all, labels_all = self.obtain_visual_embeddings_and_labels_for_all_samples(
                domain_id=cfg.current_domain_id,
                **arguments
                )
        else:
            if not cfg.debugging:
                num_epochs = max(cfg.minimum_num_samples_required_for_statistics_in_incremental_tasks // cfg.num_shots, 1)
            else:
                num_epochs = 5
            
            embeddings_all = torch.tensor([], device='cpu')
            labels_all = torch.tensor([], dtype=torch.long, device='cpu')
            
            ert = ou.EstimatedRemainingTime(total_tasks=num_epochs)
            
            for i in range(num_epochs):
                embeddings, labels = self.obtain_visual_embeddings_and_labels_for_all_samples(
                    domain_id=cfg.current_domain_id,
                    **arguments
                )
                
                embeddings_all = torch.cat([embeddings_all, embeddings], dim=0)
                labels_all = torch.cat([labels_all, labels], dim=0)
                
                remaining_time_str = ert.calculate(num_finished_tasks=i)
                ou.print_overwrite(f"{i}/{num_epochs}, {remaining_time_str}")
            
        # 2. We calculate the means and covariances.
        means_all = torch.tensor([], device=cfg.device)     # means or prototypes
        
        covariances_all: T = None
        if compute_covariances:
            covariances_all = torch.tensor([], device=cfg.device)
        
        assert not cfg.ignore_domain
        labels_unique = labels_all.unique()
        
        num_classes = len(labels_unique)
        ert = ou.EstimatedRemainingTime(total_tasks=num_classes)
        
        for i, lbl in enumerate(labels_unique.tolist()):
            mask = labels_all == lbl
            means = embeddings_all[mask].mean(dim=0, keepdim=True)
            
            means_all = torch.cat([means_all, means.to(self.device)], dim=0)
            
            if compute_covariances:
                covairances = torch.cov(embeddings_all[mask].T) + coef_regularization * torch.eye(self.dim_embed)
                covariances_all = torch.cat([covariances_all, covairances.unsqueeze(0).to(self.device)], dim=0)
                
            remaining_time_str = ert.calculate(num_finished_tasks=i)
            ou.print_overwrite(f"{i}/{num_classes}, {remaining_time_str}")
            
        means_all = means_all.to(self.device)
        if compute_covariances:
            covariances_all = covariances_all.to(self.device)
        labels_unique = labels_unique.to(self.device)
        
        return means_all, covariances_all, labels_unique
    
    @torch.no_grad()
    def update_statistics_with_frozen_backbone(
        self,
        reset=False
    ):
        # Note: We call this function after the training is finished on each domain.
        cfg = self.cfg
        
        logging.info("Computing the statistics from the frozen backbone ...")
        
        if reset and cfg.current_domain_id == 0:
            self.statistics_from_frozen_backbone = Statistics(embed_dim=self.dim_embed, device=cfg.device)
            logging.info("The statistics are recalculated.")
        
        means, covariances, labels = self._compute_statistics(
            use_PEFT=False,
            use_power_norm=True,
            use_more_epochs_for_train_set=cfg.current_domain_id > 0,
            compute_covariances=True,
            normalize=True,
            shift=False,
            calibration=False
        )
            
        self.statistics_from_frozen_backbone.update(
            means=means,
            covariances=covariances,
            labels=labels,
            domain_id=cfg.current_domain_id
        )
        
    @torch.no_grad()
    def compute_calibrated_prototypes(
        self,
        domain_id: int,
        message: str = "Computing the calibrated prototypes ...",
    ):
        # We compute and calibrate the prototypes and store them in the weights matrix of the prototypical classifier for the inference. Therefore, at the inference process, we do not need to calibrate them again.
        cfg = self.cfg
        
        domain_id = cfg.current_domain_id
        
        logging.info(message)
        
        prototypes_vision, _, prototypes_vision_labels = self._compute_statistics(
            use_PEFT=True,
            use_power_norm=True,
            compute_covariances=False,
            use_more_epochs_for_train_set=cfg.current_domain_id > 0,       # There are sufficient samples on the base domain.
            normalize=True,
            shift=True,
            calibration=True
        )
        
        prototypes_text = self.obtain_final_text_prototypes_for_one_domain(domain_id=domain_id)
        
        embeddings_calibrated = ou.interpolate(
            coef=self.calibration_coefficients.coef_inter_modal_calibration[domain_id],
            a=prototypes_vision,
            b=prototypes_text[prototypes_vision_labels % self.num_classes]
        )
        
        prototypes, _, prototypes_labels = self.calculate_means_and_covariances(
            embeddings_all=embeddings_calibrated,
            labels_all=prototypes_vision_labels,
            calculate_covariance=False
        )
        
        return prototypes, prototypes_labels, domain_id
        
    def get_optimizer_for_training(self):
        # Note: We should call it for every new domain.
        cfg = self.cfg
        
        params_all = []
        
        lr_default = cfg.lr_default
        
        params_all += ou.get_params_groups(model=self.calibration_coefficients, name_model='Learnable Coefficients', lr=cfg.lr_calibration_coefficients[cfg.current_domain_id])
        params_all += ou.get_params_groups(model=self.power_norm_vision_var, name_model='Learnable Coefficients', lr=cfg.lr_power_norm[cfg.current_domain_id])
        params_all += ou.get_params_groups(model=self.power_norm_text_var, name_model='Learnable Coefficients', lr=cfg.lr_power_norm[cfg.current_domain_id])
        
        if cfg.parameter_efficient_method == nt.PEFT_Type.CoalescentProjection:
            
            if cfg.enable_vision_CPs:
                params_all += ou.get_params_groups(model=self.coalescent_projections.CPs_shared_vision_dict, name_model='Coalescent Projections, vision,shared', lr=cfg.lr_PEFT_vision_shared[cfg.current_domain_id], disable_weight_decay=True)
                params_all += ou.get_params_groups(model=self.coalescent_projections.CPs_specific_vision_dict, name_model='Coalescent Projections, vision,specific', lr=cfg.lr_PEFT_vision_specific[cfg.current_domain_id], disable_weight_decay=True)
            if cfg.enable_text_CPs:
                params_all += ou.get_params_groups(model=self.coalescent_projections.CPs_shared_text_dict, name_model='Coalescent Projections, Text,shared', lr=cfg.lr_PEFT_text_shared[cfg.current_domain_id], disable_weight_decay=True)
                params_all += ou.get_params_groups(model=self.coalescent_projections.CPs_specific_text_dict, name_model='Coalescent Projections, Text,specific', lr=cfg.lr_PEFT_text_specific[cfg.current_domain_id], disable_weight_decay=True)
        elif cfg.parameter_efficient_method == nt.PEFT_Type.LoRA:
            
            if cfg.enable_vision_LoRAs:
                params_all += ou.get_params_groups(model=self.LoRAs_for_CLIP.LoRAs_shared_vision_dict, name_model='LoRAs, vision,shared', lr=cfg.lr_PEFT_vision_shared[cfg.current_domain_id], disable_weight_decay=True)
                
                params_all += ou.get_params_groups(model=self.LoRAs_for_CLIP.LoRAs_specific_vision_dict, name_model='LoRAs, vision,specific', lr=cfg.lr_PEFT_vision_specific[cfg.current_domain_id], disable_weight_decay=True)
            if cfg.enable_text_LoRAs:
                params_all += ou.get_params_groups(model=self.LoRAs_for_CLIP.LoRAs_shared_text_dict, name_model='LoRAs, Text,shared', lr=cfg.lr_PEFT_text_shared[cfg.current_domain_id], disable_weight_decay=True)
                params_all += ou.get_params_groups(model=self.LoRAs_for_CLIP.LoRAs_specific_text_dict, name_model='LoRAs, Text,specific', lr=cfg.lr_PEFT_text_specific[cfg.current_domain_id], disable_weight_decay=True)
        
        elif cfg.parameter_efficient_method == nt.PEFT_Type.Prompt:
            if cfg.enable_vision_prompts:
                params_all += ou.get_params_groups(model=self.prompts.prompts_shared_vision_dict, name_model='Prompts, vision,shared', lr=cfg.lr_PEFT_vision_shared[cfg.current_domain_id], disable_weight_decay=True)
                
                params_all += ou.get_params_groups(model=self.prompts.prompts_specific_vision_dict, name_model='Prompts, vision,specific', lr=cfg.lr_PEFT_vision_specific[cfg.current_domain_id], disable_weight_decay=True)
            if cfg.enable_text_prompts:
                params_all += ou.get_params_groups(model=self.prompts.prompts_shared_text_dict, name_model='Prompts, Text,shared', lr=cfg.lr_PEFT_text_shared[cfg.current_domain_id], disable_weight_decay=True)
                params_all += ou.get_params_groups(model=self.prompts.prompts_specific_text_dict, name_model='Prompts, Text,specific', lr=cfg.lr_PEFT_text_specific[cfg.current_domain_id], disable_weight_decay=True)
        else:
            raise NotImplementedError()
        
        params_all += ou.get_params_groups(model=self.classifiers, name_model='Classifiers', lr=cfg.lr_classifier[cfg.current_domain_id], weight_decay=cfg.weight_decay_classifiers[cfg.current_domain_id])
        
        params_all += ou.get_params_groups(model=self.embeddings_biases_vision, name_model="Embeddings' biases", lr=cfg.lr_embeddings_biases[cfg.current_domain_id], disable_weight_decay=True)
        params_all += ou.get_params_groups(model=self.embeddings_biases_text, name_model="Embeddings' biases", lr=cfg.lr_embeddings_biases[cfg.current_domain_id], disable_weight_decay=True)
            
        ou.show_number_of_parameters_in_pramas_groups(params_all=params_all, logger=logging)
        
        optimizer = ou.get_optimizer_from_params(params_all=params_all, optimizer_name=cfg.optimizer_name[cfg.current_domain_id], lr_default=lr_default[cfg.current_domain_id], weight_decay=cfg.weight_decay[cfg.current_domain_id])
        
        scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=1)
        
        return optimizer, scheduler
        
    def _forward_vision_frozen_backbone_single_domain(
        self,
        images: T,
        domain_id: int | None,
        normalize: bool = True,
        use_power_norm: bool = True,
        shift: bool = True,
        calibration: bool = True,
    ):
        images = images.to(self.device)
        embeddings = self.model.visual.forward(images, chosen_layer=self.cfg.chosen_layer_vision)
        
        if normalize:
            embeddings = F.normalize(embeddings, dim=-1)
        
        if domain_id is not None: 
            if shift:
                if domain_id > 0:
                    # We do not update the biases from the base domain.
                    embeddings_biases_vision_base_domain: T = F.normalize(self.embeddings_biases_vision[0].detach().requires_grad_(False), dim=-1)
                    embeddings_biases_vision_base_domain.grad = None
                    
                    if normalize:
                        embeddings_biases_vision_base_domain = F.normalize(embeddings_biases_vision_base_domain, dim=-1)
                    
                    coef = self.calibration_coefficients.coef_shift_vision_from_first_domain[domain_id]
                    
                    embeddings = embeddings + coef * embeddings_biases_vision_base_domain
                
                embeddings_biases_vision_current_domain = self.embeddings_biases_vision[domain_id]
                
                if normalize:
                    embeddings_biases_vision_current_domain = F.normalize(embeddings_biases_vision_current_domain, dim=-1)
                
                embeddings = embeddings + self.calibration_coefficients.coef_shift_vision_from_current_domain[domain_id] * embeddings_biases_vision_current_domain
                
                if normalize:
                    embeddings = F.normalize(embeddings, dim=-1)
                
            if use_power_norm:
                embeddings = ou.power_norm(embeddings, alpha=self.power_norm_vision_var[domain_id])
        
        return embeddings
    
    def _forward_vision_frozen_backbone_with_domain_ids(
        self,
        images: T,
        domain_ids: T,
        normalize: bool = True,
        use_power_norm: bool = True,
        shift: bool = True,
        calibration: bool = True,
    ):
        domain_ids_unique_list = domain_ids.unique().tolist()
        
        embeddings_final = torch.zeros(images.shape[0], self.dim_embed, device=self.device)
        
        for domain_id in domain_ids_unique_list:
            mask = domain_ids == domain_id
            images_chosen_domain = images[mask]
            
            if images_chosen_domain.dim() == 3:
                images_chosen_domain = images_chosen_domain.unsqueeze(0)
                
            assert images_chosen_domain.dim() == 4
            
            embeddings_chosen_domain = self._forward_vision_frozen_backbone_single_domain(
                images=images_chosen_domain,
                domain_id=domain_id,
                normalize=normalize,
                use_power_norm=use_power_norm,
                shift=shift,
                calibration=calibration
            )
            embeddings_final[mask] = embeddings_chosen_domain
            
        return embeddings_final
        
    def forward_vision_frozen_backbone(
        self,
        images: T,
        domain_ids: T | int | None,
        normalize: bool = True,
        use_power_norm: bool = True,
        shift: bool = True,
        calibration: bool = True,
    ):
        shared_arguments = dict(
            images=images,
            normalize=normalize,
            use_power_norm=use_power_norm,
            shift=shift,
            calibration=calibration
        )
        
        if domain_ids is None:
            embeddings = self._forward_vision_frozen_backbone_single_domain(
                domain_id=None,
                **shared_arguments
            )
        elif isinstance(domain_ids, int):
            embeddings = self._forward_vision_frozen_backbone_single_domain(
                domain_id=domain_ids,
                **shared_arguments
            )
        elif isinstance(domain_ids, T):
            embeddings = self._forward_vision_frozen_backbone_with_domain_ids(
                domain_ids=domain_ids,
                **shared_arguments
            )
        else:
            raise NotImplementedError()
            
        return embeddings
        
    
    def _forward_vision_single_domain(
        self,
        images: T,
        domain_id: int,
        normalize: bool = True,
        use_power_norm: bool = True,
        shift: bool = True,
        calibration: bool = True,
    ):
        cfg = self.cfg
        
        images = images.to(self.device)
        
        arguments_shared = dict(
            chosen_layer=self.cfg.chosen_layer_vision,
        )
        
        coalescent_projections_dict, LoRAs_dict, prompts_dict = self._obtain_PEFT_parameters(
            modality='vision',
            domain_id=domain_id
        )
        
        arguments_shared["coalescent_projections_dict"] = coalescent_projections_dict
        arguments_shared["LoRAs_dict"] = LoRAs_dict
        arguments_shared["prompts_dict"] = prompts_dict
        
        if len(images) < cfg.chunk_size_vision:
            embeddings = self.model.visual.forward(images, **arguments_shared)
        else:
            ds = TensorDataset(images)
            dl = DataLoader(ds, batch_size=cfg.chunk_size_vision, shuffle=False, drop_last=False)
            
            embeddings = torch.tensor([], device=self.device)
            
            for [x_batch] in dl:
                embd = self.model.visual.forward(x_batch, **arguments_shared)
                embeddings = torch.cat([embeddings, embd], dim=0)
        
        if normalize:
            embeddings = F.normalize(embeddings, dim=-1)
            
        if shift:
            if domain_id > 0:
                # We do not update the biases from the base domain.
                embeddings_biases_vision_base_domain: T = F.normalize(self.embeddings_biases_vision[0].detach().requires_grad_(False), dim=-1)
                embeddings_biases_vision_base_domain.grad = None
                
                if normalize:
                    embeddings_biases_vision_base_domain = F.normalize(embeddings_biases_vision_base_domain, dim=-1)
                
                coef = self.calibration_coefficients.coef_shift_vision_from_first_domain[domain_id]
                
                embeddings = embeddings + coef * embeddings_biases_vision_base_domain
            
            embeddings_biases_vision_current_domain = self.embeddings_biases_vision[domain_id]
            
            if normalize:
                embeddings_biases_vision_current_domain = F.normalize(embeddings_biases_vision_current_domain, dim=-1)
            
            embeddings = embeddings + self.calibration_coefficients.coef_shift_vision_from_current_domain[domain_id] * embeddings_biases_vision_current_domain
            
            if normalize:
                embeddings = F.normalize(embeddings, dim=-1)
        
        if use_power_norm:
            embeddings = ou.power_norm(embeddings, alpha=self.power_norm_vision_var[domain_id])
        
        return embeddings
    
    def _forward_vision_with_domain_ids(
        self,
        images: T,
        domain_ids: T,
        normalize: bool = True,
        use_power_norm: bool = True,
        shift: bool = True,
        calibration: bool = True,
    ) -> T:
        images = images.to(self.device)
        
        domain_ids_unique_list = domain_ids.unique().tolist()
        
        embeddings_final = torch.zeros(images.shape[0], self.dim_embed, device=self.device)
        
        for domain_id in domain_ids_unique_list:
            mask = domain_ids == domain_id
            images_chosen_domain = images[mask]
            
            if images_chosen_domain.dim() == 3:
                images_chosen_domain = images_chosen_domain.unsqueeze(0)
                
            assert images_chosen_domain.dim() == 4
            
            embeddings_chosen_domain = self._forward_vision_single_domain(
                images=images_chosen_domain,
                domain_id=domain_id,
                normalize=normalize,
                use_power_norm=use_power_norm,
                shift=shift,
                calibration=calibration
            )
            embeddings_final[mask] = embeddings_chosen_domain
            
        return embeddings_final
    
    def forward_vision(
        self,
        images: T,
        domain_ids: int | T | None,
        normalize: bool = True,
        use_power_norm: bool = True,
        shift: bool = True,
        calibration: bool = True,
    ) -> T:
        images = images.to(self.device)
        
        shared_arguments = dict(
            images=images,
            normalize=normalize,
            use_power_norm=use_power_norm,
            shift=shift,
            calibration=calibration
        )
        
        if domain_ids is None:
            embeddings = self.forward_vision_frozen_backbone(
                domain_ids=domain_ids,
                **shared_arguments
            )
        elif isinstance(domain_ids, int):
            embeddings = self._forward_vision_single_domain(
                domain_id=domain_ids,
                **shared_arguments
            )
        elif isinstance(domain_ids, T):
            embeddings = self._forward_vision_with_domain_ids(
                domain_ids=domain_ids,
                **shared_arguments
            )
        else:
            raise NotImplementedError()
            
        return embeddings
            
    def _forward_text_frozen_backbone(
        self,
        text_tokens: T,
        normalize: bool = True
    ):
        text_tokens = text_tokens.to(self.device)
        
        embeddings = self.model.encode_text(text_tokens)
        if normalize:
            embeddings = F.normalize(embeddings, dim=-1)
        return embeddings
    
    def _forward_text_single_domain(
        self,
        text_tokens: T,
        domain_id: int,
        normalize: bool = True,
        use_power_norm: bool = True,
        shift: bool = True,
        use_PEFT: bool = True,
    ) -> T:
        cfg = self.cfg
        text_tokens = text_tokens.to(self.device)
        
        if use_PEFT:
            coalescent_projections_dict, LoRAs_dict, prompts_dict = self._obtain_PEFT_parameters(
                modality='text',
                domain_id=domain_id
            )
        else:
            coalescent_projections_dict = None
            LoRAs_dict = None
            prompts_dict = None
            
        shared_arguments = dict(
            coalescent_projections_dict=coalescent_projections_dict,
            LoRAs_dict=LoRAs_dict,
            prompts_dict=prompts_dict
        )
        
        if len(text_tokens) < cfg.chunk_size_text:
            embeddings = self.model.encode_text(
                text=text_tokens,
                **shared_arguments
            )
        else:
            ds = TensorDataset(text_tokens)
            dl = DataLoader(ds, batch_size=cfg.chunk_size_text, shuffle=False, drop_last=False)
            
            embeddings = torch.tensor([], device=self.device)
            
            for [x_batch] in dl:
                embd = self.model.encode_text(
                    text=x_batch,
                    **shared_arguments
                )
                embeddings = torch.cat([embeddings, embd], dim=0)
                
        if normalize:
            embeddings = F.normalize(embeddings, dim=-1)
            
        if shift:
            shift_vision_current_domain = self.embeddings_biases_vision[domain_id].unsqueeze(0)
            learnable_shift_text = self.embeddings_biases_text
            
            if normalize:
                shift_vision_current_domain = F.normalize(shift_vision_current_domain, dim=-1)
                learnable_shift_text = F.normalize(learnable_shift_text, dim=-1)
                
            text_bias_correction = self.calibration_coefficients.coef_shift_text[domain_id] * learnable_shift_text + self.calibration_coefficients.coef_shift_text_from_current_domain_in_vision_modality[domain_id] * shift_vision_current_domain
            
            if domain_id > 0:
                embeddings_biases_vision_base_domain: T = F.normalize(self.embeddings_biases_vision[0].detach().requires_grad_(False), dim=-1)
                embeddings_biases_vision_base_domain.grad = None
                
                text_bias_correction = text_bias_correction + self.calibration_coefficients.coef_shift_text_from_first_domain_in_vision_modality[domain_id] * embeddings_biases_vision_base_domain
                
            embeddings = embeddings + text_bias_correction
        
        if use_power_norm:
            embeddings = ou.power_norm(embeddings, alpha=self.power_norm_text_var[domain_id])
        
        return embeddings
    
    def _forward_text_with_domain_ids(
        self,
        text_tokens: T,
        domain_ids: T,
        normalize: bool = True,
        use_power_norm: bool = True,
        shift: bool = True,
        use_PEFT: bool = True       # Despite having the domain IDs, we may require to ignore the CPs.
    ):
        text_tokens = text_tokens.to(self.device)
        
        domain_ids_unique_list = domain_ids.unique().tolist()
        
        embeddings_final = torch.zeros(text_tokens.shape[0], self.dim_embed, device=self.device)
        
        for domain_id in domain_ids_unique_list:
            mask = domain_ids = domain_id
            text_tokens_chosen_domain = text_tokens[mask]
            
            embeddings_chosen_domain = self._forward_text_single_domain(
                use_PEFT=use_PEFT,
                text_tokens=text_tokens_chosen_domain,
                domain_id=domain_id,
                normalize=normalize,
                shift=shift,
                use_power_norm=use_power_norm
            )
            embeddings_final[mask] = embeddings_chosen_domain
            
        return embeddings_final
    
    def forward_text(
        self,
        texts: Union[str, List[str]],
        domain_ids: int | T | None,
        use_power_norm: bool = True,
        normalize: bool = True,
        shift: bool = True,
        use_PEFT: bool = True
    ):
        
        text_tokens = clip.tokenize(texts).to(self.device)
        
        if domain_ids is None:
            embeddings = self._forward_text_frozen_backbone(
                text_tokens=text_tokens,
                normalize=normalize
            )
        elif isinstance(domain_ids, int):
            embeddings = self._forward_text_single_domain(
                use_PEFT=use_PEFT,
                text_tokens=text_tokens,
                domain_id=domain_ids,
                normalize=normalize,
                shift=shift,
                use_power_norm=use_power_norm
            )
        elif isinstance(domain_ids, T):
            embeddings = self._forward_text_with_domain_ids(
                use_PEFT=use_PEFT,
                text_tokens=text_tokens,
                domain_ids=domain_ids,
                normalize=normalize,
                shift=shift,
                use_power_norm=use_power_norm
            )
            
        
        return embeddings
    
    def prepare_for_current_domain(self):
        cfg = self.cfg
        
        domain_id = cfg.current_domain_id
        
        if cfg.parameter_efficient_method == nt.PEFT_Type.CoalescentProjection:
            self.coalescent_projections.prepare_for_a_new_task(domain_id=domain_id)
        elif cfg.parameter_efficient_method == nt.PEFT_Type.LoRA:
            self.LoRAs_for_CLIP.prepare_for_a_new_task(domain_id=domain_id)
        elif cfg.parameter_efficient_method == nt.PEFT_Type.Prompt:
            self.prompts.prepare_for_a_new_task(domain_id=domain_id)
        else:
            raise NotImplementedError()
        
        self.classifiers.prepare_for_a_new_task(domain_id=domain_id)
        
        self._total_classes = self._known_classes + self.num_classes
        
        if cfg.ignore_domain:
            if domain_id == 0:
                self.label_to_domain_id = torch.zeros(self._total_classes - self._known_classes, dtype=torch.long, device=self.device)
        else:
            self.label_to_domain_id = torch.cat([
                self.label_to_domain_id, 
                domain_id * torch.ones(self._total_classes - self._known_classes, dtype=torch.long, device=self.device)
            ])

        logging.info(f"Domain: {domain_id + 1}/{self.total_sessions}, Classes: {self._known_classes}-{self._total_classes}")
        
        if domain_id > 0 and cfg.peft_for_new_domain in [nt.InitializationApproachForIncrementalTasks.CopyFromFirstDomain, nt.InitializationApproachForIncrementalTasks.CopyFromPreviousDomain]:
            old_domain = domain_id - 1
            
            with torch.no_grad():
                self.embeddings_biases_vision.data[domain_id] = self.embeddings_biases_vision.data[old_domain]
                # self.embeddings_biases_text.data[domain_id] = self.embeddings_biases_text.data[old_domain]
                self.embeddings_biases_vision.grad = None
                self.embeddings_biases_text.grad = None
                
                self.calibration_coefficients.coef_synonyms_prototypes.data[domain_id] = self.calibration_coefficients.coef_synonyms_prototypes.data[old_domain]
                
                self.calibration_coefficients.coef_visual_prototypes_calibration.data[domain_id] = self.calibration_coefficients.coef_visual_prototypes_calibration.data[old_domain]
                
                self.calibration_coefficients.coef_inter_modal_calibration.data[domain_id] = self.calibration_coefficients.coef_inter_modal_calibration.data[old_domain]
                
                self.calibration_coefficients.coef_shift_text_from_current_domain_in_vision_modality.data[domain_id] = self.calibration_coefficients.coef_shift_text_from_current_domain_in_vision_modality.data[old_domain]
                
                self.calibration_coefficients.coef_shift_text_from_first_domain_in_vision_modality.data[domain_id] = self.calibration_coefficients.coef_shift_text_from_first_domain_in_vision_modality.data[old_domain]
                
                self.calibration_coefficients.coef_shift_text.data[domain_id] = self.calibration_coefficients.coef_shift_text.data[old_domain]
                
                self.calibration_coefficients.coef_shift_vision_from_first_domain.data[domain_id] = self.calibration_coefficients.coef_shift_vision_from_first_domain.data[old_domain]
                
                self.calibration_coefficients.coef_shift_vision_from_current_domain.data[domain_id] = self.calibration_coefficients.coef_shift_vision_from_current_domain.data[old_domain]
                
                self.power_norm_vision_var.data[domain_id] = self.power_norm_vision_var.data[old_domain]
                self.power_norm_text_var.data[domain_id] = self.power_norm_text_var.data[old_domain]
                
    def _evaluate(self, y_pred, y_true):
        ret = {}
        grouped = accuracy_domain_shot(
            y_pred.T[0],
            y_true,
            self._known_classes,
            num_classes=self.num_classes,
            many_shot=self.data_manager.many_shot_classes,
            medium_shot=self.data_manager.medium_shot_classes,
            few_shot=self.data_manager.few_shot_classes,
        )
        ret["grouped"] = grouped
        ret["top1"] = grouped["total"]
        return ret

    def compute_text_prototypes_with_templates(
        self, 
        templates_list: list[str],
        names_list: list[str],
        domain_ids: int | T | None,
        use_power_norm: bool = True,
        normalize: bool = True,
        shift: bool = True,
        use_PEFT: bool = True
    ):
        """This method calculates the mean of embeddings for all templates for every class name

        Args:
            names_list (list[str]): The list of names

        Returns:
            _type_: _description_
        """
        cfg = self.cfg
        
        texts_list = []
        
        for name in names_list:
            texts_list += [template.format(class_name=name) for template in templates_list]
        
        embeddings = self.forward_text(
            use_PEFT=use_PEFT,
            texts=texts_list,
            domain_ids=domain_ids,
            use_power_norm=use_power_norm,
            normalize=normalize,
            shift=shift
        )
        
        # Its shape: [len(names_list), len(templates_list), dim_embed]
        embeddings = eo.rearrange(embeddings, '(m n) d -> m n d', m=len(names_list), n=len(templates_list))
        prototypes = embeddings.mean(dim=1)
            
        if normalize:
            embeddings = F.normalize(embeddings, dim=-1)
            prototypes = F.normalize(prototypes, dim=-1)
        
        return prototypes, embeddings
    
    def compute_text_prototypes_with_weighted_synonyms_and_templates(
        self,
        domain_id: int,
        num_synonyms_limit: int = 10,
        use_power_norm: bool = True,
        normalize: bool = True,
        shift: bool = True,
        use_PEFT: bool = True
    ):
        cfg = self.cfg
        
        # This method will return the interpolation of the prototype of the original name and the weighted synonyms prototypes.
        
        prototypes_original_class_names, embeddings_texts = self.compute_text_prototypes_with_templates(
            templates_list=cfg.CLIP_templates,
            names_list=self.class_names,
            domain_ids=domain_id,
            use_power_norm=use_power_norm,
            normalize=normalize,
            shift=shift
        )
        
        names_list_all = []
        
        for i, name in enumerate(self.class_names):
            names_list_all += self.synonyms_dict[name][:num_synonyms_limit]
        
        with torch.no_grad():
            # Its shape: [self.num_classes * (1 + num_synonyms_limit), dim_embed]
            prototypes_synonyms_all, embeddings_texts = self.compute_text_prototypes_with_templates(
                templates_list=cfg.CLIP_templates,
                names_list=names_list_all,
                domain_ids=domain_id,
                use_power_norm=use_power_norm,
                normalize=normalize,
                shift=shift
            )
        
        prototypes_synonyms_list = []
        
        for i, name in enumerate(self.class_names):
            # Its shape: [num_synonyms_current_class_name, dim_embed]
            index_start = i * num_synonyms_limit
            index_end = (i + 1) * num_synonyms_limit
            prototypes_synonyms_current_name = prototypes_synonyms_all[index_start:index_end].detach()   # We ignore their gradients.
            
            similarity_current_class_name = ou.similarity(prototypes_original_class_names[i].unsqueeze(0), prototypes_synonyms_current_name)        # Its shape: [num_synonyms_current_class_name, num_synonyms]
            weights = similarity_current_class_name.softmax(1)      # Its shape: [1, num_synonyms]
            weighted_prototype_current_class = torch.einsum('ij, jk -> k', weights, prototypes_synonyms_current_name)
            prototypes_synonyms_list.append(weighted_prototype_current_class)
        
        prototypes_weighted_synonyms = torch.stack(prototypes_synonyms_list)     # Its shape: [num_classes, dim_embed]
        
        prototypes_final = ou.interpolate(
            coef=self.calibration_coefficients.coef_synonyms_prototypes[domain_id],
            a=prototypes_weighted_synonyms,
            b=prototypes_original_class_names
        )
        
        return prototypes_final
    
    def compute_text_prototypes_with_weighted_class_names_and_synonyms_and_templates(
        self,
        domain_id: int,
        num_synonyms_limit: int = 10,
        use_power_norm: bool = True,
        normalize: bool = True,
        shift: bool = True,
        use_PEFT: bool = True
    ):
        # This method will return the weighted class names + synonyms prototypes. We don't perform interpolation here!
        cfg = self.cfg
        
        # Its shape: [num_classes, dim_embed]
        prototypes_original_class_names, _ = self.compute_text_prototypes_with_templates(
            templates_list=cfg.CLIP_templates,
            names_list=self.class_names,
            domain_ids=domain_id,
            use_power_norm=use_power_norm,
            normalize=normalize,
            shift=shift
        )
        
        prototypes_synonyms_list = []
        
        for i, name in enumerate(self.class_names):
            # names_list = [self.class_names[i]]      # Real class names
            names_list = self.synonyms_dict[name][:num_synonyms_limit]              # We add its synonyms to the list
            # Its shape: [num_synonyms_current_class_name, dim_embed]
            prototypes_synonyms_current_name, _ = self.compute_text_prototypes_with_templates(
                use_PEFT=use_PEFT,
                templates_list=cfg.CLIP_templates,
                names_list=names_list,
                domain_ids=domain_id,
                use_power_norm=use_power_norm,
                normalize=normalize,
                shift=shift
            )
            similarity_current_class_name = ou.similarity(prototypes_original_class_names[i].unsqueeze(0), prototypes_synonyms_current_name)        # Its shape: [num_synonyms_current_class_name, num_synonyms]
            weights = similarity_current_class_name.softmax(1)      # Its shape: [1, num_synonyms]
            # weighted_prototype_current_class = (weights.view(-1, 1) * prototypes_synonyms_current_name).sum(0)
            weighted_prototype_current_class = torch.einsum('ij, jk -> k', weights, prototypes_synonyms_current_name)
            prototypes_synonyms_list.append(weighted_prototype_current_class)
        
        prototypes_weighted_synonyms = torch.stack(prototypes_synonyms_list)     # Its shape: [num_classes, dim_embed]
        
        return prototypes_weighted_synonyms
    
    def obtain_final_text_prototypes_for_one_domain(
        self,
        domain_id: int,
        use_power_norm: bool = True,
        normalize: bool = True,
        shift: bool = True,
        use_PEFT: bool = True
    ):
        """We need to call this method once to compute the text prototypes.

        Raises:
            NotImplementedError: _description_
        """
        cfg = self.cfg
        
        assert isinstance(domain_id, int) or domain_id is None
        
        criterion = cfg.prototype_calculation_mode_text
            
        if criterion == nt.PrototypeTextModality.Templates:
            prototypes_text, _ = self.compute_text_prototypes_with_templates(
                templates_list=cfg.CLIP_templates,
                names_list=self.class_names,
                domain_ids=domain_id,
                use_power_norm=use_power_norm,
                normalize=normalize,
                shift=shift
            )
        elif criterion == nt.PrototypeTextModality.WeightedSynonyms:
            prototypes_text = self.compute_text_prototypes_with_weighted_synonyms_and_templates(
                domain_id=domain_id,
                use_power_norm=use_power_norm,
                normalize=normalize,
                shift=shift,
                num_synonyms_limit=cfg.num_synonyms_limit
            )
        elif criterion == nt.PrototypeTextModality.WeightedClassNamesAndSynonyms:
            prototypes_text = self.compute_text_prototypes_with_weighted_class_names_and_synonyms_and_templates(
                domain_id=domain_id,
                use_power_norm=use_power_norm,
                normalize=normalize,
                shift=shift
            )
        else:
            raise NotImplementedError()
        
        if normalize:
            prototypes_text = F.normalize(prototypes_text, dim=-1)
        
        return prototypes_text
    
    def obtain_final_text_prototypes_for_various_domains(
        self,
        domain_ids: int | T,
        use_power_norm: bool = True,
        normalize: bool = True,
        shift: bool = True,
        use_PEFT: bool = True
    ) -> Dict[int, T]:
        # Note: We use this method for evaluation of the test set.
        shared_arguments = dict(
            use_power_norm=use_power_norm,
            normalize=normalize,
            shift=shift
        )
        
        domain_id_to_prototypes_text_dict = {}
        
        if isinstance(domain_ids, int):
            prototypes_text = self.obtain_final_text_prototypes_for_one_domain(
                domain_id=None,
                **shared_arguments
            )
            
            domain_id_to_prototypes_text_dict[domain_ids] = prototypes_text
        elif isinstance(domain_ids, T):
            domain_ids_unique = domain_ids.unique().tolist()
            
            for domain_id in domain_ids_unique:
                prototypes_text = self.obtain_final_text_prototypes_for_one_domain(
                    domain_id=domain_id,
                    **shared_arguments
                )
                domain_id_to_prototypes_text_dict[domain_id] = prototypes_text
        else:
            raise NotImplementedError()
        
        return domain_id_to_prototypes_text_dict
        
    def predict_with_image_embedding(
        self,
        embeddings_images: T,
        prototypes: T,
        labels_prototypes: T,
        domain_ids_for_oracle: T = None,
        normalize_prototypes: bool = True,
        return_logits: bool = False
    ):
        
        cfg = self.cfg
        assert embeddings_images.dim() == 2
        
        embeddings_images = F.normalize(embeddings_images, dim=-1)
        if normalize_prototypes:
            prototypes = F.normalize(prototypes, dim=-1)
            
        logits_final: T = None
        
        if cfg.ignore_domain:
            num_classes = self.num_classes
        else:
            num_classes = self.num_classes * (cfg.current_domain_id + 1)
        
        if domain_ids_for_oracle is not None:
            assert len(domain_ids_for_oracle) == len(embeddings_images)
            
            predicted_labels = torch.zeros_like(domain_ids_for_oracle) - 1
            logits_final = torch.zeros(len(embeddings_images), num_classes, device=self.device)
            
            for domain_id in range(cfg.current_domain_id + 1):
                mask_prototypes_this_domain = self.label_to_domain_id[labels_prototypes] == domain_id
                prototypes_this_domain = prototypes[mask_prototypes_this_domain]
                labels_prototypes_this_domain = labels_prototypes[mask_prototypes_this_domain]
                
                mask_embeddings = domain_ids_for_oracle == domain_id
                embeddings_images_this_domain = embeddings_images[mask_embeddings]
                
                similarities = ou.similarity(features_1=embeddings_images_this_domain, features_2=prototypes_this_domain, cosine=True)
                
                if return_logits:
                    if len(prototypes_this_domain) != self.num_classes:
                        raise NotImplementedError()
                    logits = similarities.softmax(dim=-1)
                    # assert torch.isnan(logits).sum() == 0
                    
                    if cfg.ignore_domain:
                        logits_final[mask_embeddings] = logits
                    else:
                        lb = self.num_classes * domain_id
                        ub = self.num_classes * (domain_id + 1)
                        logits_final[mask_embeddings][:, lb:ub] = logits
                        pass
                    
                predicted_indices = similarities.argmax(dim=1)
                predicted_labels[mask_embeddings] = labels_prototypes_this_domain[predicted_indices]
                
            if cfg.ignore_domain:
                predicted_labels = predicted_labels % num_classes
        else:
            raise NotImplementedError()
                
        if return_logits:
            assert len(predicted_labels) == len(logits_final)
        
        predicted_labels = predicted_labels.to(self.device)
        return predicted_labels, logits_final
    
    @torch.no_grad()
    def calculate_accuracy_per_domain(
        self,
        labels_predicted: T,
        labels_ground_truth: T
    ):
        cfg = self.cfg
        true_domains_ids = self.label_to_domain_id[labels_ground_truth]

        acc_per_domain = torch.zeros(cfg.current_domain_id + 1)

        for domain_id in range(cfg.current_domain_id + 1):
            mask = true_domains_ids == domain_id
            acc_per_domain[domain_id] = self.calculate_accuracy(labels_predicted=labels_predicted[mask], labels_ground_truth=labels_ground_truth[mask])
            
        average_accuracy = acc_per_domain.mean()
            
        return acc_per_domain, average_accuracy
    
    @torch.no_grad()
    def calculate_accuracy(
        self,
        labels_predicted: T,
        labels_ground_truth: T
    ) -> float:
        labels_predicted = labels_predicted.to(self.device)
        labels_ground_truth = labels_ground_truth.to(self.device)
        
        acc = 100.0 * (labels_predicted % self.num_classes == labels_ground_truth % self.num_classes).float().mean().item()
        return acc
    
    @torch.no_grad()
    def evaluate_on_test_set(
        self,
        use_predicted_task_ids: bool = True,
        normalize: bool = True,
        use_power_norm: bool = True,
        shift: bool = True,
        calibration: bool = True,
    ):
        cfg = self.cfg
        logging.info(f"Evaluating on the test set after domain {cfg.current_domain_id + 1}/{cfg.total_sessions} ...")
        
        labels_predicted_all = torch.tensor([], dtype=torch.long, device=self.device)
        labels_predicted_all_with_oracle = torch.tensor([], dtype=torch.long, device=self.device)
        labels_test_all = torch.tensor([], dtype=torch.long, device=self.device)
        
        domain_ids_predicted_all = torch.tensor([], dtype=torch.long, device=self.device)
        domain_ids_oracle_all = torch.tensor([], dtype=torch.long, device=self.device)
        
        shared_arguments = dict(
            normalize=normalize,
            use_power_norm=use_power_norm,
            shift=shift
        )
        
        domain_id_to_prototypes_text_dict = self.obtain_final_text_prototypes_for_various_domains(
            domain_ids=torch.arange(cfg.current_domain_id + 1, device=self.device),
            **shared_arguments
        )
        
        num_batches = len(self.test_loader)
        ert = ou.EstimatedRemainingTime(total_tasks=num_batches)
        
        for iter_num, batch in enumerate(self.test_loader):
            _, [images_not_aug], labels_test = batch
            
            domain_ids_predicted: T = self._predict_task_ids(images_not_aug)
            
            domain_ids_predicted_all = torch.cat([domain_ids_predicted_all, domain_ids_predicted], dim=0)
            
            domain_id_oracle = self.label_to_domain_id[labels_test]
            domain_ids_oracle_all = torch.cat([domain_ids_oracle_all, domain_id_oracle], dim=0)
            
            # Without Oracle
            
            embeddings_vision_predicted = self.forward_vision(
                images=images_not_aug,
                domain_ids=domain_ids_predicted if use_predicted_task_ids else None,
                calibration=calibration,
                **shared_arguments
            )
            
            if normalize:
                embeddings_vision_predicted = F.normalize(embeddings_vision_predicted, dim=-1)
                
            if use_predicted_task_ids:
                labels_predicted_current_batch = torch.zeros(len(images_not_aug), dtype=torch.long, device=self.device) - 1
                    
                for domain_id in domain_id_to_prototypes_text_dict.keys():
                    mask = domain_ids_predicted == domain_id
                    
                    logits_fused_chosen_domain = self.classifiers.forward_and_bimodal_calibration_with_final_prototypes(
                        embeddings_vision=embeddings_vision_predicted[mask],
                        prototypes_text=domain_id_to_prototypes_text_dict[domain_id],
                        coef_inter_modal_calibration=self.calibration_coefficients.coef_inter_modal_calibration[domain_id],
                        coef_visual_prototypes_calibration=self.calibration_coefficients.coef_visual_prototypes_calibration[domain_id],
                        calibrate_vision_prototypes=cfg.calibrate_vision_prototypes,
                        domain_id=domain_id
                    )
                    
                    labels_predicted_current_batch[mask] = logits_fused_chosen_domain.argmax(dim=-1) + (domain_id * self.num_classes)
            else:
                prototypes_final, prototypes_final_labels = self.classifiers.obtain_all_final_prototypes()
                    
                logits_current_batch = embeddings_vision_predicted @ prototypes_final.T
                
                labels_predicted = logits_current_batch.argmax(-1)
                
                labels_predicted_current_batch = prototypes_final_labels[labels_predicted]
                
            labels_predicted_all = torch.cat([labels_predicted_all, labels_predicted_current_batch], dim=0)
                
            labels_test_all = torch.cat([labels_test_all, labels_test.to(self.device)], dim=0)
            
            # With Oracle (We reveal the domain IDs for the Oracle)
            embeddings_vision_oracle = self.forward_vision(
                images=images_not_aug,
                domain_ids=domain_id_oracle,
                calibration=calibration,
                **shared_arguments
            )
            
            if normalize:
                embeddings_vision_oracle = F.normalize(embeddings_vision_oracle, dim=-1)
            
            labels_predicted_current_batch = torch.zeros(len(images_not_aug), dtype=torch.long, device=self.device) - 1
                
            for domain_id in domain_id_to_prototypes_text_dict.keys():
                mask = domain_id_oracle == domain_id
                
                logits_fused_chosen_domain = self.classifiers.forward_and_bimodal_calibration_with_final_prototypes(
                    embeddings_vision=embeddings_vision_oracle[mask],
                    prototypes_text=domain_id_to_prototypes_text_dict[domain_id],
                    coef_inter_modal_calibration=self.calibration_coefficients.coef_inter_modal_calibration[domain_id],
                    coef_visual_prototypes_calibration=self.calibration_coefficients.coef_visual_prototypes_calibration[domain_id],
                    calibrate_vision_prototypes=cfg.calibrate_vision_prototypes,
                    domain_id=domain_id
                )
                
                labels_predicted_current_batch[mask] = logits_fused_chosen_domain.argmax(dim=-1) + (domain_id * self.num_classes)
                
            labels_predicted_all_with_oracle = torch.cat([labels_predicted_all_with_oracle, labels_predicted_current_batch], dim=-1)
            
            if iter_num % 10 == 9:
                _, acc_temp = self.calculate_accuracy_per_domain(labels_predicted_all, labels_ground_truth=labels_test_all)
                
                report_temp = f"Batch: {iter_num + 1}/{num_batches}, Domain: {cfg.current_domain_id + 1}/{cfg.total_sessions}, AA (w/o Oracle) {acc_temp:.2f}"
                
                _, acc_temp = self.calculate_accuracy_per_domain(labels_predicted_all_with_oracle, labels_ground_truth=labels_test_all)
                
                report_temp += f", AA (with Oracle) {acc_temp:.2f}"
                
                remaining_time_str = ert.calculate(num_finished_tasks=iter_num)
                    
                ou.print_overwrite(f"{report_temp}, {remaining_time_str}")
                
            if cfg.debugging and cfg.current_domain_id == 0 and iter_num > 10:
                break
            
        acc_per_domain_without_oracle, _ = self.calculate_accuracy_per_domain(labels_predicted_all, labels_ground_truth=labels_test_all)
        
        acc_per_domain_with_oracle, _ = self.calculate_accuracy_per_domain(labels_predicted_all_with_oracle, labels_ground_truth=labels_test_all)
        # logging.info(f'Acc. (with Oracle): {acc_with_oracle:.2f}')
    
        acc_domain_id = self.calculate_accuracy(domain_ids_predicted_all, domain_ids_oracle_all)
        
        acc_without_oracle = self.calculate_accuracy(labels_predicted_all, labels_test_all)
        
        acc_with_oracle = self.calculate_accuracy(labels_predicted_all_with_oracle, labels_test_all)
    
        return acc_without_oracle, acc_with_oracle, acc_per_domain_without_oracle, acc_per_domain_with_oracle, acc_domain_id
    
    def obtain_visual_embeddings_and_labels_for_all_samples(
        self,
        domain_id: int,     # When we want to obtain the embeddings for the samples of the current domain, but with the parameters of the previous tasks for calculating the displacement.
        use_train_set: bool,
        use_test_set: bool,
        use_PEFT: bool,
        normalize: bool = True,
        shift: bool = False,
        calibration: bool = True,
        use_power_norm: bool = True,
        use_more_epochs_for_train_set: bool = True
        # shuffling: bool = None
    ):
        # Note: This method have access only to the dataloader from the current domain. By setting the domain_id_for_PEFT_parameters, we can obtain the embeddings with different PEFT parameters (potentially for the prediction of Domain ID).
        cfg = self.cfg
        
        assert use_train_set or use_test_set
        assert self.num_classes > 0
        
        def obtain_embeddings_for_all_samples(
            cfg: cg.Configuration,
            domain_id: int,
            train_or_test: bool,
            use_PEFT: bool,
            normalize: bool,
            use_power_norm: bool,
            calibration: bool,
            shift: bool
            # shuffling: bool
        ):
            cfg = self.cfg
            
            embeddings_all = torch.tensor([])
            labels_all = torch.tensor([], dtype=torch.long)
            
            # if flag_recompute:
            embeddings_all = torch.tensor([], device='cpu')
            labels_all = torch.tensor([], device='cpu', dtype=torch.long)
                
            num_batches = len(self.train_loader) if train_or_test else len(self.test_loader)
            ert = ou.EstimatedRemainingTime(total_tasks=num_batches)
            
            for iter_num, batch in enumerate(self.train_loader if train_or_test else self.test_loader):
                
                if train_or_test:
                    _, [images_aug, images_not_aug], labels = batch
                else:
                    _, [images_not_aug], labels = batch
                
                other_arguments = dict(
                    normalize=normalize,
                    use_power_norm=use_power_norm,
                    shift=shift,
                    calibration=calibration
                )
                
                if train_or_test and use_more_epochs_for_train_set:
                    embeddings = self.forward_vision(
                        images=images_aug,
                        domain_ids=domain_id,
                        **other_arguments
                    )
                    
                    embeddings_all = torch.cat([embeddings_all, embeddings.cpu()], dim=0)
                    labels_all = torch.cat([labels_all, labels], dim=0)
                
                if use_PEFT:
                    embeddings = self.forward_vision(
                        images=images_not_aug,
                        domain_ids=domain_id,
                        **other_arguments
                    )
                else:
                    embeddings = self.forward_vision_frozen_backbone(
                        images=images_not_aug,
                        domain_ids=domain_id,
                        **other_arguments
                    )
                    
                embeddings_all = torch.cat([embeddings_all, embeddings.cpu()], dim=0)
                labels_all = torch.cat([labels_all, labels], dim=0)
                
                remaining_time_str = ert.calculate(num_finished_tasks=iter_num)
                
                if num_batches > 50:
                    ou.print_overwrite(f"Statistics -> Batch: {iter_num + 1}/{num_batches}, {remaining_time_str}")
                    
                if cfg.debugging and domain_id == 0 and iter_num > 1000:
                    break
            
            assert embeddings_all.dim() == 2
            assert labels_all.dim() == 1
            
            if cfg.ignore_domain:
                labels_all = labels_all % self.num_classes
            
            return embeddings_all.to(self.device), labels_all
        
        embeddings_all_final = torch.tensor([])
        labels_all_final = torch.tensor([], dtype=torch.long)
        
        other_arguments = dict(
            normalize=normalize,
            use_power_norm=use_power_norm,
            shift=shift,
            calibration=calibration
        )
        
        if use_train_set:
            embeddings, labels = obtain_embeddings_for_all_samples(
                cfg=cfg,
                domain_id=domain_id,
                train_or_test=True,
                use_PEFT=use_PEFT,
                **other_arguments
            )
            
            embeddings_all_final = torch.cat([embeddings_all_final, embeddings.cpu()], dim=0)
            labels_all_final = torch.cat([labels_all_final, labels], dim=0)
            
        if use_test_set:
            embeddings, labels = obtain_embeddings_for_all_samples(
                cfg=cfg,
                domain_id=domain_id,
                train_or_test=False,
                use_PEFT=use_PEFT,
                **other_arguments
            )
            
            embeddings_all_final = torch.cat([embeddings_all_final, embeddings], dim=0)
            labels_all_final = torch.cat(tensors=[labels_all_final, labels], dim=0)
           
        return embeddings_all_final, labels_all_final
    
    def _obtain_PEFT_parameters(
        self,
        modality: str,
        domain_id: int
    ):
        cfg = self.cfg
        
        coalescent_projections_dict: dict = None
        LoRAs_dict: dict = None
        prompts_dict: dict = None
        
        if cfg.parameter_efficient_method == nt.PEFT_Type.CoalescentProjection:
            coalescent_projections_dict = self.coalescent_projections.obtain_CPs_of_a_domain(modality=modality, domain_id=domain_id)
        elif cfg.parameter_efficient_method == nt.PEFT_Type.LoRA:
            LoRAs_dict = self.LoRAs_for_CLIP.obtain_LoRAs_of_a_domain(modality=modality, domain_id=domain_id)
        elif cfg.parameter_efficient_method == nt.PEFT_Type.Prompt:
            prompts_dict = self.prompts.obtain_prompts_of_a_domain(modality=modality, domain_id=domain_id)
        else:
            raise NotImplementedError()
        
        return coalescent_projections_dict, LoRAs_dict, prompts_dict
    
    @torch.no_grad()
    def _predict_task_ids(
        self,
        images: T,
        normalize: bool = True,
        use_power_norm: bool = True,
        shift: bool = False,
        calibration: bool = False,
    ) -> T:
        """
        arguments:
            embeddings: [bs, embed_dim]
        return:
            indices: [bs], indices of selected prompts
        """
        cfg = self.cfg
        
        if cfg.current_domain_id == 0:          # When the model observed only the first domain.
            return torch.zeros(images.shape[0], dtype=torch.long, device=self.device)
        
        scores_over_tasks = []
        
        for domain_id in range(cfg.current_domain_id + 1):
            means_over_classes, covariance_inverses, labels = self.statistics_from_frozen_backbone.obtain_statistics(domain_id=domain_id, separated_or_shared_covariances=cfg.separate_or_shared_covariance_for_domain_id)
            
            other_arguments = dict(
                normalize=normalize,
                use_power_norm=use_power_norm,
                shift=shift,
                calibration=calibration,
            )
            
            embeddings = self.forward_vision(
                images=images,
                domain_ids=None,
                **other_arguments
            )

            num_labels, _ = means_over_classes.shape
            
            distances_over_classes_list = []
            for c in range(num_labels):
                score = mahalanobis(embeddings, means_over_classes[c], covariance_inverses[c], norm=2)
                distances_over_classes_list.append(score)
            # [num_labels, n]
            distances_over_classes = torch.stack(distances_over_classes_list)
            score, _ = distances_over_classes.min(dim=0)

            scores_over_tasks.append(score)
        # [task_num, n]
        scores_over_tasks = torch.stack(scores_over_tasks, dim=0)
        
        _, indices = torch.min(scores_over_tasks, dim=0)
        
        return indices
    
    @torch.no_grad()
    def calculate_means_and_covariances(
        self,
        embeddings_all: T,      # Its shape: [num_samples, dim_embed]
        labels_all: T,
        calculate_covariance: bool = True,
        coef_regularization: float = 1e-3
    ):
        labels_unique = labels_all.unique()

        means_for_each_class = torch.tensor([], device=embeddings_all.device)
        cov_over_classes = torch.tensor([], device=embeddings_all.device)
        
        for c in labels_unique.tolist():
            embeds = embeddings_all[labels_all == c]
            mean = embeds.mean(dim=0, keepdim=True)
            means_for_each_class = torch.cat([means_for_each_class, mean], dim=0)
            
            if calculate_covariance:
                covariance = torch.cov(embeds.T) + coef_regularization * torch.eye(self.dim_embed)        # Its shape: [dim_embed, dim_embed]
                cov_over_classes = torch.cat([cov_over_classes, covariance.unsqueeze(0)], dim=0)

        if calculate_covariance:
            # cov_over_classes.shape: [num_classes, dim_embed, dim_embed]
            covariance = cov_over_classes.mean(dim=0)
            return means_for_each_class, covariance, labels_unique
        else:
            return means_for_each_class, None, labels_unique
            
    def import_and_verify_descriptions(self):
        cfg = self.cfg
        
        if cfg.prototype_calculation_mode_text in [nt.PrototypeTextModality.WeightedSynonyms, nt.PrototypeTextModality.WeightedClassNamesAndSynonyms] and os.path.exists(cfg.synonyms_json_file_path):
            with open(cfg.synonyms_json_file_path, 'r') as file:
                self.synonyms_dict = json.load(file)
                
            # We automatically infer the number of synonyms for each class name.
            num_synonyms_old = -1
            
            for name in self.class_names:
                if name not in self.synonyms_dict:
                    raise Exception(f'The synonyms of the class "{name}" are not defined!')
                
                num_syn = len(self.synonyms_dict[name])
                if num_synonyms_old == -1:      # For the first item
                    num_synonyms_old = num_syn
                
            self.num_synonyms = num_synonyms_old
    
    def load_the_backbone(self):
        cfg = self.cfg
        
        if cfg.backbone_type == nt.BackboneType.CLIP_ViT_B16:
            self.model, self.preprocess = clip.load("ViT-B/16", device=cfg.device, num_vision_prompts=cfg.num_vision_prompts, num_text_prompts=cfg.num_text_prompts, use_checkpinting=cfg.use_checkpinting)
            logging.info("ViT-B/16 is loaded")
        elif cfg.backbone_type == nt.BackboneType.CLIP_ViT_L14:
            self.model, self.preprocess = clip.load("ViT-L/14", device=cfg.device, use_checkpinting=cfg.use_checkpinting)
            logging.info("ViT-B/16 is loaded")
        else:
            raise NotImplementedError("Error: This model isn't supported yet!")
        
        self.model = self.model.float()
        self.model = self.model.eval()
        self.model = self.model.to(self.device)
        self.dim_embed = self.model.ln_final.weight.shape[0]
        self.num_heads_vision = self.model.vision_heads
        self.num_heads_text = self.model.transformer_heads
        self.dim_embed_vision = self.model.dim_embed_vision
        self.dim_embed_text = self.model.dim_embed_text
        
        ou.freeze_or_unfreeze(self.model, requires_grad=False)
        
    def _prepare_data_manager(self):
        if self.data_manager is not None:
            return
        
        cfg = self.cfg
        
        data_manager = DataManager(
            cfg=cfg,
            shuffle_class_order=False,
            seed=cfg.seed_current
        )
    
        self.data_manager = data_manager
        self.num_classes = data_manager.num_classes
        self.class_names = data_manager.class_names
        self.domain_names_for_this_order = data_manager.domain_names_for_this_order
        
        logging.info(f'Domains: {self.domain_names_for_this_order}')
        pass
    
    def _prepare_dataloaders(self):
        cfg = self.cfg
        
        args_common_data_loader = dict(num_workers=cfg.num_workers, persistent_workers=True)

        # These are for the calibration phase
        if cfg.current_domain_id == 0:
            # We use all samples in the first (base) domain.
            num_shots_current_domain = -1
            drop_last_train = True
            batch_size_train = cfg.batch_size
        else:
            num_shots_current_domain = self.num_shots
            drop_last_train = False
            batch_size_train = cfg.batch_size

        train_dataset = self.data_manager.get_dataset(np.arange(self._known_classes, self._total_classes), source="train", aug_modes_list=["train", "test"], num_shots=num_shots_current_domain)
        self.train_loader = DataLoader(train_dataset, batch_size=batch_size_train, drop_last=drop_last_train, shuffle=True, **args_common_data_loader)

        # We use all samples from the test set.
        test_dataset = self.data_manager.get_dataset(np.arange(0, self._total_classes), source="test", aug_modes_list=["test"], num_shots=-1)
        self.test_loader = DataLoader(test_dataset, batch_size=cfg.batch_size, drop_last=False, shuffle=True, **args_common_data_loader)
        
def mahalanobis(
    embeddings: T,
    mean: T,
    cov_inv: T,
    norm=2
):
    """
    args:
        embeddings: [num_samples, dim_embed]
        mean: [dim_embed]
        cov_inv: [dim_embed, dim_embed]
    return:
        [num_samples]
    """
    
    assert embeddings.dim() == 2
    assert mean.dim() == 1
    assert cov_inv.dim() == 2
    
    diff = embeddings - mean
    maha_dis = (diff @ cov_inv) * diff

    if norm == 2:
        return maha_dis.sum(dim=1)
    elif norm == 1:
        return maha_dis.abs().sqrt().sum(dim=1)
    elif norm == 'inf':
        return maha_dis.max(dim=1)
    else:
        raise NotImplementedError()
