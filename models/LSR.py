import torch
from torch import nn
from torch import Tensor as T
from torch.nn import functional as F
import new_types as nt
import our_utils as ou
import configs as cg
import random
from torch.distributions import Beta
from typing import Any, NoReturn, Union, List, Tuple, Generator
from models.other_components import Statistics


class LSR:
    def __init__(
        self,
        cfg: cg.Configuration,
        num_classes: int,
        dim_embed: int
        ):
        self.cfg = cfg
        self.num_candidates = cfg.LSR_num_candidates
        self.num_ways_after_filtering_1 = cfg.LSR_num_ways_after_filtering_1
        self.num_ways_after_filtering_2 = cfg.LSR_num_ways_after_filtering_2
        self.num_pseudo_embeddings = cfg.num_pseudo_embeddings
        self.separate_covariances = cfg.LSR_separate_covariances
        self.distributions_domain = cfg.LSR_distributions_domain
        self.generated_classes_labels = cfg.LSR_generated_classes_labels
        
        self.num_classes = num_classes
        self.dim_embed = dim_embed
        
        self.device = cfg.device
        self.beta_dist = Beta(1.0, 1.0)
        
    def obtain_pseudo_embedding_generation_iterator(
        self,
        current_domain_id: int,
        statistics: Statistics
        ) -> Generator[Tuple[T, T]]:
        while True:
            embeddings_generated, labels_generated_one_hot = self._generate_pseudo_embeddings(
                current_domain_id=current_domain_id,
                statistics=statistics
                )
            
            yield embeddings_generated, labels_generated_one_hot
    
    def _generate_pseudo_embeddings(
        self,
        current_domain_id: int,
        statistics: Statistics
    ) -> Tuple[T, T]:
        # means.shape = [self.LSR_num_candidates, dim_embed]
        # covariances.shape = [self.LSR_num_candidates, dim_embed, dim_embed]
        means_base, covariances_base = self._obtain_means_and_covariances_to_mix(current_domain_id=current_domain_id, statistics=statistics)
        
        means_candidates, covariances_candidates, class_pairs, coefficients_generated_distributions = self._mix_distributions_to_generate_candidates(
            means_base=means_base,
            covariances_base=covariances_base
        )
        
        # We have now the embeddings candidates.
        means_after_novel_novel_filter, covariances_after_novel_novel_filter, class_pairs_after_novel_novel_filter, coefficients_generated_distributions_after_novel_novel_filter = self._novel_novel_filter(
            means_generated=means_candidates,
            covariances_generated=covariances_candidates,
            coefficients_generated_distributions=coefficients_generated_distributions,
            class_pairs=class_pairs,
            num_indices_to_choose=min(self.num_ways_after_filtering_1, self.num_candidates)
        )
        
        means_after_base_novel_filter, covariances_after_base_novel_filter, class_pairs_after_base_novel_filter, coefficients_generated_distributions_after_base_novel_filter = self._base_novel_filter(
            means_chosen=means_after_novel_novel_filter,
            covariances_chosen=covariances_after_novel_novel_filter,
            coefficients_generated_distributions=coefficients_generated_distributions_after_novel_novel_filter,
            class_pairs=class_pairs_after_novel_novel_filter,
            num_indices_to_choose=self.num_ways_after_filtering_2,
            means_base=means_base,
            covariances_base=covariances_base
        )
        
        assert means_after_base_novel_filter.shape[0] == self.num_ways_after_filtering_2
        
        # We have now the means and covariances to sample from for the new tasks
        # Next, we generate pseudo embeddings and treat them as new embeddings.
        
        embeddings_generated = torch.tensor([], device=self.device)
        
        labels_generated_one_hot = torch.tensor([], dtype=torch.long, device=self.device)
        
        # Labels for the generated new classes (self.num_ways_after_filtering_2 classes):
        if self.generated_classes_labels == nt.LSR_GeneratedClassesLabels.NewLabels:
            num_classes_after_generation = self.num_classes + self.num_ways_after_filtering_2
            labels_for_each_class = torch.arange(self.num_classes, num_classes_after_generation, device=self.device)
            labels_for_each_class_one_hot = F.one_hot(labels_for_each_class, num_classes=num_classes_after_generation).to(self.device)
            
        elif self.generated_classes_labels == nt.LSR_GeneratedClassesLabels.InterpolatedLogits:
            labels_first_classes_of_the_pairs = class_pairs_after_base_novel_filter[:, 0]
            labels_second_classes_of_the_pairs = class_pairs_after_base_novel_filter[:, 1]
            
            labels_one_hot_first_classes_of_pairs = F.one_hot(labels_first_classes_of_the_pairs, num_classes=self.num_classes)
            labels_one_hot_second_classes_of_pairs = F.one_hot(labels_second_classes_of_the_pairs, num_classes=self.num_classes)
            
            coef = coefficients_generated_distributions_after_base_novel_filter
            
            labels_for_each_class_one_hot = coef.unsqueeze(1) * labels_one_hot_first_classes_of_pairs + (1.0 - coef.unsqueeze(1)) * labels_one_hot_second_classes_of_pairs
        else:
            raise NotImplementedError()
        
        # Pseudo-novel episode
        for i in range(self.num_ways_after_filtering_2):
            pseudo_embedding = ou.sample_from_gaussian(mean=means_after_base_novel_filter[i], covariance=covariances_after_base_novel_filter[i], num_samples=self.num_pseudo_embeddings)
            
            embeddings_generated = torch.cat([embeddings_generated, pseudo_embedding.detach().clone()], dim=0)
            
            labels_current_class_one_hot = labels_for_each_class_one_hot[i].repeat(self.num_pseudo_embeddings, 1)
            
            labels_generated_one_hot = torch.cat([labels_generated_one_hot, labels_current_class_one_hot])
            
        embeddings_generated = embeddings_generated.to(self.device)
        labels_generated_one_hot = labels_generated_one_hot.to(self.device)
        
        return embeddings_generated, labels_generated_one_hot
    
    @torch.no_grad()
    def _obtain_means_and_covariances_to_mix(
        self,
        current_domain_id: int,
        statistics: Statistics
    ):
        if self.distributions_domain == nt.LSR_Distributions_Domain.Current:
            domain_id = current_domain_id       # We mix the distributions from the model trained until the last seen domain.
        else:
            domain_id = 0       # We mix the first domain's distributions
            
        means_base, covariances_base, labels = statistics.obtain_statistics(
            domain_id=domain_id,
            separated_or_shared_covariances=self.separate_covariances
        )
        
        return means_base, covariances_base
    
    @torch.no_grad()
    def _mix_distributions_to_generate_candidates(
        self,
        means_base: T,
        covariances_base: T
    ):
        means_generated = torch.tensor([], device=self.device)     # Prototypes
        covariances_generated = torch.tensor([], device=self.device)
        coefficients_generated_distributions = torch.tensor([])
        class_pairs = torch.tensor([], dtype=torch.long)
        
        for _ in range(self.num_candidates):
            # We randomly choose a pair of distributions to mix.
            first_class_id, second_class_id = random.sample(range(self.num_classes), k=2)
            
            mean_first_class = means_base[first_class_id]
            covariance_first_class = covariances_base[first_class_id]
            mean_second_class = means_base[second_class_id]
            covariance_second_class = covariances_base[second_class_id]
            
            # lambda_coef = torch.rand(1).to(configs.device)
            lambda_coef_tensor = self.beta_dist.sample()
            
            lambda_coef = lambda_coef_tensor.item()
            
            means_mixed = lambda_coef * mean_first_class + (1.0 - lambda_coef) * mean_second_class
            covariances_mixed = lambda_coef * covariance_first_class + (1.0 - lambda_coef) * covariance_second_class
            
            means_generated = torch.cat([means_generated, means_mixed.detach().clone().unsqueeze(0)], dim=0)
            covariances_generated = torch.cat([covariances_generated, covariances_mixed.detach().clone().unsqueeze(0)], dim=0)
            coefficients_generated_distributions = torch.cat([coefficients_generated_distributions, lambda_coef_tensor.reshape(1)])
            class_pairs = torch.cat([class_pairs, torch.tensor([first_class_id, second_class_id]).unsqueeze(0)], dim=0)
            
        class_pairs = class_pairs.to(self.device)   # Its shape = [self.num_candidates, 2]
        
        means_generated = means_generated.to(self.device)
        covariances_generated = covariances_generated.to(self.device)
        class_pairs = class_pairs.to(self.device)
        coefficients_generated_distributions = coefficients_generated_distributions.to(self.device)
        
        return means_generated, covariances_generated, class_pairs, coefficients_generated_distributions
    
    @torch.no_grad()
    def _novel_novel_filter(
        self,
        means_generated: T,
        covariances_generated: T,
        class_pairs: T,
        coefficients_generated_distributions: T,
        num_indices_to_choose: int
    ) -> Tuple[T, T, T, T]:
        means_generated = means_generated.to(self.device)
        
        means_generated, covariances_generated = ou.to_device([means_generated, covariances_generated], device=self.device)
        
        # similarity.shape becomes [self.LSR_num_candidates, self.LSR_num_candidates]
        means_generated_normalized = F.normalize(means_generated, -1)
        similarity: T = means_generated_normalized @ means_generated_normalized.T
        # similarity: T = means_generated @ means_generated.T
        similarity.fill_diagonal_(0.0)              # Removes self-similarities
        score = similarity.sum(dim=1)                  # Smaller is better
        
        _, indices = torch.topk(score, num_indices_to_choose, largest=False)   # We find the smallest values.
        
        indices = indices.tolist()
        
        # means_chosen = means_generated[indices]
        means_chosen = means_generated[indices]
        covariances_chosen = covariances_generated[indices]
        coefficients_generated_distributions_chosen = coefficients_generated_distributions[indices]
        class_pairs_chosen = class_pairs[indices]
        
        return means_chosen, covariances_chosen, class_pairs_chosen, coefficients_generated_distributions_chosen
        
    @torch.no_grad()
    def _base_novel_filter(
        self,
        means_chosen: T,
        covariances_chosen: T,
        class_pairs: T,
        coefficients_generated_distributions: T,
        num_indices_to_choose: int,
        means_base: T,
        covariances_base: T,
    ) -> Tuple[T, T, T, T]:
        diversities = ou.kl_divergence_fast(base_means=means_base, base_covs=covariances_base, cand_means=means_chosen, cand_covs=covariances_chosen)
        
        _, indices = torch.topk(diversities, k=num_indices_to_choose, largest=True)
        
        means_chosen = means_chosen[indices]
        covariances_chosen = covariances_chosen[indices]
        coefficients_generated_distributions = coefficients_generated_distributions[indices]
        class_pairs = class_pairs[indices]
        
        return means_chosen, covariances_chosen, class_pairs, coefficients_generated_distributions
