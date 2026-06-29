import os
from pathlib import Path
# import sys
import our_utils as ou
import logging
import new_types as nt
import torchvision
import json
import platform
from enum import Enum
import sys
import torch
from typing import Any, Union, List, Generator, Tuple, Dict


class Configuration:
    def __init__(
        self,
        args,
        dir_script: str,
        current_working_directory: Path
    ):
        
        self.args = args
        self.dir_script = dir_script
        self.current_working_directory = current_working_directory
        self.config_file = args.config_file
        
        self.current_domain_id: int = -1
        
        self.prefix: str = 'CVLC'
        
        self.model = None
        
        self.printable_types_of_attributes = {int, float, str, list, bool, tuple, dict, torch.device, Path}
        
        self.seeds_list = [0]
        self.seed_current: int = 0
        
        self.gpu_id: int = 0
        self.device = 'cuda:0'
        
        # Defulat values
        self.logdir: str = 'logs'
        self.save_last_model: bool = False
        self.save_model_after_each_task: bool = False
        self.load_path: str = ''
        
        self.dataset_name: str = ''
        self.shuffle: bool = True
        self.batch_size: int = -1
        self.data_path: str = ''
        self.num_workers = 8
        
        # FSDIL settings
        self.order: int = 1         # Order of the domains
        self.num_shots: int = 5     # Few-Shot
        self.init_cls: int = -1
        self.increment: int = -1
        self.class_order = []
        self.total_sessions = -1
        self.n_components: int = 2
        self.num_epochs_list: list[int] = []
        self.min_shots_test_set: int = 1        # If we have less than this number of samples in one of the domains, we ignore those classes.
        
        self.core50_split_seed: int = 1993
        self.core50_test_fraction = 0.2
        
        self.classifier_type: nt.ClassifierType = nt.ClassifierType.Cosine
        
        self.d_model: int = -1
        self.embd_dim = 768
        
        self.use_temporary_classifier: bool = True
        
        # Optimizer
        self.optimizer_name = ['AdamW', 'AdamW', 'AdamW', 'AdamW', 'AdamW']
        self.min_lr: float = 1e-8
        self.lr_default: List[float] = [0.0] * 5
        self.lr_calibration_coefficients: List[float] = [0.0] * 5
        self.lr_power_norm: List[float] = [0.0] * 5
        self.lr_PEFT_vision_shared: List[float] = [0.0] * 5
        self.lr_PEFT_vision_specific: List[float] = [0.0] * 5
        self.lr_PEFT_text_shared: List[float] = [0.0] * 5
        self.lr_PEFT_text_specific: List[float] = [0.0] * 5
        self.lr_classifier: List[float] = [0.0] * 5
        self.weight_decay: List[float] = [0.0] * 5
        self.weight_decay_classifiers: List[float] = [0.0] * 5
        self.lr_embeddings_biases: List[float] = [0.0] * 5
        
        self.parameter_efficient_method = nt.PEFT_Type.CoalescentProjection
        
        # Coalescent Projection
        self.shared_CPs_shared_across_heads: bool = True
        self.specific_CPs_shared_across_heads: bool = True
        self.std_for_CPs = 0.02
        self.task_shared_layers: list[int] = [0, 1, 2, 3, 4, 5]
        self.task_specific_layers: list[int] = [6, 7, 8, 9, 10, 11]
        self.enable_vision_CPs: bool = True
        self.enable_text_CPs: bool = True
        
        # LoRA
        self.enable_vision_LoRAs: bool = False
        self.enable_text_LoRAs: bool = False

        self.LoRA_rank_vision: int = 2
        self.LoRA_rank_text: int = 2
        self.LoRA_QKV_mask: list = [True, False, True]
        
        # Prompts
        self.enable_vision_prompts: bool = False
        self.enable_text_prompts: bool = False
        self.num_vision_prompts: int = 0
        self.num_text_prompts: int = 0
        
        # Early stopping
        self.MA_loss_capacity: int = 10
        self.MA_loss_threshold_for_early_stopping: float = 0.0       # We only set it for large dataset
        
        # Initial value for the calibration coefficients, if we use them as hyperparameters with lr=0
        self.coef_synonyms_prototypes_init_value: float = 0.5
        self.coef_visual_prototypes_calibration_init_value: float = 0.5
        self.coef_inter_modal_calibration_init_value: float = 0.5
        self.coef_shift_text_from_current_domain_in_vision_modality_init_value: float = 0.5
        self.coef_shift_text_from_first_domain_in_vision_modality_init_value: float = 0.5
        self.coef_shift_text_init_value: float = 0.5
        self.coef_shift_vision_from_first_domain_init_value: float = 0.5
        self.coef_shift_vision_from_current_domain_init_value: float = 0.5
        self.power_norm_alpha_vision_init_value: float = 1.0
        self.power_norm_alpha_text_init_value: float = 1.0
        
        self.calibrate_vision_prototypes: bool = False
        
        self.peft_for_new_domain: nt.InitializationApproachForIncrementalTasks = nt.InitializationApproachForIncrementalTasks.CopyFromPreviousDomain
        
        self.use_transformation_module: bool = True
        
        self.bias_domain_classifier: bool = True
        self.bias_domain_classifier: bool = True
        self.margin: float = 1.0
        
        self.debugging: bool = False
        self.disable_caching_mechanism: bool = False
        self.dir_cache = 'cache'
        os.makedirs(self.dir_cache, exist_ok=True)
        
        self.chosen_layers_for_intermediate_domain_classifiers: list[str] = ['final']
        self.confidence_threshold: list[float] = [0.5]
        
        self.init_temperature_cosine_classifier: float = 0.07
        self.init_temperature_stochastic_classifier: float = 16.0
        
        self.minimum_num_samples_required_for_statistics_in_incremental_tasks: int = 100
        self.minimum_num_samples_per_class_required_for_domain_id_prediction = 10
        
        self.separate_or_shared_covariance_for_domain_id: bool = False
        
        # Latent-space reservation:
        self.use_LSR: List[bool] = None
        self.LSR_generated_classes_labels = nt.LSR_GeneratedClassesLabels.NewLabels
        self.LSR_num_candidates: int = 100
        self.LSR_num_ways_after_filtering_1 = 8
        self.LSR_num_ways_after_filtering_2 = 4
        self.LSR_beta: float = 1.0
        self.LSR_distributions_domain = nt.LSR_Distributions_Domain.Current
        self.LSR_separate_covariances: bool = True
        self.num_pseudo_embeddings: int = 10
        
        self.backbone_type: nt.BackboneType = nt.BackboneType.CLIP_ViT_B16
        
        self.batch_size_limit_for_incremental_tasks: int = 1000000        # For DomainNet experiments, we need to limit the memory consumption.
        self.chunk_size_vision: int = 64
        self.chunk_size_text: int = 64
        
        # Prototype displacement correction
        self.use_prototype_correction: bool = True
        
        # When a dataset requires too much memory, we calculate the statistics for a subset of samples.
        self.cache_synthetic_embeddings: bool = True
        # For faster testing with sufficient embeddings for statistics without affecting the evaluation.
        self.max_number_of_embeddings_for_statistics = 1e8
        
        self.UMAP: bool = False
        
        self.date_time_str = ou.get_time_str()
        self.date_str = ou.get_time_str(add_time=False)
        
        self.logs_directory_name: str = ''
        self.log_file_name: str = ''
        
        # CLIP
        self.CLIP_templates: list[str] = [
            "{class_name} ..",
        ]
        self.synonyms_json_file_path: str = ''
        self.num_synonyms_limit = 10
        self.max_num_descriptions: int = 5
        self.prototype_calculation_mode_text: nt.PrototypeTextModality = nt.PrototypeTextModality.Templates
        
        self.use_cache: bool = True
        
        self.chosen_layer_vision = -1
        
        self.ignore_domain: bool = False    # If we want always see num_classes classes in all domains. From prototypes we can roughly guess the domain.
        
        self.experiment_description: str = ""
        
        self.use_checkpinting: bool = False
        
        self.load_settings_from_json_file_and_program_arguments()
        self.prepare_logger()
        self.verify_setting()
        self.set_remaining_variables_automatically()
        
        self.print_arguments()

    def prepare_logger(self):
        self.logs_directory_name = os.path.join(self.logdir, self.date_str)

        os.makedirs(self.logs_directory_name, exist_ok=True)

        self.log_file_name = f"{self.logs_directory_name}/DB={self.dataset_name},{self.prefix},Order={self.order},n_shots={self.num_shots},Seed={self.seed_current},{self.parameter_efficient_method},Time={self.date_time_str}"
        
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s [%(filename)s:%(lineno)d] => %(message)s",
            handlers=[
                logging.FileHandler(filename=self.log_file_name + ".log"),
                logging.StreamHandler(sys.stdout),
            ],
        )
        
        # Global exception handler
        def global_exception_handler(exc_type, exc_value, exc_traceback):
            # Ignore KeyboardInterrupt so a console python program can exit with Ctrl + C
            if issubclass(exc_type, KeyboardInterrupt):
                sys.__excepthook__(exc_type, exc_value, exc_traceback)
                return

            # Log the error and the traceback
            logging.error("Uncaught exception", exc_info=(exc_type, exc_value, exc_traceback))

        # Assign your custom handler to sys.excepthook
        sys.excepthook = global_exception_handler
    
    def print_arguments(self):
        
        lines = self.obtain_arguments_as_strings()
        
        for line in lines:
            logging.info(line)
        
    def obtain_arguments_as_strings(self):
        lines = []
        separator = '-' * 80
        
        # GPU
        lines.append(ou.highlighted_message('Software and hardware details'))
        lines.append(f"GPU ID: {self.gpu_id}")
        lines.append(f'Python version: {platform.python_version()}')
        lines.append(f'PyTorch version: {torch.__version__},TorchVision version: {torchvision.__version__}')
        lines.append(f"CUDA availability: {torch.cuda.is_available()}")
        
        if torch.cuda.is_available():
            lines.append(f"CUDA version: {torch.version.cuda}")
            lines.append(f"GPU Name: {torch.cuda.get_device_name(0)}")
            lines.append(f"Available GPU memory: {torch.cuda.get_device_properties(0).total_memory / (1024**3):.3f} GB")
            lines.append(f"cuDNN version: {torch.backends.cudnn.version()}")
            lines.append(f"cuDNN enabled: {torch.backends.cudnn.enabled}")
            # lines.append(f"Driver: {current_gpu.driver}")
            lines.append(f"Driver: {ou.obtain_driver_version()}")
        else:
            exit(0)
        lines.append(separator)
        
        attributes = self.__dict__
        ignore_set = {'args', 'log_file', 'split_to_tar_file'}

        lines.append(ou.highlighted_message("The given arguments"))
        
        wait_list = ['device']
        
        def print_name_and_values(name, value):
            if name == 'device':
                lines.append(f"{name} = \"{str(value)}\"")
            elif type_attr is not str:
                lines.append("%s = %s" % (name, str(value)))
            else:
                lines.append('%s = "%s"' % (name, value))

        for name in attributes.keys():
            value = getattr(self, name)
            type_attr = type(value)
            if name.startswith("_") or name in ignore_set or name in wait_list or type_attr not in self.printable_types_of_attributes and not isinstance(value, Enum):        # or value is None
                continue
            
            print_name_and_values(name, value)

        # We show these settings at last.
        for name in wait_list:
            value = getattr(self, name)
            print_name_and_values(name, value)
            
        lines.append(separator)
        
        return lines
    
    def load_settings_from_json_file_and_program_arguments(self):
        with open(self.config_file, 'r') as file:
            data_json = json.load(file)
            
        # Detection and conversion of enum classes
        for k, v in data_json.items():
            value_final = v
            
            if isinstance(v, str):
                if '.' in v and len(v.split('.')) == 2:    # Enums
                    class_name, enum_value_str = v.split('.')
                    # enum_class = globals().get(class_name)
                    if hasattr(nt, class_name):
                        enum_class = getattr(nt, class_name)
                    
                        if enum_class is not None:
                            value_final = enum_class[enum_value_str]
            
            setattr(self, k, value_final)
            
        # Set the values from the args
        for k, v in self.args._get_kwargs():
            if v is None:
                continue
            setattr(self, k, v)
            
        # We must have these keys in the JSON configuration files.
        required_settings_in_JSON_files = []
        
        required_settings_in_JSON_files = ['task_shared_layers', 'task_specific_layers', 'lr_default']
        
        for key in required_settings_in_JSON_files:
            if key not in data_json:
                raise Exception(f'Error: You must set the "{key}" in the "{self.config_file}" file.')
                
    def set_remaining_variables_automatically(self):
        self.task_shared_layers = sorted(self.task_shared_layers)
        self.task_specific_layers = sorted(self.task_specific_layers)
        
        self.gpu_count = torch.cuda.device_count()
        if self.gpu_count == 1:
            self.gpu_id = 0
        
        self.device = torch.device(f"cuda:{self.gpu_id}")
        
        if not self.enable_vision_prompts:
            self.num_vision_prompts = 0
            
        if not self.enable_text_prompts:
            self.num_text_prompts = 0
        
    def verify_setting(self):
        assert self.init_cls == self.increment      # We may consider other cases in the future.
        
        if not ou.are_consecutive(self.task_shared_layers) or not ou.are_consecutive(self.task_specific_layers):
            raise NotImplementedError
        
        assert self.order in [1, 2, 3, 4, 5] or (self.dataset_name in ['domainnet'] and self.order == 6) or (self.dataset_name in ('core50') and self.order in [0, 1, 2, 3, 4, 5])
        assert self.dataset_name in ['cddb', 'officehome', 'domainnet', 'core50']
        
        assert set(self.task_shared_layers) & set(self.task_specific_layers) == set()
        
        # We ensure that all the layers are covered by the CPs.
        assert len(set(self.task_shared_layers) | set(self.task_specific_layers)) == 12
        
        assert self.num_shots >= 0
        
        assert len(self.num_epochs_list) == self.total_sessions
        
        if not self.LSR_separate_covariances:
            raise NotImplementedError()
        
        assert self.total_sessions == len(self.use_LSR)
        
        assert self.total_sessions == len(self.lr_default)
        assert self.total_sessions == len(self.lr_calibration_coefficients)
        assert self.total_sessions == len(self.lr_power_norm)
        assert self.total_sessions == len(self.lr_PEFT_vision_shared)
        assert self.total_sessions == len(self.lr_PEFT_vision_specific)
        assert self.total_sessions == len(self.lr_PEFT_text_shared)
        assert self.total_sessions == len(self.lr_PEFT_text_specific)
        assert self.total_sessions == len(self.lr_classifier)
        assert self.total_sessions == len(self.lr_embeddings_biases)
        assert self.total_sessions == len(self.weight_decay)
        assert self.total_sessions == len(self.weight_decay_classifiers)
        assert self.total_sessions == len(self.optimizer_name)
        
        if self.parameter_efficient_method == nt.PEFT_Type.LoRA:
            assert self.enable_vision_LoRAs or self.enable_text_LoRAs
            
        if self.parameter_efficient_method == nt.PEFT_Type.CoalescentProjection:
            assert self.enable_vision_CPs or self.enable_text_CPs
            
        if self.parameter_efficient_method == nt.PEFT_Type.Prompt:
            assert self.enable_vision_prompts or self.enable_text_prompts
        
    def save(self, file_path: str):
        # We save args to know what was the setting in the past.
        state = {}
        state['model_state'] = self.model._network.state_dict()
        
        state['current_domain_id'] = self.current_domain_id
        
        torch.save(state, file_path)
        logging.info(f'The parameters are saved after task: {self.current_domain_id}!')
        
    def load(self, file_path: str):
        state_dict = torch.load(file_path)
        
        last_finished_task_id = state_dict['current_domain_id']
        
        state_dict.pop('current_domain_id')
        
        self.model._network.load_state_dict(state_dict.model_state)
        
        state_dict.pop('model_state')
        
        logging.info(f'The parameters are loaded for task: {last_finished_task_id}!')
        return last_finished_task_id
        
