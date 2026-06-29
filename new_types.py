from enum import Enum
    

class BackboneType(Enum):
    ViT_B16_ImageNet1K = 1
    ViT_B16_ImageNet21K = 2     # pretrained_vit_b16_224_in21k
    ViT_B16_LoRA = 3            # vit_base_patch16_224_conec_lora
    CLIP_ViT_B16 = 4            # ViT-B/16
    CLIP_ViT_L14 = 5            # ViT-L/14


class PrototypeTextModality(Enum):
    Templates = 1
    WeightedSynonyms = 2
    MultipleDescriptions = 3
    WeightedClassNamesAndSynonyms = 4
    Intra_modal_calibration = 5


class DistanceFunction(Enum):
    Euclidean = 1
    CosineDistance = 2
    CosineSimilarity = 3
    L1 = 4
        
    
class InitializationType(Enum):     # For the Coalescent Projection
    Normal = 1
    Zeros = 2
    Uniform = 3
    Kaiming = 4
    CloseToDiagonalMatrix = 5
    Diagonal = 6
    
class ClassifierType(Enum):
    Linear = 1
    Cosine = 2
    Stochastic = 3
    

class LSR_Distributions_Domain(Enum):
    Base = 1
    Current = 2
    
class LSR_GeneratedClassesLabels(Enum):
    NewLabels = 1
    InterpolatedLogits = 2
    
    
class InitializationApproachForIncrementalTasks(Enum):
    Reinitialize = 1
    CopyFromPreviousDomain = 2
    CopyFromFirstDomain = 3
    MeanOfEncounteredDomains = 4


class PEFT_Type(Enum):
    CoalescentProjection = 1
    LoRA = 2
    Prompt = 3
