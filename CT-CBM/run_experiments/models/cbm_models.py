import os
import torch
import torch.nn as nn
from torch.nn import Parameter
import torch.nn.functional as F
import torch.utils.model_zoo as model_zoo

from cbm_template_models import MLP,FC,ModelXtoC,End2EndModel

# Joint Model
def ModelXtoCtoY_function(concept_classes, label_classes, n_attributes, bottleneck, expand_dim,n_class_attr, use_relu, use_sigmoid,connect_CY=False,Lstm=False,aux_logits=False, config = 728):
    dim = config.dim
    ModelXtoC_layer = ModelXtoC(num_classes = concept_classes, n_attributes=n_attributes, bottleneck=bottleneck, expand_dim=expand_dim,connect_CY=connect_CY,Lstm=Lstm,aux_logits=aux_logits, dim=dim)
    if n_class_attr !=0:
        ModelCtoY_layer = MLP(input_dim=n_attributes * n_class_attr, num_classes=label_classes, expand_dim=expand_dim)
    else:
        ModelCtoY_layer = MLP(input_dim=n_attributes, num_classes=label_classes, expand_dim=expand_dim)
    return End2EndModel(
        model1 = ModelXtoC_layer, 
        model2 = ModelCtoY_layer, 
        use_relu = use_relu, 
        use_sigmoid = use_sigmoid, 
        n_attributes = n_attributes,
        n_class_attr = n_class_attr
        )

