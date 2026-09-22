import os
import torch
import numpy as np 
import torch.nn as nn
from torch.nn import Parameter
import torch.nn.functional as F

class FC(torch.nn.Module):

    def __init__(self, input_dim, output_dim, expand_dim=0, stddev=None):
        """
        Extend standard Torch Linear layer
        expand dim  = 0 --> linear (x)
        expend dim != 0 --> linear ( linear (relu(x)) )
        it mean if expand_dim not 0
        just do line a linar activation  

        """
        super(FC, self).__init__()
        self.expand_dim = expand_dim
        # expand_dim is the number of neurons in the additional layer
        # expand_dim = 0 means no additional 
        # we define relu and fc_new for esthetic reasons, so that the code is more readable
        if self.expand_dim > 0:
            self.relu = torch.nn.ReLU()
            self.fc_new = torch.nn.Linear(input_dim, expand_dim)
            self.fc = torch.nn.Linear(expand_dim, output_dim)
        else:
            self.fc = torch.nn.Linear(input_dim, output_dim)
        if stddev:
            self.fc.stddev = stddev
            if expand_dim > 0:
                self.fc_new.stddev = stddev

    def forward(self, x):
        if self.expand_dim > 0:
            x = self.fc_new(x) # linear layer
            x = self.relu(x) # activation function
        x = self.fc(x) # linear layer
        return x  # outputs and not logits


class MLP(torch.nn.Module):
    """ Simple MLP with one hidden layer
    if expand_dim is not 0, add an additional layer with expand_dim neurons
    """
    def __init__(self, input_dim, num_classes, expand_dim):
        super(MLP, self).__init__()
        self.expand_dim = expand_dim
        if self.expand_dim:
            self.linear = torch.nn.Linear(input_dim, expand_dim)
            self.activation = torch.nn.ReLU()
            self.linear2 = torch.nn.Linear(expand_dim, num_classes)  # softmax is automatically handled by loss function
        self.linear = torch.nn.Linear(input_dim, num_classes)

    def forward(self, x):
        x = self.linear(x)
        if hasattr(self, 'expand_dim') and self.expand_dim:
            x = self.activation(x)
            x = self.linear2(x)
        return x

class End2EndModel(torch.nn.Module):
    """ End2EndModel is a wrapper for two models, 
    where the output of the first model is used 
    as input for the second model"""
    def __init__(self, model1, model2, use_relu=False, use_sigmoid=False,
                 n_attributes=0, n_class_attr=0):
        super(End2EndModel, self).__init__()
        # define the two models
        self.first_model = model1
        self.sec_model = model2
        self.use_relu = use_relu
        self.use_sigmoid = use_sigmoid
        self.n_attributes = n_attributes
        self.n_class_attr = n_class_attr

    def forward_stage2(self, stage1_out):
        """ Forward pass for the second model
        Args:
        stage1_out: list of outputs from the first model
        Returns:
        stacked list of outputs from the second model and the first model
        """
        # stage1_out is a list of outputs from the first model
        # we got condition on the use of relu or sigmoid for the output of the first model
        # which is the task prediction of the attributes XtoC
        if self.use_relu:
            attr_outputs = [torch.nn.ReLU()(o) for o in stage1_out]
        elif self.use_sigmoid:
            attr_outputs = [torch.nn.Sigmoid()(o) for o in stage1_out]
        else:
            attr_outputs = stage1_out

        # passer le output de la première étape sous forme de logits : relu ou sigmoid
        stage2_inputs = attr_outputs

        # stage2_inputs = torch.cat(stage2_inputs, dim=1)
        XtoC_logits = torch.stack(stage2_inputs, dim=0)
        # empile les tenseurs en dimension 0 ie
        # [(3,4), (3,4), (3,4), .., (3,4)] -> tenseurs(n,3,4)

        XtoC_logits=torch.transpose(XtoC_logits, 0, 1)
        # above switch dim0 et dim1, eg: (n,3,4) -> (3,n,4)

        # prediction des concepts below
        predictions_concept_labels = XtoC_logits.reshape(-1,self.n_attributes*self.n_class_attr)
        # above formatage des prédictions des attributs pour les passer au modèle 2

        # ------------------------- NEW -----------
        hard_mode = False
        if hard_mode :
            # Apply sigmoid activation (if not already applied) to get probabilities
            attr_outputs = [torch.nn.Sigmoid()(o) for o in stage1_out]
            XtoC_logits = torch.stack(attr_outputs, dim=0).transpose(0, 1)
            
            # Round to get binary predictions (0 or 1)
            predictions_concept_labels = torch.round(XtoC_logits)
            # print("Binary predictions (rounded logits):", predictions_concept_labels)
            
            # Reshape and convert to float for consistency with downstream layers
            predictions_concept_labels = predictions_concept_labels.reshape(-1, self.n_attributes * self.n_class_attr)
            predictions_concept_labels = predictions_concept_labels.to(torch.float32)
            # print("predictions_concept_labels", predictions_concept_labels)

        # below may serve if we are above 3 labels per concepts (absence/presence)
        #     predictions_concept_labels = F.one_hot(predictions_concept_labels)
        #     print("F.one_hot(predictions_concept_labels)", predictions_concept_labels)
        #     predictions_concept_labels = predictions_concept_labels.reshape(-1,self.n_attributes*self.n_class_attr)
        #     predictions_concept_labels = predictions_concept_labels.to(torch.float32)
        # -----------------------------------------

        stage2_inputs = predictions_concept_labels
        all_out = [self.sec_model(stage2_inputs)] # XtoY_output in position 0
        # just to ensure what is needed in the training of joint model: here we just need the logit for the BCELoss (not cross entropy)

        all_out.extend(stage1_out) # XtoC_output in position 1
        return all_out # list : [Y,C]

    def forward(self, x):
        # arg training is used to determine whether train the first model or not
        # Weird : the logic is awkward because we execute the same thing even if the first model is not trained
        # the if is not realy usefull here
        if self.first_model.training:
            outputs = self.first_model(x)
            return self.forward_stage2(outputs)
        else:
            outputs = self.first_model(x)
            return self.forward_stage2(outputs)

class ModelXtoC(torch.nn.Module):
    def __init__(self, num_classes, n_attributes=4, bottleneck=False, expand_dim=0,connect_CY=False,Lstm=False,aux_logits=False, dim = 1024):
        """
        Args:
        num_classes: number of main task classes
        aux_logits: whether to also output auxiliary logits
        transform input: whether to invert the transformation by ImageNet (should be set to True later on)
        n_attributes: number of attributes to predict
        bottleneck: whether to make X -> A model
        expand_dim: if not 0, add an additional fc layer with expand_dim neurons
        three_class: whether to count not visible as a separate class for predicting attribute
        conect_CY: whether to connect the main task with the auxiliary task
        """
        super(ModelXtoC, self).__init__()
        self.n_attributes = n_attributes
        self.bottleneck = bottleneck
        self.all_fc = torch.nn.ModuleList()  # separate fc layer for each prediction task. If main task is involved, it's always the first fc in the list
        # It's worth noting that torch.nn.ModuleList() is used instead of a regular Python 
        # list because it has additional functionality specifically designed for working
        #  with PyTorch modules. This includes features like automatically registering 
        # the modules with the PyTorch computation graph and properly handling gradients 
        # during backpropagation.
        # Overall, this line of code initializes an empty torch.nn.ModuleList() object
        # and assigns it to the variable all_fc, which will be used to store fully 
        # connected layers in a neural network.

        self.num_classes = num_classes
        self.Lstm = Lstm
        self.aux_logits = aux_logits

        #dim = 768
        if self.Lstm:
            dim = 128
        
        if self.aux_logits:
            #aux_logits: whether to also output auxiliary logits
            self.AuxLogits = ModelXtoCAux(num_classes = self.num_classes, n_attributes = self.n_attributes, bottleneck = self.bottleneck, expand_dim = 0,Lstm = self.Lstm)
        if connect_CY:
            self.cy_fc = FC(n_attributes, num_classes, expand_dim)
        else:
            self.cy_fc = None

        if self.n_attributes > 0:
            if not bottleneck: #multitasking
                self.all_fc.append(FC(dim, num_classes, expand_dim))
            for i in range(self.n_attributes):
                self.all_fc.append(FC(dim, num_classes, expand_dim))
        else:
            self.all_fc.append(FC(dim, num_classes, expand_dim))

    def forward(self, x):
        out = []
        for fc in self.all_fc:
            out.append(fc(x))
        if self.n_attributes > 0 and not self.bottleneck and self.cy_fc is not None:
            attr_preds = torch.cat(out[1:], dim=1)
            out[0] += self.cy_fc(attr_preds)
        if self.aux_logits:
            out_aux = self.AuxLogits(x)
            aux_concepts_logits = [item.cpu().detach().numpy() for item in out_aux]
            np.save('aux_concepts_logits.npy',np.array(aux_concepts_logits))
        return out

    def forward_fc_layer(self, x, layer_idx = 0):
        """
        Forward pass for a single FC layer, used for integrated gradients.
        
        Args:
            x (Tensor): Input tensor.
            layer_idx (int): Index of the FC layer in `all_fc` to forward pass through.
            
        Returns:
            Tensor: Output of the specified FC layer.
        """
        assert 0 <= layer_idx < len(self.all_fc), "Layer index out of range"
        return self.all_fc[layer_idx](x)
    
class ModelXtoCAux(torch.nn.Module):
    def __init__(self, num_classes, n_attributes, bottleneck=False, expand_dim=0, connect_CY=False,Lstm=False, dim = 1024):
        """
        Args:
        num_classes: number of main task classes
        aux_logits: whether to also output auxiliary logits
        transform input: whether to invert the transformation by ImageNet (should be set to True later on)
        n_attributes: number of attributes to predict
        bottleneck: whether to make X -> A model
        expand_dim: if not 0, add an additional fc layer with expand_dim neurons
        three_class: whether to count not visible as a separate class for predicting attribute
        """
        super(ModelXtoCAux, self).__init__()
        self.n_attributes = n_attributes
        self.bottleneck = bottleneck
        self.all_fc = torch.nn.ModuleList() #separate fc layer for each prediction task. If main task is involved, it's always the first fc in the list
        self.num_classes = num_classes
        self.Lstm = Lstm

        #dim = 768
        if self.Lstm:
            dim = 128

        if connect_CY:
            # real meaning in comment
            # self.cy_fc = FC(input_dim =  n_attributes,output_dim = num_classes, expand_dim)
            self.cy_fc = FC(n_attributes, num_classes, expand_dim)
        else:
            self.cy_fc = None

        # Check if there are attributes to predict
        if self.n_attributes > 0:
            # If not bottleneck, add a fully connected layer for the main task
            if not bottleneck: #multitasking
                self.all_fc.append(FC(dim, num_classes, expand_dim))
            # Add a fully connected layer for each attribute
            for i in range(self.n_attributes):
                self.all_fc.append(FC(dim, num_classes, expand_dim))
        else:
            # If there are no attributes, add a single fully connected layer
            self.all_fc.append(FC(dim, num_classes, expand_dim))

    def forward(self, x):
        out = []
        for fc in self.all_fc:
            out.append(fc(x))
        # Check if there are attributes to predict and if the model is not a bottleneck
        if self.n_attributes > 0 and not self.bottleneck and self.cy_fc is not None:
            attr_preds = torch.cat(out[1:], dim=1) # Concatenate the attribute predictions
            out[0] += self.cy_fc(attr_preds) # Add the main task predictions with the attribute predictions
        return out
