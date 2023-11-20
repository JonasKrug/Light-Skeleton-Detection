import torch
from torch import nn

from keypoint_rcnn import KeypointRCNN
from torchvision.models.resnet import resnet50, ResNet50_Weights
from torchvision.models.detection.backbone_utils import _resnet_fpn_extractor, _validate_trainable_layers

from torchvision.ops import misc as misc_nn_ops

def get_LSD_model(weights_path=None, device=None, disable_keypoint_predictor=False, reset_keypoint_state_predictor_weights=False):
    if weights_path is None:
        weights_backbone = ResNet50_Weights.verify(ResNet50_Weights.IMAGENET1K_V2)
        is_trained = True
    else:
        weights_backbone = None
        is_trained = False
    trainable_backbone_layers = _validate_trainable_layers(is_trained, None, 5, 3)
    norm_layer = misc_nn_ops.FrozenBatchNorm2d if is_trained else nn.BatchNorm2d

    backbone = resnet50(weights=weights_backbone, progress=True, norm_layer=norm_layer)
    backbone = _resnet_fpn_extractor(backbone, trainable_backbone_layers)

    if device is not None:
        backbone.to(device)
    
    if not disable_keypoint_predictor:
        model = KeypointRCNN(backbone=backbone,
                             num_classes = 9,
                             num_keypoints = 7
                            )
    else:
        model = KeypointRCNN(backbone=backbone,
                             num_classes = 9,
                             num_keypoints = 7,
                             keypoint_predictor=None,
                             keypoint_state_predictor=None,
                             disable_keypoint_predictor=disable_keypoint_predictor
                            )
    
    if weights_path:
        pretrained_state = torch.load(weights_path)
        model_state = model.state_dict()

        pretrained_state = { k:v for k,v in pretrained_state.items() if k in model_state and v.size() == model_state[k].size() }

        if reset_keypoint_state_predictor_weights:
            pretrained_state = { k:v for k,v in pretrained_state.items() if not k.startswith('roi_heads.keypoint_state_predictor') }
        
        model_state.update(pretrained_state)
        model.load_state_dict(model_state)
    model.to(device)

    return model
