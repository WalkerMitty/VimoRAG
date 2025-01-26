import torch.nn as nn
import re


class IdentityMap(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x, *args, **kwargs):
        return x

    @property
    def config(self):
        return {"mm_projector_type": 'identity'}


def build_vision_projector(config, **kwargs):
    """
        mm_hidden_size = 1408 for InternVideo2-Stage2_1B-224p-f4 (TODO: Update it if you use a different video encoder)
    """
    # import ipdb;ipdb.set_trace()
    image_mm_projector = kwargs['image_mm_projector']
    if image_mm_projector:
        projector_type = getattr(config, 'image_mm_projector_type', 'linear')
        config.mm_hidden_size = 1024
    else:
        config.mm_hidden_size = 1408
        if 'InternVideo2' not in config.mm_vision_tower:
            config.mm_hidden_size = 768

        projector_type = getattr(config, 'mm_projector_type', 'linear')
    print(f"Building {projector_type}")

    if projector_type == 'linear':
        projector = nn.Linear(config.mm_hidden_size, config.hidden_size)
        config.mm_hidden_size = 1408
        config.image_mm_hidden_size = 1024
        return projector

    print("projector_type:", projector_type) #mlp2x_gelu
    mlp_gelu_match = re.match(r'^mlp(\d+)x_gelu$', projector_type)
    if mlp_gelu_match:
        mlp_depth = int(mlp_gelu_match.group(1))
        modules = [nn.Linear(config.mm_hidden_size, config.hidden_size)]
        for _ in range(1, mlp_depth):
            modules.append(nn.GELU())
            modules.append(nn.Linear(config.hidden_size, config.hidden_size))
        config.mm_hidden_size = 1408
        if 'InternVideo2' not in config.mm_vision_tower:
            config.mm_hidden_size = 768
        config.image_mm_hidden_size = 1024
        return nn.Sequential(*modules)

    if projector_type == 'identity':
        projector = IdentityMap()
        config.mm_hidden_size = 1408
        config.image_mm_hidden_size = 1024
        return projector

    raise ValueError(f'Unknown projector type: {projector_type}')


def build_motion_projector(config):
    """
        motion_mm_hidden_size = 563 for pretrained motion encoder (TODO: Update it if you use a different motion encoder)
    """

    projector_type = getattr(config, 'mm_projector_type', 'linear')
    config.motion_mm_hidden_size = 17*512
    print(f"Building {projector_type}")

    if projector_type == 'linear':
        projector = nn.Linear(config.motion_mm_hidden_size, config.hidden_size)
        return projector

    print("motion projector_type:", projector_type) #mlp2x_gelu
    mlp_gelu_match = re.match(r'^mlp(\d+)x_gelu$', projector_type)
    if mlp_gelu_match:
        mlp_depth = int(mlp_gelu_match.group(1))
        modules = [nn.Linear(config.motion_mm_hidden_size, config.hidden_size)]
        for _ in range(1, mlp_depth):
            modules.append(nn.GELU())
            modules.append(nn.Linear(config.hidden_size, config.hidden_size))
        config.motion_mm_hidden_size = 17*512
        return nn.Sequential(*modules)

    if projector_type == 'identity':
        projector = IdentityMap()
        config.motion_mm_hidden_size = 17*512
        return projector

    raise ValueError(f'Unknown projector type: {projector_type}')


def build_fusion_model(config):
    """
        motion_mm_hidden_size = 563 for pretrained motion encoder (TODO: Update it if you use a different motion encoder)
    """

    fusion_model = nn.Linear(768,config.hidden_size)
    return fusion_model
    # projector_type = getattr(config, 'mm_projector_type', 'linear')
    # config.motion_mm_hidden_size = 563
    # print(f"Building {projector_type}")

    # if projector_type == 'linear':
    #     projector = nn.Linear(config.motion_mm_hidden_size, config.hidden_size)
    #     return projector

    # print("motion projector_type:", projector_type) #mlp2x_gelu
    # mlp_gelu_match = re.match(r'^mlp(\d+)x_gelu$', projector_type)
    # if mlp_gelu_match:
    #     mlp_depth = int(mlp_gelu_match.group(1))
    #     modules = [nn.Linear(config.motion_mm_hidden_size, config.hidden_size)]
    #     for _ in range(1, mlp_depth):
    #         modules.append(nn.GELU())
    #         modules.append(nn.Linear(config.hidden_size, config.hidden_size))
    #     config.motion_mm_hidden_size = 563
    #     return nn.Sequential(*modules)

    # if projector_type == 'identity':
    #     projector = IdentityMap()
    #     config.motion_mm_hidden_size = 563
    #     return projector

    # raise ValueError(f'Unknown projector type: {projector_type}')