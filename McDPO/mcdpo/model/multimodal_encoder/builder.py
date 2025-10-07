from .clip_encoder import CLIPVisionTower
from mcdpo.model.internvideo.build_internvideo import build_internvideo
from mcdpo.model.internvideo_1.build_internvideo_1 import build_internvideo_1
from mcdpo.model.motion_encoder.build_motion_encoder import build_motion_encoder

def build_vision_tower(vision_tower_cfg, **kwargs):
    image_vision_tower = kwargs['image_vision_tower']
    if image_vision_tower:
        vision_tower = getattr(vision_tower_cfg, 'image_mm_vision_tower', getattr(vision_tower_cfg, 'image_vision_tower', None))
        kwargs.pop('image_vision_tower', None)
    else:
        vision_tower = getattr(vision_tower_cfg, 'mm_vision_tower', getattr(vision_tower_cfg, 'vision_tower', None))

    print(f"Building {vision_tower}")

    if 'openai' in vision_tower or vision_tower.startswith("laion"):
        return CLIPVisionTower(vision_tower, args=vision_tower_cfg, **kwargs)

    elif 'InternVideo2' in vision_tower:
        InternVideoTower = build_internvideo(vision_tower)
        InternVideoTower.requires_grad_(False)
        return InternVideoTower
    elif 'InternVideo' in vision_tower:
        InternVideoTower = build_internvideo_1(vision_tower)
        InternVideoTower.requires_grad_(False)
        return InternVideoTower

    raise ValueError(f'Unknown vision tower: {vision_tower}')


def build_motion_tower(motion_tower_cfg):
    motion_tower = getattr(motion_tower_cfg, 'motion_mm_tower', getattr(motion_tower_cfg, 'motion_tower', None))
    MotionTower = build_motion_encoder(motion_tower)
    MotionTower.requires_grad_(False)
    return MotionTower
