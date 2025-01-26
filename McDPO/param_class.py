from dataclasses import dataclass, field
from typing import Any, Optional,  List

@dataclass
class EvaluationParams:
    val_loader: Optional[Any] = None
    net: Optional[Any] = None
    model: Optional[Any] = None
    logger: Optional[Any] = None
    tokenizer: Optional[Any] = None
    tokenizer_image_token: Optional[Any] = None
    eval_wrapper: Optional[Any] = None
    temperature: float = 1.0  # Assuming 1.0 as a reasonable default
    video_dir: Optional[str] = None
    image_processor: Optional[Any] = None
    video_processor: Optional[Any] = None
    start_id: int = -1  # Assuming 0 as a default value
    end_id: int = -1  # Assuming 0 as a default value
    out_dir: Optional[str] = None
    retrieval_result: Optional[str] = None
    text_only: bool = False  # Assuming False as a default value
    generated_file: Optional[str] = None
    candidate_files: List[str] = field(default_factory=list) 
    fid_weight: float = 0.5
    match_weight: float = 0.5
    motion_encoder: bool = False
