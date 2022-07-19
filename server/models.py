from typing import List, Tuple, Union
from dataclasses import dataclass
from pydantic import BaseModel


@dataclass
class ModelConfiguration:
    prompts: List[str] = None
    images: List[str] = None
    batch_size : int = 1
    checkpoint: str = None
    clip_guidance_scale: float = 1000
    cutn: int = 64 # 64 is max on this GPU
    cut_pow: float = 1.
    device: str = None
    eta: float = 0.
    init: str = None
    method: str = 'plms' # ddpm can be okay, too
    model: str = 'yfcc_1' # cc12m_1_cfg, or yfcc_1
    n: int = 1
    seed: int = 0
    size: Tuple[int, int] = (256, 256)
    starting_timestep: float = 0.9
    steps: int = 500

class JobRequest(BaseModel):
    """
    A basic request for a new diffusion job,
    including a list or prompts and hyperparameters.
    """
    prompts : List[str]
    seed : int
    steps : int
    samples : int
    sampler : Union[str, None]
    size: Union[Tuple[int, int], None]

class JobResponse(BaseModel):
    """
    A description of a populated job request. Includes key data you
    need to query the job state, including the JID identifier. This
    JID can be used to get images and statuses in other routes.
    """
    jid: str
    author: str
    status: str
    prompts: List[str]
    seed: int
    steps: int
    samples: int
    sampler: Union[str, None]
    size: Union[Tuple[int, int], None]
    percentage_complete: float
    estimated_duration: float
    actual_duration: Union[float, None]

class JobList(BaseModel):
    author: str
    jobs: List[JobResponse]

class LogConfig(BaseModel):
    LOGGER_NAME : str = "gen-api"
    LOG_FORMAT : str = "%(levelprefix)s worker: %(asctime)s | %(message)s"
    LOG_LEVEL : str = "DEBUG"

    version = 1
    disable_existing_loggers = False
    formatters = {
        "default": {
            "()": "uvicorn.logging.DefaultFormatter",
            "fmt": LOG_FORMAT,
            "datefmt": "%Y-%m-%d %H:%M:%S"
        }
    }
    handlers = {
        "default": {
            "formatter": "default",
            "class": "logging.StreamHandler",
            "stream": "ext://sys.stderr"
        }
    }
    loggers = {
        "gen-api": {"handlers": ["default"], "level": LOG_LEVEL}
    }
