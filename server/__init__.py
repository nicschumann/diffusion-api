from typing import List, Tuple, Union
from dataclasses import dataclass
from queue import Queue
from time import sleep, perf_counter
from uuid import uuid4
from multiprocessing import Process, Queue, Manager, set_start_method
import numpy as np
from imageio import imwrite

from fastapi import FastAPI, HTTPException, Depends, status
from fastapi.responses import FileResponse
from fastapi.security import HTTPBasicCredentials

from pydantic import BaseModel

import clip_sample
import cfg_sample
from .auth import check_credentials
from .models import ModelConfiguration, JobRequest, LogConfig, VALID_SAMPLERS
from .sampling import process_pending_queue

import logging
from logging.config import dictConfig

dictConfig(LogConfig().dict())
logger = logging.getLogger('gen-api')

jobs = Queue()
completed = None
inference_thread = None
app = FastAPI()

# startup and teardown events
@app.on_event("startup")
async def start_background_processes():
    global inference_thread
    global completed

    completed = Manager().dict()
    inference_thread = Process(target=process_pending_queue, args=((jobs),(completed)))
    inference_thread.start()


@app.on_event("shutdown")
async def stop_background_processes():
    global inference_thread
    inference_thread.terminate()


# routes
@app.get("/")
async def root(credentials: HTTPBasicCredentials = Depends(check_credentials)):
    return {'hello': 'world'}


@app.post("/prompt", status_code=status.HTTP_201_CREATED)
async def prompt(job_request: JobRequest, credentials: HTTPBasicCredentials = Depends(check_credentials)):

    jid = str(uuid4())

    req = {
        'jid': jid,
        'status': 'Not Started',
        'prompts': job_request.prompts,
        'seed': job_request.seed,
        'steps': job_request.steps,
        'samples': job_request.samples,
        'percentage_complete': 0.0,
        'estimated_duration': job_request.steps / 2.34 # crude estimate based on model
    }

    if job_request.size is not None:
        if job_request.size == (256, 256) or job_request.size == (512, 512):
            req['size'] = job_request.size
        else:
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                detail="Image size must be (256,256) or (512,512)"
            )

    if job_request.sampler is not None:
        if job_request.sampler in VALID_SAMPLERS:
            req['sampler'] = job_request.sampler
        else:
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                detail=f"Sampler must be one of {VALID_SAMPLERS}"
            )


    jobs.put(req)
    completed[jid] = req

    return req


@app.get("/job/{jid}/status")
async def job(jid : str, credentials: HTTPBasicCredentials = Depends(check_credentials)):
    try:
        job = completed[jid]
        return job

    except KeyError:
        raise HTTPException(status_code=404, detail="Job not found")


@app.get("/job/{jid}/results/{index}.png")
async def image(jid : str, index : int, credentials: HTTPBasicCredentials = Depends(check_credentials)):
    try:
        job = completed[jid]
        if job['status'] != 'Not Started':
            if job['samples'] > index:
                path = f'output/{jid}_{index:05}.png'
                return FileResponse(path)
            else:
                raise HTTPException(status_code=404, detail=f"Index {index} out of range")

        else:
            raise HTTPException(status_code=404, detail="Job not started")

    except KeyError:
        raise HTTPException(status_code=404, detail="Job not found")
