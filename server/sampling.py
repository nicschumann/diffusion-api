import logging
import numpy as np
from time import perf_counter
from imageio import imwrite

import clip_sample
import cfg_sample

from .models import ModelConfiguration

logger = logging.getLogger('gen-api')
# model_sampling_handler

def process_pending_queue(jobs, completed):
    logger.info('starting job thread...')
    conf = ModelConfiguration()

    if conf.model == 'yfcc_1' or conf.model == 'yfcc_2' or conf.model == 'cc12m_1':
        model_data = clip_sample.prepare(conf)
        model_sample = clip_sample.sample
        conf.size = (512, 512)

    elif conf.model == 'cc12m_1_cfg':
        model_data = cfg_sample.prepare(conf)
        model_sample = cfg_sample.sample
        # conf.size = (256, 256)

    elif conf.model == 'wikiart_256':
        model_data = clip_sample.prepare(conf)
        model_sample = clip_sample.sample
        # conf.size = (256, 256)

    else:
        print(f'invalid model specified: {conf.model}')
        return

    logger.info('model loaded.')

    try:
        while True:
            if not jobs.empty():
                job = jobs.get()
                job['status'] = 'In Progress'
                completed[job['jid']] = job
                logging_interval = int(job['steps'] / 20) # 20 intermediate outs during diffusion

                def write_progress_update(info):
                    if info['i'] % logging_interval == 0:
                        logger.debug(f"jid: {job['jid']} | {(info['i'] / job['steps']) * 100:.2f}% ")
                        preds = info['pred'].cpu().numpy()
                        for i in range(preds.shape[0]):
                            imname = f"./output/{job['jid']}_{i:05}.png"
                            pred = preds[i]

                            p_max = pred.max()
                            p_min = pred.min()
                            pred = (((pred - p_min) / (p_max - p_min)) * 255.0).astype(np.uint8)

                            imwrite(imname, np.moveaxis(pred, 0, -1))

                    job['percentage_complete'] = float(info['i'] / job['steps'])
                    completed[job['jid']] = job

                conf.prompts = job['prompts']
                conf.images = []
                conf.seed = job['seed']
                conf.batch_size = job['samples']
                conf.n = job['samples']
                conf.steps = job['steps']
                if 'size' in job: conf.size = job['size']
                if 'sampler' in job: conf.sampler = job['sampler']

                logger.info(f"jid: {job['jid']} | starting")

                s = perf_counter()
                model_sample(job['jid'], conf, model_data, callback=write_progress_update, quiet=True)
                e = perf_counter()

                logger.info(f"jid: {job['jid']} | ending | duration {e - s:.2f}s")

                job['status'] = 'Completed'
                job['actual_duration'] = (e - s)
                job['percentage_complete'] = 1.0
                completed[job['jid']] = job


    except KeyboardInterrupt:
        logger.debug('exiting job thread.')
        return
