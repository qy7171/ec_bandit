#!/usr/bin/env python
# encoding: utf-8
import logging, coloredlogs
logger = logging.getLogger(__name__)
logging_level = "INFO"
coloredlogs.install(level=logging_level, logger=logger, fmt='%(asctime)s %(hostname)s %(module)s %(funcName)s [%(lineno)d] %(levelname)s %(message)s')

import os
dataPath = "./data/rawData.pkl"
resultFolder = './result/'
os.makedirs(resultFolder, exist_ok=True)

E_FEATURE_DIM = 18
C_FEATURE_DIM = 72
raw_feature_dim = E_FEATURE_DIM + C_FEATURE_DIM

algs = ['Random'] + ['EC_Bandit']
algs += ['Logistic']
algs += ['hLinUCB']
algs += ['PBMUCB']
REPEAT_TIME = 100
