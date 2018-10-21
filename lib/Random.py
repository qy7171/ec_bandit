#!/usr/bin/env python
# encoding: utf-8
import logging, coloredlogs
logger = logging.getLogger(__name__)
logging_level = "INFO"
coloredlogs.install(level=logging_level, logger=logger, fmt='%(asctime)s %(hostname)s %(module)s-%(funcName)s[%(lineno)d]%(levelname)s %(message)s')

import numpy as np
import random
class Random:
  def __init__(self, dimension, alpha, lambda_,  init="zero"):  # n is number of users
    pass
  def decide(self, vidx, c_m, e_m):
    return np.random.randint(0, len(c_m))
  def updateParameters(self, x_c, x_e, reward, vidx):
    pass

