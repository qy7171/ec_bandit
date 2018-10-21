#!/usr/bin/env python
# encoding: utf-8
import logging, coloredlogs
logger = logging.getLogger(__name__)
logging_level = "INFO"
coloredlogs.install(level=logging_level, logger=logger, fmt='%(asctime)s %(hostname)s %(module)s %(funcName)s [%(lineno)d] %(levelname)s %(message)s')
import numpy as np
from collections import defaultdict

class PBMUCB:
  def __init__(self, delta):
    self.delta = delta
    self.videos = {}
  def setAgent(self, vidx, gamma):
    self.videos[vidx] = PBMUCB_agent(self.delta, gamma)
  def decide(self, vidx, qidx_list):
    return self.videos[vidx].decide(qidx_list)
  def updateParameters(self, vidx, qidx, pos, reward):
    self.videos[vidx].updateParameters(qidx, pos, reward)

class PBMUCB_agent:
  def __init__(self, delta, gamma):
    self.delta = delta
    self.gamma = gamma
    self.S = defaultdict(lambda: 0)
    self.N = defaultdict(lambda: 1)
    self.tildN = defaultdict(lambda: 1)

  def decide(self, qidx_list):
    final_idx = float('inf')
    score = -float('inf')
    equal_times = 2
    for idx, qidx in enumerate(qidx_list):
      cur_score = self.getUCB(qidx)
      if cur_score > score:
        final_idx, score = idx, cur_score
      elif cur_score == score:
        if np.random.randint(0, equal_times) == 1: # ties break even
          final_idx, score = idx, cur_score
        equal_times += 1
      else:
        pass
    return final_idx

  def getUCB(self, k):
    return self.S[k] / self.tildN[k] + np.sqrt(self.N[k] / self.tildN[k]) * np.sqrt(self.delta / (2 * self.tildN[k]))

  def updateParameters(self, qidx, pos, reward):
    assert(pos < len(self.gamma))
    self.S[qidx] += reward
    self.N[qidx] += 1
    self.tildN[qidx] += self.gamma[pos]
