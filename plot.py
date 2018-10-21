#!/usr/bin/env python
# encoding: utf-8
import logging, coloredlogs
logger = logging.getLogger(__name__)
logging_level = "INFO"
coloredlogs.install(level=logging_level, logger=logger, fmt='%(asctime)s %(hostname)s %(module)s %(funcName)s [%(lineno)d] %(levelname)s %(message)s')

from config import resultFolder, algs, REPEAT_TIME
import matplotlib
matplotlib.use("Agg")
from matplotlib import pyplot as plt
import numpy as np
import pickle, os
from lib.hLinUCB import HLinUCBAlgorithm
from lib.Random import Random
from lib.ClickModel import EC_Bandit, Logistic, HybridEC_Bandit
from lib.PBMUCB import PBMUCB
def plot(maxRounds=50000):
  repeat_time = 100
  algs = ['hLinUCB', 'PBMUCB', 'EC_Bandit', 'Logistic']
  base = np.array([pickle.load(open(os.path.join(resultFolder, '{}_{}.pkl'.format('Random', t)), 'rb'))[:maxRounds] for t in range(repeat_time)])
  base = list(np.mean(base, axis=0).reshape(-1))
  for alg in algs:
    try:
      data = np.array([pickle.load(open(os.path.join(resultFolder, '{}_{}.pkl'.format(alg, t)), 'rb'))[:maxRounds] / (np.array(base) + 1) for t in range(repeat_time)])
      print(data.shape)
    except:
      print(alg)
    mean = list(np.mean(data, axis=0).reshape(-1))
    std = np.std(data, axis=0).reshape(-1)
    print(len(data))
    print(alg, mean[-1], std[-1])
    plt.plot(list(range(maxRounds)), mean, label=alg)
  plt.legend()
  plt.xlabel('Decision Point')
  plt.ylabel("Relative CTR")
  plt.savefig('ctr.png')

if __name__ == '__main__':
  plot()

