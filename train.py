#!/usr/bin/env python
# encoding: utf-8
import logging, coloredlogs
logger = logging.getLogger(__name__)
logging_level = "DEBUG"
coloredlogs.install(level=logging_level, logger=logger, fmt='%(asctime)s %(hostname)s %(module)s-%(funcName)s[%(lineno)d]%(levelname)s %(message)s')
import pandas as pd
from config import E_FEATURE_DIM, C_FEATURE_DIM, raw_feature_dim, dataPath, resultFolder
import pickle, os
from lib.hLinUCB import HLinUCBAlgorithm
from lib.Random import Random
from lib.ClickModel import EC_Bandit, Logistic
from lib.PBMUCB import PBMUCB
class simulateOnlineData(object):
  def __init__(self):
    pass

  def runAlgorithms(self, algorithms, data=None):
    cumulative_reward = {}
    alg_iter = {}
    for alg_name in algorithms:
      cumulative_reward[alg_name] = [0]
      alg_iter[alg_name] = 0
    for vidx, arm, reward, candidate_arms, candidate_q_features, candidate_e_features, pos in data:
      for alg_name, alg in list(algorithms.items()):
        if alg_name in ['Random', 'EC_Bandit', 'Logistic']:
          arm_idx = alg.decide(vidx, candidate_q_features, candidate_e_features)
        elif alg_name in ['hLinUCB']:
          arm_idx = alg.decide(vidx, candidate_arms, candidate_q_features, candidate_e_features)
        elif alg_name in ['PBMUCB']:
          qidx_list = [(a - 1) // 6 for a in candidate_arms]
          arm_idx = alg.decide(vidx, qidx_list)
        else:
          return
        alg_iter[alg_name] += 1
        if alg_iter[alg_name] % 5000 == 0:
          logger.info("iter: {}, hit: {}, ratio: {}".format(alg_iter[alg_name], len(cumulative_reward[alg_name]), len(cumulative_reward[alg_name]) / float(alg_iter[alg_name])))
          logger.info("average reward:{}".format(cumulative_reward[alg_name][-1] / float(len(cumulative_reward[alg_name]))))
        if candidate_arms[arm_idx] != arm:
          continue
        if alg_name in ['EC_Bandit', 'Logistic']:
          alg.updateParameters(candidate_q_features[arm_idx], candidate_e_features[arm_idx], reward, vidx)
        elif alg_name in ['hLinUCB']:
          alg.updateParameters(candidate_q_features[arm_idx], candidate_e_features[arm_idx], reward, vidx, arm)
        elif alg_name in ['PBMUCB']:
          alg.updateParameters(vidx, (arm - 1) // 6, pos, reward)
        elif alg_name in ['Random']:
          pass
        else:
          alg.updateParameters(x_c=None, x_e=None, reward=reward, vidx=vidx)
        cumulative_reward[alg_name].append(cumulative_reward[alg_name][-1] + reward)
    return cumulative_reward[alg_name]

def initialPBMUCB():
  PBMUCBAlg = PBMUCB(0.05)
  """
  In PBMUCB, the probability of examination at any position is assumed known beforehand. Here we adopted the paper's approach, i.e., estimating them from logged data using EM algorithm.
  The values below are what we get using this approach.
  """
  PBMUCBAlg.setAgent(19, [0.1368,  0.1691, 0.1142, 0.1373, 0.1803, 0.0296])
  PBMUCBAlg.setAgent(5,  [0.4099,  0.1477, 0.1685, 0.1942, 0.1805, 0.1807])
  PBMUCBAlg.setAgent(8,  [0.1431,  0.1274, 0.0989, 0.1209, 0.1443, 0.1382])
  PBMUCBAlg.setAgent(9,  [0.2568,  0.1654, 0.0912, 0.1079, 0.1377, 0.0886])
  PBMUCBAlg.setAgent(10, [0.3378,  0.2287, 0.2456, 0.2181, 0.1821, 0.0556])
  PBMUCBAlg.setAgent(11, [0.2181,  0.2451, 0.1035, 0.0945, 0.0652, 0.2687])
  PBMUCBAlg.setAgent(12, [0.0746,  0.2455, 0.1707, 0.2462, 0.0946, 0.2491])
  PBMUCBAlg.setAgent(13, [0.2220,  0.2348, 0.2158, 0.1737, 0.1476, 0.1270])
  PBMUCBAlg.setAgent(16, [0.3516,  0.1060, 0.2453, 0.0796, 0.1222, 0.1450])
  PBMUCBAlg.setAgent(17, [0.2498,  0.0949, 0.0703, 0.1340, 0.1021, 0.1366])
  PBMUCBAlg.setAgent(18, [0.1895,  0.1747, 0.2150, 0.2924, 0.0856, 0.1733])
  PBMUCBAlg.setAgent(51, [0.2692,  0.2510, 0.1113, 0.1400, 0.1848, 0.1855])
  PBMUCBAlg.setAgent(52, [0.1131,  0.3457, 0.2735, 0.1483, 0.1787, 0.1364])
  PBMUCBAlg.setAgent(21, [0.2141,  0.1583, 0.2750, 0.0397, 0.0775, 0.1428])
  PBMUCBAlg.setAgent(41, [0.1835,  0.2717, 0.1025, 0.1476, 0.0280, 0.1238])
  PBMUCBAlg.setAgent(56, [0.1838,  0.2534, 0.2853, 0.1463, 0.2493, 0.1992])
  PBMUCBAlg.setAgent(25, [0.3068,  0.1915, 0.0909, 0.2055, 0.2030, 0.0957])
  PBMUCBAlg.setAgent(26, [0.1170,  0.1157, 0.1237, 0.1330, 0.1203, 0.1809])
  PBMUCBAlg.setAgent(27, [0.3224,  0.0864, 0.0617, 0.1707, 0.0891, 0.1156])
  PBMUCBAlg.setAgent(28, [0.2108,  0.2320, 0.2117, 0.0957, 0.0964, 0.0655])
  PBMUCBAlg.setAgent(15, [0.1235, 0.2136, 0.1173, 0.2200, 0.0427, 0.0051])
  return PBMUCBAlg
def main(lambda_f = 1, algs=None, num_alpha=None, data=None):
  lambda_ = 0.1   # Initialize A
  alg_name = algs #if alg_name is None else alg_name
  simExperiment = simulateOnlineData()
  algorithms = {}
  if alg_name == 'EC_Bandit':
    algorithms[alg_name] = EC_Bandit(feature_dim = raw_feature_dim, E_FEATURE_DIM=E_FEATURE_DIM, C_FEATURE_DIM=C_FEATURE_DIM)
  if alg_name == 'hLinUCB':
    n_users = 505
    n_articles = 3000
    algorithms[alg_name] = HLinUCBAlgorithm(context_dimension = raw_feature_dim, latent_dimension = 5, alpha = 0.2, alpha2 = 0.2, lambda_ = lambda_, n = n_users, itemNum=n_articles, init='random', window_size = -1)
  if alg_name == 'Logistic':
    algorithms[alg_name] = Logistic(feature_dim = raw_feature_dim, E_FEATURE_DIM=E_FEATURE_DIM, C_FEATURE_DIM=C_FEATURE_DIM)
  if alg_name == 'Random':
    algorithms[alg_name] = Random(dimension = raw_feature_dim, alpha = num_alpha, lambda_ = lambda_)
  if alg_name == 'PBMUCB':
    algorithms[alg_name] = initialPBMUCB()
  print(algorithms)
  return simExperiment.runAlgorithms(algorithms, data=data)

def task(DEBUG_VIDX=None, alg=None, repeat_time=0):
  print(alg)
  data = pickle.load(open(dataPath, 'rb'))
  cumulative_reward = main(algs=alg, data=data)
  pickle.dump(cumulative_reward, open(os.path.join(resultFolder, "{}_{}.pkl".format(alg, repeat_time)), 'wb'))
if __name__ == '__main__':
  from config import algs, REPEAT_TIME
  for repeat_time in range(REPEAT_TIME):
    for alg in algs:
      task(alg=alg, repeat_time=repeat_time)
