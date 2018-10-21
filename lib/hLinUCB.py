import logging, coloredlogs
logger = logging.getLogger(__name__)
logging_level = "DEBUG"
coloredlogs.install(level=logging_level, logger=logger, fmt='%(asctime)s %(hostname)s %(module)s-%(funcName)s[%(lineno)d]%(levelname)s %(message)s')

import numpy as np
class HLinUCBArticleStruct:
  def __init__(self, id, context_dimension, latent_dimension, lambda_, init="random", context_feature=None):
    self.id = id
    self.context_dimension = context_dimension
    self.latent_dimension = latent_dimension
    self.d = context_dimension+latent_dimension

    self.A2 = lambda_*np.identity(n = self.latent_dimension)
    self.b2 = np.zeros(self.latent_dimension)
    self.A2Inv = np.linalg.inv(self.A2)

    self.count = {}
    self.time = 0
    if (init=="random"):
      self.V = np.random.rand(self.d)
    else:
      self.V = np.zeros(self.d)

  def updateParameters(self, user, click):
    self.time += 1
    if user.id in self.count:
      self.count[user.id] += 1
    else:
      self.count[user.id] = 1
    self.A2 += np.outer(user.U[self.context_dimension:], user.U[self.context_dimension:])
    t = click - user.U[:self.context_dimension].dot(self.V[:self.context_dimension])
    self.b2 += user.U[self.context_dimension:]*t #(click - user.U[:self.context_dimension].dot(self.V[:self.context_dimension]))
    self.A2Inv  = np.linalg.inv(self.A2)
    self.V[self.context_dimension:] = np.dot(self.A2Inv, self.b2)
    # projection
    norm = np.linalg.norm(self.V)
    #logger.info(norm)
    self.V = self.V / norm if norm > 1 else self.V

class HLinUCBvideostruct:
  def __init__(self, id, context_dimension, latent_dimension, lambda_, init="zero"):
    self.id = id
    self.context_dimension = context_dimension
    self.latent_dimension = latent_dimension
    self.d = context_dimension+latent_dimension

    self.A = lambda_*np.identity(n = self.d)
    self.b = np.zeros(self.d)
    self.AInv = np.linalg.inv(self.A)

    self.count = {}
    self.time = 0
    if (init=="random"):
      self.U = np.random.rand(self.d)
    else:
      self.U = np.zeros(self.d)
  def updateParameters(self, article, click):
    self.time += 1
    if article.id in self.count:
      self.count[article.id] += 1
    else:
      self.count[article.id] = 1
    self.A += np.outer(article.V,article.V)
    self.b += article.V*click
    self.AInv = np.linalg.inv(self.A)

    self.U = np.dot(self.AInv, self.b)
    #projection
    norm = np.linalg.norm(self.U)
    self.U = self.U / norm if norm > 1 else self.U

  def getProb(self, alpha, alpha2, article):
    if alpha == -1:
      alpha = 0.1*np.sqrt(np.log(self.time+1))+0.1*(1-0.8**self.time)
      alpha2 = 0.1*np.sqrt(np.log(article.time+1))+0.1*(1-0.8**article.time)
    mean = np.dot(self.U, article.V)
    var = np.sqrt(np.dot(np.dot(article.V, self.AInv),  article.V))
    var2 = np.sqrt(np.dot(np.dot(self.U[self.context_dimension:], article.A2Inv),  self.U[self.context_dimension:]))
    pta = mean + alpha * var + alpha2*var2
    return pta, mean, (var, var2)

class HLinUCBAlgorithm:
  def __init__(self, context_dimension, latent_dimension, alpha, alpha2, lambda_, n, itemNum, init="zero", window_size = 1, max_window_size = 50):  # n is number of videos
    self.context_dimension = context_dimension
    self.latent_dimension = latent_dimension
    self.d = context_dimension + latent_dimension
    self.videos = []
    for i in range(n):
      self.videos.append(HLinUCBvideostruct(i, context_dimension, latent_dimension, lambda_ , init))
    self.articles = []
    for i in range(itemNum):
      self.articles.append(HLinUCBArticleStruct(i, context_dimension, latent_dimension, lambda_ , init))
    self.alpha = alpha
    self.alpha2 = alpha2
    self.window = []
    self.time = 0

  def decide(self, vidx, qidx_list, c_m, e_m):
    maxPTA = float('-inf')
    final_idx = -1
    for idx in range(len(qidx_list)):
      self.articles[qidx_list[idx]].V[:self.context_dimension] = np.hstack([c_m[idx], e_m[idx]])[-self.context_dimension:]
      x_pta, mean, var = self.videos[vidx].getProb(self.alpha, self.alpha2, self.articles[qidx_list[idx]])
      if maxPTA < x_pta:
        maxPTA = x_pta
        final_idx = idx
    return final_idx
  def updateParameters(self, x_c, x_e, reward, vidx, qidx):
    self.time += 1
    self.videos[vidx].updateParameters(self.articles[qidx], reward)
    self.articles[qidx].updateParameters(self.videos[vidx], reward)
