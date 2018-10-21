#!/usr/bin/env python
# encoding: utf-8
import logging, coloredlogs
logger = logging.getLogger(__name__)
logging_level = "DEBUG"
coloredlogs.install(level=logging_level, logger=logger, fmt='%(asctime)s %(hostname)s %(module)s %(funcName)s [%(lineno)d] %(levelname)s %(message)s')
import os
import pandas as pd
import pickle
import numpy as np
import random
import scipy as sci
from scipy.special import expit
from scipy.misc import logsumexp
from numpy import tanh, sqrt

def get_prob(theta_C, theta_E, X_C, X_E):
  return expit(X_C.T @ theta_C) * expit(X_E.T @ theta_E)
def get_reward(theta_C, theta_E, X_C, X_E):
  return 1 if random.random() < get_prob(theta_C, theta_E, X_C, X_E) else 0
def lambda_func(kesi):
  return tanh(kesi / 2) / (4 * kesi) if kesi > 0 else 1 / 8
def big_F_expectation(H, kesi):
  """
  the expectation of the q's factor
  """
  return (H - kesi) / 2 + np.log(expit(kesi))
def big_F_without_Expectation(H, kesi):
  return (H - kesi) / 2 + np.log(expit(kesi)) - lambda_func(kesi) * (H**2 - kesi**2)
def update_q(HC0, HE1, HE0, kesi1, kesi2, kesi3):
  alpha1 = big_F_expectation(HC0, kesi1)
  alpha2 = big_F_expectation(HE1, kesi2)
  alpha3 = big_F_expectation(HE0, kesi3)
  return expit(alpha1 + alpha2 - alpha3)
def cal_lower_prob(kesi, H_s):
  return expit(kesi) * np.exp((H_s - kesi) / 2 - lambda_func(kesi) * (H_s ** 2 - kesi ** 2))
def cal_q_lower_prob(q, kesi_C, kesi_E, HC0, HE1, HE0):
  return np.exp(-q * np.log(q) - (1-q) * np.log(1-q) + q * big_F_without_Expectation(HC0, kesi_C) + q * big_F_without_Expectation(HE1, kesi_E),\
                + (1-q) * big_F_without_Expectation(HE0, kesi_E))
def big_H(S, theta, X):
  return (2*S-1) * theta.T @ X
def pick_arm(C_theta, E_theta, C_arms, E_arms):
  return np.argmax(expit(C_theta.T @ C_arms) * expit(E_theta.T @ E_arms))

class EC_Bandit:
  def __init__(self, feature_dim, C_FEATURE_DIM=None, E_FEATURE_DIM=None):
    self.feature_dim = feature_dim
    self.C_FEATURE_DIM, self.E_FEATURE_DIM = C_FEATURE_DIM, E_FEATURE_DIM
    self.E_sigma_init = np.eye(self.E_FEATURE_DIM) * 10
    self.C_sigma_init = np.eye(self.C_FEATURE_DIM) * 10
    self.E_mu_init = np.zeros((1, self.E_FEATURE_DIM)).reshape(-1, 1)
    self.C_mu_init = np.zeros((1, self.C_FEATURE_DIM)).reshape(-1, 1)
    self.videos = {}
  def decide(self, vidx, c_m, e_m):
    if not vidx in self.videos:
      self.videos[vidx] = VI_agent(self.C_sigma_init, self.C_mu_init, self.E_sigma_init, self.E_mu_init)
    return self.videos[vidx].decide(c_m[:, -self.C_FEATURE_DIM:].T, e_m[:, :self.E_FEATURE_DIM].T)
  def updateParameters(self, x_c, x_e, reward, vidx):
    x_c, x_e = x_c[-self.C_FEATURE_DIM:].reshape(-1, 1), x_e[:self.E_FEATURE_DIM].reshape(-1, 1)
    self.videos[vidx].update(x_c, x_e, reward)

class Logistic:
  def __init__(self, feature_dim, C_FEATURE_DIM=None, E_FEATURE_DIM=None):
    self.feature_dim = feature_dim
    self.C_FEATURE_DIM, self.E_FEATURE_DIM = C_FEATURE_DIM, E_FEATURE_DIM
    self.sigma_init = np.eye(self.feature_dim) * 10
    self.mu_init = np.zeros((1, self.feature_dim)).reshape(-1, 1)
    self.videos = {}
  def decide(self, vidx, c_m, e_m):
    if not vidx in self.videos:
      self.videos[vidx] = LaplaceApproxiate_agent(self.sigma_init, self.mu_init)
    x = np.hstack([c_m[:, -self.C_FEATURE_DIM:], e_m[:, :self.E_FEATURE_DIM]]).T
    return self.videos[vidx].decide(x)
  def updateParameters(self, x_c, x_e, reward, vidx):
    self.videos[vidx].update(np.vstack([x_c[-self.C_FEATURE_DIM:].reshape(-1, 1), x_e[:self.E_FEATURE_DIM].reshape(-1, 1)]), x_e[:self.E_FEATURE_DIM].reshape(-1, 1), reward)

class VI_agent:
  def __init__(self, C_sigma, C_mu, E_sigma=None, E_mu=None):
    self.name = 'ts_vi_agent'
    self.C_mu = C_mu
    self.C_sigma = C_sigma
    self.E_sigma = E_sigma
    self.E_mu = E_mu
    self.ReSampleFlag = True # for accelarating the decide process, if the mu and sigma has not been updated, then the sample last round could be used this round
    self.update_time = 0
  def decide(self, C_arms, E_arms):
    if self.ReSampleFlag:
      try:
        self.sampled_C_theta = np.random.multivariate_normal(self.C_mu.reshape(-1), self.C_sigma).reshape(-1, 1)
      except:
        self.sampled_C_theta = np.random.multivariate_normal(self.C_mu.reshape(-1), self.C_sigma + 0.001 * np.eye(self.C_sigma.shape[0])).reshape(-1, 1)
      try:
        self.sampled_E_theta = np.random.multivariate_normal(self.E_mu.reshape(-1), self.E_sigma).reshape(-1, 1)
      except:
        self.sampled_E_theta = np.random.multivariate_normal(self.E_mu.reshape(-1), self.E_sigma + 0.001 * np.eye(self.E_sigma.shape[0])).reshape(-1, 1)
      self.ReSampleFlag = False
    chosen_arm = pick_arm(self.sampled_C_theta, self.sampled_E_theta, C_arms, E_arms)
    return chosen_arm
  def get_eignvalue(self):
    return (np.linalg.norm(self.C_sigma, ord=2), np.linalg.norm(self.E_sigma, ord=2))
  def update(self, X_C, X_E, S, examination=None, update_rounds=5):
    if S == 1:
      self._update_1(X_C, X_E, S=1, update_rounds=update_rounds)
    elif examination is None:
      self._update_0(X_C, X_E, S=0, update_rounds=update_rounds)
    else:
      assert(examination == 0)
      self._update_with_examination(X_C, X_E, S=0, examination=0, update_rounds=update_rounds)
    self.ReSampleFlag, self.sampled_C_theta, self.sampled_E_theta = True, None, None # reset the resample state
    self.update_time += 1
  def _update_1_helper(self, sigma, mu, X, kesi, S, q=1, addition=1):
    inverse_sigma = np.linalg.inv(sigma)
    post_inverse_sigma = inverse_sigma + 2 * q * lambda_func(kesi) * X @ X.T
    new_sigma = np.linalg.inv(post_inverse_sigma) # post sigma
    new_mu = new_sigma @ ( inverse_sigma @ mu + (q * (S - 0.5) *(2 * addition - 1)) * X)
    return new_mu, new_sigma
  def _update_helper_kesi(self, sigma, mu, X):
    return sqrt(X.T @ sigma @ X + ( X.T @ mu )**2)
  def _update_with_examination(self, X_C, X_E, S=0, examination=0, update_rounds=5):
    kesi_C, kesi_E = 0, 0
    for i in range(update_rounds):
      self.C_mu, self.C_sigma = self._update_1_helper(self.C_sigma, self.C_mu, X_C, kesi_C, S=S)
      self.E_mu, self.E_sigma = self._update_1_helper(self.E_sigma, self.E_mu, X_E, kesi_E, S=examination)
      kesi_C = self._update_helper_kesi(self.C_sigma, self.C_mu, X_C)
      kesi_E = self._update_helper_kesi(self.E_sigma, self.E_mu, X_E)
  def _update_1(self, X_C, X_E, S=1, update_rounds=5):
    kesi_C, kesi_E = 0, 0
    for i in range(update_rounds):
      self.C_mu, self.C_sigma = self._update_1_helper(self.C_sigma, self.C_mu, X_C, kesi_C, S=1)
      self.E_mu, self.E_sigma = self._update_1_helper(self.E_sigma, self.E_mu, X_E, kesi_E, S=1)
      kesi_C = self._update_helper_kesi(self.C_sigma, self.C_mu, X_C)
      kesi_E = self._update_helper_kesi(self.E_sigma, self.E_mu, X_E)
  def _update_0(self, X_C, X_E, S=0, update_rounds=5):
    for i in range(update_rounds):
      kesi_C = self._update_helper_kesi(self.C_sigma, self.C_mu, X_C) if i > 0 else 0
      kesi_E = self._update_helper_kesi(self.E_sigma, self.E_mu, X_E) if i > 0 else 0
      HC0 = big_H(S=0, theta=self.C_mu, X=X_C)
      HE1 = big_H(S=1, theta=self.E_mu, X=X_E)
      HE0 = big_H(S=0, theta=self.E_mu, X=X_E)
      q = update_q(HC0, HE1, HE0, kesi_C, kesi_E, kesi_E) if i > 0 else 1 / 2 # initial q set as 1/2
      self.C_mu, self.C_sigma = self._update_1_helper(self.C_sigma, self.C_mu, X_C, kesi_C, S=0, q=q)
      self.E_mu, self.E_sigma = self._update_1_helper(self.E_sigma, self.E_mu, X_E, kesi_E, S=1, q=1, addition=q)

class LaplaceApproxiate_agent(VI_agent):
  def __init__(self, sigma, mu):
    super().__init__(sigma, mu)
    self.name = 'ts_la'
  def get_eignvalue(self):
    return np.linalg.norm(self.C_sigma, ord=2), None
  def decide(self, x):
    if self.ReSampleFlag:
      try:
        self.sampled_theta = np.random.multivariate_normal(self.C_mu.reshape(-1), self.C_sigma).reshape(-1, 1)
      except: # numerical stability
        self.sampled_theta = np.random.multivariate_normal(self.C_mu.reshape(-1), self.C_sigma + 0.001 * np.eye(self.C_sigma.shape[0])).reshape(-1, 1)
      self.ReSampleFlag = False
    chosen_arm = np.argmax(self.sampled_theta.T @ x)
    return chosen_arm
  def update(self, X_C, X_E, S, update_rounds=None):
    p_estimate = expit(self.C_mu.T @ X_C)
    inverse_sigma = np.linalg.inv(self.C_sigma)
    post_inverse_sigma = inverse_sigma + p_estimate * (1 - p_estimate) * X_C @ X_C.T
    self.C_sigma = np.linalg.inv(post_inverse_sigma)
    self.C_mu = self.C_mu + (S - p_estimate) * self.C_sigma @ X_C
    self.ReSampleFlag, self.sample_theta = True, None
    self.update_time += 1

if __name__ == '__main__':
  pass
