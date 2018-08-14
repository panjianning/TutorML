from __future__ import print_function

import warnings
from abc import ABCMeta, abstractmethod
import six
import sklearn.cluster as cluster

import numpy as np
from scipy.misc import logsumexp

from ..utils import check_random_state

class BaseEstimator(object):
    pass

class DensityMixin(object):
    def score(self, X, y=None):
        pass

class BaseMixture(six.with_metaclass(ABCMeta, DensityMixin, BaseEstimator)):
    
    def __init__(self, n_components, tol, reg_covar, max_iter, n_init,
        init_params, random_state, warm_start, verbose, verbose_interval):
        self.n_components = n_components
        self.tol = tol
        self.reg_covar =  reg_covar
        self.max_iter = max_iter
        self.n_init = n_init
        self.init_params = init_params
        self.random_state = random_state
        self.warm_start = warm_start
        self.verbose = verbose
        self.verbose_interval = verbose_interval

    def _initialize_parameters(self, X, random_state):
        n_samples, _ = X.shape
        if self.init_params == 'kmeans':
            z_cond_x = np.zeros((n_samples, self.n_components))
            label = cluster.KMeans(n_clusters=self.n_components, n_init=1,
                random_state=random_state).fit(X).labels_
            z_cond_x[np.arange(n_samples), label] = 1
        elif self.init_params == 'random':
            z_cond_x = random_state.rand(n_samples, self.n_components)
            z_cond_x /= z_cond_x.sum(axis=1)[:,np.newaxis]
        else:
            raise ValueError("Unimplemented initialization method '%s'" % self.init_params)

        self._initialize(X, z_cond_x)

    @abstractmethod
    def _initialize(self, X, z_cond_x):
        pass

    def _check_initial_parameters(self, X):
        # do some check
        self._check_parameters(X)

    @abstractmethod
    def _check_parameters(self, X):
        pass

    @abstractmethod
    def _check_is_fitted(self):
        pass

    def fit(self, X, y=None):

        self._check_initial_parameters(X)

        do_init = not(self.warm_start and hasattr(self, 'converged_'))
        n_init = self.n_init if do_init else 1
        max_lower_bound = -np.infty
        self.converged_ = False

        random_state = check_random_state(self.random_state)

        n_samples, _ = X.shape
        for init in range(n_init):
            if do_init:
                self._initialize_parameters(X, random_state)
                self.lower_bound_ = -np.infty
            for n_iter in range(self.max_iter):
                prev_lower_bound = self.lower_bound_
                mean_log_x, log_z_cond_x = self._e_step(X)
                self._m_step(X, log_z_cond_x)
                self.lower_bound_ = self._compute_lower_bound(log_z_cond_x, mean_log_x)
                change = self.lower_bound_ - prev_lower_bound
                if abs(change) < self.tol:
                    self.converged_ = True
                    break

            if self.lower_bound_ > max_lower_bound:
                max_lower_bound = self.lower_bound_
                best_params = self._get_parameters()
                best_n_iter = n_iter

        if not self.converged_:
            warnings.warn("Model not converged.")

        self._set_parameters(best_params)
        self.n_iter_ = best_n_iter

        return self

    def _e_step(self, X):
        log_x, log_z_cond_x = self._log_z_cond_x_prob(X)
        return np.mean(log_x), log_z_cond_x

    @abstractmethod
    def _m_step(self, X, log_resp):
        pass

    @abstractmethod
    def _get_parameters(self):
        pass

    abstractmethod
    def _set_parameters(self, params):
        pass

    @abstractmethod
    def _compute_lower_bound(self, log_z_cond_x, mean_log_x):
        pass

    @abstractmethod
    def _log_z_prob(self):
        pass

    @abstractmethod
    def _log_x_cond_z_prob(self, X):
        pass

    def _log_x_and_z_prob(self, X):
        return self._log_x_cond_z_prob(X) + self._log_z_prob()

    def _log_x_prob(self, X):       
        self._check_is_fitted()
        return logsumexp(self._log_x_and_z_prob(X),axis=1)

    def _log_z_cond_x_prob(self, X):
        log_x_and_z = self._log_x_and_z_prob(X)
        log_x = logsumexp(log_x_and_z, axis=1)
        log_z_cond_x = log_x_and_z - log_x[:,np.newaxis]
        return log_x, log_z_cond_x

    def predict(self, X):
        self._check_is_fitted()
        return self._log_x_and_z_prob(X).argmax(axis=1)

    def predict_proba(self, X):
        _, log_z_cond_x = self._log_z_cond_x_prob(X)
        return np.exp(log_z_cond_x)

    def score(self, X, y=None):
        """ mean log likelihood
        """
        return self._log_x_prob(X).mean()

    def sample(self, n_samples=1):
        self._check_is_fitted()
        _, n_features = self.means_.shape
        rng = check_random_state(self.random_state)
        z_samples = rng.multinomial(n_samples, self.z_prob_)
        X = np.vstack([
            rng.multivariate_normal(mean, covariance, int(sample))
            for (mean, covariance, sample) in zip(
                self.means_, self.covariances_, z_samples)])
        y = np.concatenate([np.full(sample, j, dtype=int) for j,
            sample in enumerate(z_samples)])
        return (X,y)