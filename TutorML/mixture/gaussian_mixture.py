import numpy as np
from scipy import linalg
from .base import BaseMixture

def _estimate_gaussian_covariance(z_cond_x, X, nk, means, reg_covar):
    n_components, n_features = means.shape
    covariances = np.empty((n_components, n_features, n_features))
    for k in range(n_components):
        diff = X - means[k]
        covariances[k] = np.dot(z_cond_x[:,k] * diff.T, diff) / nk[k]
        covariances[k].flat[::n_features + 1] += reg_covar # diag
    return covariances

def _estimate_gaussian_parameters(X, z_cond_x, reg_covar):
    nk = z_cond_x.sum(axis=0)# + 10 * np.finfo(z_cond_x.dtype).eps
    means = np.dot(z_cond_x.T, X) / nk[:, np.newaxis]
    covariances = _estimate_gaussian_covariance(z_cond_x, 
        X, nk, means, reg_covar)
    return nk, means, covariances

def _compute_precision_cholesky(covariances):
    n_components, n_features, _ = covariances.shape
    precisions_chol = np.empty((n_components, n_features, n_features))
    for k, covariance in enumerate(covariances):
        try:
            cov_chol = linalg.cholesky(covariance, lower=True)
        except linalg.LinAlgError:
            raise ValueError("Cholesky decomposition failed.")
        precisions_chol[k] = linalg.solve_triangular(cov_chol, 
            np.eye(n_features), lower=True).T
    return precisions_chol

def _compute_log_det_cholesky(matrix_chol, n_features):
    n_components, _, _ =  matrix_chol.shape
    log_det_chol = (np.sum(np.log(matrix_chol.reshape(n_components, 
        -1)[:,::n_features + 1]),axis=1))
    return log_det_chol

def _gaussian_log_x_cond_z_prob(X, means, covariances):

    n_samples, n_features = X.shape
    n_components, _ = means.shape

    precisions_chol = _compute_precision_cholesky(covariances)
    log_det = _compute_log_det_cholesky(precisions_chol, n_features)

    log_prob = np.empty((n_samples, n_components))
    for k, (mu, prec_chol) in enumerate(zip(means, precisions_chol)):
        y = np.dot(X, prec_chol) - np.dot(mu, prec_chol)
        log_prob[:,k] = np.sum(np.square(y), axis=1)

    return - 0.5 * (n_features * np.log(2 * np.pi) + log_prob) + log_det


class GaussianMixture(BaseMixture):

    def __init__(self, n_components=1, tol=1e-3,reg_covar=1e-6, 
        max_iter=100, n_init=1, init_params="kmeans", z_prob_init=None, 
        means_init=None, covariances_init=None,random_state=None, 
        warm_start=False, verbose=0, verbose_interval=10):
        
        super(GaussianMixture, self).__init__(n_components=n_components,
            tol=tol, reg_covar=reg_covar, max_iter=max_iter, n_init=n_init,
            init_params=init_params, random_state=random_state, warm_start=warm_start,
            verbose=verbose,verbose_interval=verbose_interval)
        
        self.z_prob_init = z_prob_init
        self.means_init = means_init
        self.covariances_init = covariances_init

    def _initialize(self, X, z_cond_x):
        """ Model parameters (z_prob_, means_, covariances_)
            are initialized here
        """
        n_samples, _ = X.shape
        nk, means, covariances = _estimate_gaussian_parameters(
            X, z_cond_x, self.reg_covar)
    
        self.z_prob_ = (nk / n_samples) if self.z_prob_init is None else self.z_prob_init
        self.means_ = means if self.means_init is None else self.means_init
        self.covariances_ = covariances if self.covariances_init is None else self.covariances_init


    def _check_parameters(self, X):
        pass

    def _check_is_fitted(self):
        pass

    def _compute_lower_bound(self, _, mean_log_x):
        return mean_log_x

    def _get_parameters(self):
        return (self.z_prob_, self.means_, self.covariances_)

    def _set_parameters(self, params):
        (self.z_prob_, self.means_, self.covariances_) = params

    def _log_z_prob(self):
        return np.log(self.z_prob_)

    def _log_x_cond_z_prob(self, X):
        return _gaussian_log_x_cond_z_prob(X, self.means_, self.covariances_)

    def _m_step(self, X, log_z_cond_x):
        """ Model parameters (z_prob_, means_, covariances_)
            are updated here
        """
        n_samples, _ = X.shape
        nk, self.means_, self.covariances_ = (
            _estimate_gaussian_parameters(X, np.exp(log_z_cond_x), self.reg_covar))  
        self.z_prob_ = nk / n_samples