import numpy as np
from scipy.special import loggamma
import warnings

class UnivariateGaussianVB(object):

    def __init__(self, prior_a=1.0, prior_b=1.0, prior_mu = 1.0, 
        kappa=1.0, tol=1e-5, max_iter=20):
        """ use factored prior: p(mu,lambda) = p(mu|lambda)p(lambda)
            Gamma prior distribition for precision:
            p(lambda) = Gamma(lambda|prior_a,prior_b)
            Gaussian prior distribution for mean:
            p(mu|ldambda) = Normal(mu|prior_mu,(kappa*lambda)^-1)
        """
        self.prior_a = prior_a
        self.prior_b = prior_b
        self.prior_mu = prior_mu
        self.kappa = kappa

        self.post_a = prior_a
        self.post_b = prior_b
        self.post_mu = prior_mu
        self.post_mu_precision = kappa

        self.tol = tol
        self.max_iter = max_iter

        self._check_parameters()

    def _check_parameters(self):
        if self.prior_a <= 0 or self.prior_b <= 0:
            raise ValueError("prior_a and prior_be must be a positive number.")
        if self.kappa < 0:
            raise ValueError("kappa must be a non-negative number.")

    def _check_X(self,X):
        X = np.squeeze(X)
        if len(X.shape) != 1:
            raise ValueError("X should be 1-d array.")
        return X

    def _compute_lower_bound(self):
        return (-1/2*np.log(self.post_mu_precision) + loggamma(
            self.post_a)-self.post_a*np.log(self.post_b))

    def fit(self, X):
        X = self._check_X(X)
        N = X.size
        xbar = np.mean(X)
        x_square = np.sum(X**2)
        x_sum = np.sum(X)
        
        self.lower_bound_ = -np.infty
        self.converged_ = False
        lower_bounds = []

        for iter in range(1,self.max_iter+1):
            prev_lower_bound = self.lower_bound_

            # update q(mu)
            E_lambda = self.post_a / self.post_b
            self.post_mu = ((self.kappa * self.prior_mu +  N*xbar) 
                / (self.kappa + N))
            self.post_mu_precision = (self.kappa + N) * E_lambda

            # update q(lambda)
            E_mu = self.post_mu
            E_mu_square = 1 / self.post_mu_precision + self.post_mu ** 2
            self.post_a = self.prior_a + (N+1)/2
            self.post_b = self.prior_b
            self.post_b += 1/2*(x_square + N*E_mu_square - 2*E_mu*x_sum)
            self.post_b += self.kappa * (E_mu_square + self.prior_mu**2 - 
                2*E_mu*self.prior_mu)

            self.lower_bound_ = self._compute_lower_bound()
            lower_bounds.append(self.lower_bound_)

            change = self.lower_bound_ - prev_lower_bound
            if abs(change) < self.tol:
                self.converged_ = True
                break

        self.best_iter_ = iter
        self.lower_bounds_ = np.array(lower_bounds)
        
        if not self.converged_:
            warnings.warn("Model not converged.")
