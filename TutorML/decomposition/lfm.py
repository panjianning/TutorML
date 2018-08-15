import numpy as np
import os

class LFM(object):
    def __init__(self, n_factors=5, learning_rate=1e-4, reg_lambda=1.0, 
                 max_iter=5, print_every=1,early_stopping=10, p_init=None, 
                 q_init=None):
        """ Latent Fator Model: R=PQ'
            Training with Gradient Descent.
        """
        self.n_factors = n_factors
        self.learning_rate = learning_rate
        self.reg_lambda = reg_lambda
        self.max_iter = max_iter
        self.print_every = print_every
        self.early_stopping = early_stopping
        self.p_init = p_init
        self.q_init = q_init
        self._check_parameters()
        
    def _check_parameters(self):
        pass
    
    def _check_X(self, X):
        n_users, n_items = X.shape
        if self.p_init is not None and n_users != self.p_init.shape[0]:
            raise ValueError("Number of rows of X and p_init should be equal.")
        if self.q_init is not None and n_items != self.q_init.shape[0]:
            raise ValueError("Number of columns of X should be equal to the number of \
                rows of q_init")
            
    def _initialize_parameters(self, X):
        n_users, n_items = X.shape
        if self.p_init is None:
            self.p = np.random.rand(n_users, self.n_factors)
        else:
            self.p = self.p_init
        if self.q_init is None:
            self.q = np.random.rand(n_items, self.n_factors)
        else:
            self.q = self.q_init
            
    def fit(self, X, mask, test_data=None):
        self._check_X(X)
        self._initialize_parameters(X)
        n_users, n_items = X.shape
        
        if test_data is not None:
            test_idx = test_data[0]
            y_test = test_data[1].ravel()
        
        early_stop = 0
        best_mse = np.infty
        best_iter = 0
        mse_history = []
        
        for it in range(1, self.max_iter+1):
            self.r = np.dot(self.p, self.q.T)
            # gradient descent
            mask_diff = (self.r - X) * mask
            p = self.p - self.learning_rate*(np.dot(mask_diff, self.q) + 
                                        self.reg_lambda*self.p)
            q = self.q - self.learning_rate*(np.dot(mask_diff.T, self.p) + 
                                        self.reg_lambda*self.q)
            self.p = p
            self.q = q
            
            # display some info
            train_mse = np.sum(mask_diff**2)/np.sum(mask)
            if test_data is not None:
                y_test_pred = self.r.ravel()[test_idx]
                test_mse = np.mean((y_test-y_test_pred)**2)
                
            if self.print_every > 0 and it % self.print_every == 0:
                print('[Iter %03d] train mse: %.4f' % (it, train_mse), end='')
                if test_data is not None:
                    print('    test mse: %.4f' % test_mse)
                else:
                    print()

            if test_data is not None:
                mse_history.append((train_mse,test_mse))
            else:
                mse_history.append(train_mse)
            
            self.n_iters = it
            
            # check early stopping
            early_stop += 1
            mse = test_mse if test_data is not None else train_mse
            msg = 'test' if test_data is not None else 'train'
        
            if self.early_stopping <= 0:
                continue
            elif best_mse > mse:
                early_stop = 0
                best_mse = mse
                best_iter = it
            elif best_mse <= mse and early_stop >= self.early_stopping:
                print("[EarlyStop] best %s mse at iter %03d: %.4f" % (
                    msg, best_iter, best_mse))
                break

        self.best_iter = best_iter        
        self.best_mse = best_mse
        self.mse_history = np.array(mse_history)
        
    def predict(self):
        return self.r