# TutorML
[![License](https://img.shields.io/pypi/l/Django.svg)](LICENSE)
[![Language](https://img.shields.io/badge/language-python-orange.svg)](README.md)

A machine learning library for tutorial

### GMM 
[code](https://github.com/PanJianning/TutorML/blob/master/TutorML/mixture/gaussian_mixture.py) ,
[doc](https://github.com/PanJianning/TutorML/blob/master/document/GMM.pdf) ,
[test](https://github.com/PanJianning/TutorML/blob/master/gmm_test.ipynb)

Gaussian Mixture Model, with parameters learned by EM algotithm.

### BernouliRBM
BernouliRBM model traning with Contrasitive Divergense

### LFM
[code](https://github.com/PanJianning/TutorML/blob/master/TutorML/decomposition/lfm.py) ,
[test](https://github.com/PanJianning/TutorML/blob/master/lfm_movielens.ipynb)

Latent Factor Model. Gradient descent method are applied to trained the model. I test it on the movielens 100k dataset(download from [here](http://files.grouplens.org/datasets/movielens/ml-100k.zip))

![LMF result](http://ok669z6cd.bkt.clouddn.com/lfm_result_.png?attname=)

### GaussianVB
[code](https://github.com/PanJianning/TutorML/blob/master/TutorML/demo/variational_bayes/unigauss_vb.py) ,
[test](https://github.com/PanJianning/TutorML/blob/master/unigauss_vb_test.ipynb)

A demo showing how to use Variational Bayes to learn the posterior distribution of univariate Gaussian's parameters

## Reference
[1] Murphy, K. P. (2013), Machine learning : a probabilistic perspective , MIT Press .
