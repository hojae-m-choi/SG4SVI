
SMOOTHED GRADIENT FOR STOCHASTIC VARIATIONAL INFERENCE

Hojae Choi
hojae.choi@kaist.ac.kr

(C) Copyright 2016, Hojae Choi

This is free software, you can redistribute it and/or modify it under
the terms of the GNU General Public License.

The GNU General Public License does not permit this software to be
redistributed in proprietary programs.

This software is distributed in the hope that it will be useful, but
WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.

You should have received a copy of the GNU General Public License
along with this program; if not, write to the Free Software
Foundation, Inc., 59 Temple Place, Suite 330, Boston, MA 02111-1307
USA

------------------------------------------------------------------------

# SG4SVI

This python code implements smoothed gradient (SG) for stochastic variational inference (SVI)
in latent dirichlet allocation (LDA) model presented in the paper
 "Smoothed Gradients for Stochastic Variational Inference", Mandt S. and Blei D, NIPS 2014.

This code is based on the project of blei-lab/onlineldavb (https://github.com/blei-lab/onlineldavb)

File provided:
* batchldavb.py : A package functions for fitting LDA using stochastic 
 optimization with smoothed gradient
* calculate_prop.py : A script for calculate mean squared bias (MSB), variance, 
 mean squared error (MSE).
* wikiabs.py : A script for measure predictive probability in iterations
 to validate hyper parameter (L) of smothed gradient
* dictnostops.txt : A vocabulary of English words with the stop words removed.
* README.md : This file
* corpus.py : A package functions to read corpus. 
* parsexml.py : A script to parse contencts and extract word-count from
 XML file into corpus format.

You will need to have the `numpy` and `scipy` packages installed somewhere 
that Python can find them to use these scripts.
And you also need to have the `multiprocessing` and `threading` modules
which are used in `wikiabs.py` to validate models

## Examples
 
 Example 1
  
 `python batchldavb.py [path to corpus] 100 0.5 0.5 -1 200 300 [path to vocab]`
 
This will set LDA model and stochastic optimizer,
K = 100, alpha = 0.5, eta = 0.5, kappa = -1, windowsize(L) = 200, minibach size = 300;
and store the gradients matrix at some iterations in "gradient-[L]" folder.
 
 Example 2
 
 `python wikiabs.py 30`
 
 This will set parameters same as case of the paper (Mandt S., 2014). And 
 measuring predictive probabilities for L = 30 per 300 iterations
 
 Example 3
 
 `python calculate_prop.py`
 
 This will calculate MSB, variance and MSE using gradient get from "Example 1"

------------------------------------------------------------------------

# Acknowledgement

 This code was implemented during final project of 2016 spring EE531 : Statistical Learning Theory, KAIST

 Advisor : Changdong Yoo

 Team members : Hojae Choi, Soorin Yim, Yunwon Kang

 Soorin Yim, Yunwon Kang help me understand the fundamentals and theoretical base of
 variational inference, stochastic optimization and latent dirichlet allocation.