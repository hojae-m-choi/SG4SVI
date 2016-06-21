
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
* batchldavb.py :
* calculate_prop.py :


------------------------------------------------------------------------

# Acknowledgement

 This code was implemented during final project of 2016 spring EE531 : Statistical Learning Theory, KAIST

 Advisor : Changdong Yoo

 Team members : Hojae Choi, Soorin Yim, Yunwon Kang

 Soorin Yim, Yunwon Kang help me understand the fundamentals and theoretical base of
 variational inference, stochastic optimization and latent dirichlet allocation.