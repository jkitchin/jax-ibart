# References

The scientific literature this package is based on. `jax-ibart` is an
independent reimplementation; please cite the original iBART papers and the
relevant method papers below.

## iBART and operator-induced structure (the method)

1. C.-Y. Liu, S. Ye, M. Li, T. P. Senftle. **"A rapid feature selection method
   for catalyst design: Iterative Bayesian additive regression trees (iBART)."**
   *The Journal of Chemical Physics* **156**, 164105 (2022).
   doi:10.1063/5.0090055
2. S. Ye, T. P. Senftle, M. Li. **"Operator-induced structural variable
   selection for identifying materials genes."** *Journal of the American
   Statistical Association* **119**(545):81-94 (2024). arXiv:2110.10195.
   doi:10.1080/01621459.2023.2294527

## Bayesian Additive Regression Trees (the selection engine)

3. H. A. Chipman, E. I. George, R. E. McCulloch. **"BART: Bayesian additive
   regression trees."** *The Annals of Applied Statistics* **4**(1):266-298
   (2010).
4. J. Bleich, A. Kapelner, E. I. George, S. T. Jensen. **"Variable selection for
   BART: An application to gene regulation."** *The Annals of Applied Statistics*
   **8**(3):1750-1781 (2014). (The Global-SE permutation threshold used here.)
5. G. Petrillo. **"Very fast Bayesian Additive Regression Trees on GPU."**
   arXiv:2410.23244 (2024). (The `bartz` library used by the faithful backend.)

## Descriptor / materials-genome context and baselines

6. L. M. Ghiringhelli, J. Vybiral, S. V. Levchenko, C. Draxl, M. Scheffler.
   **"Big data of materials science: critical role of the descriptor."**
   *Physical Review Letters* **114**, 105503 (2015).
7. R. Ouyang, S. Curtarolo, E. Ahmetcik, M. Scheffler, L. M. Ghiringhelli.
   **"SISSO: A compressed-sensing method for identifying the best
   low-dimensional descriptor in an immensity of offered candidates."**
   *Physical Review Materials* **2**, 083802 (2018). (The main baseline; see
   [iBART vs SISSO](https://github.com/mattsheng/iBART).)

## Differentiable selection (the `diff` backend)

8. C. Louizos, M. Welling, D. P. Kingma. **"Learning Sparse Neural Networks
   through L0 Regularization."** *ICLR* (2018). arXiv:1712.01312.
   (The hard-concrete gates used by `DiffSelector`.)

## Uncertainty quantification

9. V. Vovk, A. Gammerman, G. Shafer. **"Algorithmic Learning in a Random World."**
   Springer (2005). (Conformal prediction.)
10. J. Lei, M. G'Sell, A. Rinaldo, R. J. Tibshirani, L. Wasserman.
    **"Distribution-Free Predictive Inference for Regression."** *Journal of the
    American Statistical Association* **113**(523):1094-1111 (2018).
    (Split-conformal regression, the basis of `conformal_intervals`.)
11. A. N. Angelopoulos, S. Bates. **"Conformal Prediction: A Gentle
    Introduction."** *Foundations and Trends in Machine Learning* (2023).

## Dynamical-systems discovery (SINDy and DAEs)

12. S. L. Brunton, J. L. Proctor, J. N. Kutz. **"Discovering governing equations
    from data by sparse identification of nonlinear dynamical systems
    (SINDy)."** *PNAS* **113**(15):3932-3937 (2016). (Basis of `discover_ode`.)
13. N. M. Mangan, S. L. Brunton, J. L. Proctor, J. N. Kutz. **"Inferring
    biological networks by sparse identification of nonlinear dynamics."** *IEEE
    Transactions on Molecular, Biological and Multi-Scale Communications*
    **2**(1):52-63 (2016). (Implicit-SINDy; foundation for the DAE algebraic
    part.)
14. K. Kaheman, J. N. Kutz, S. L. Brunton. **"SINDy-PI: a robust algorithm for
    parallel implicit sparse identification of nonlinear dynamics."**
    *Proceedings of the Royal Society A* **476**:20200279 (2020). (Implicit /
    rational / DAE identification.)
15. D. A. Messenger, D. M. Bortz. **"Weak SINDy for partial differential
    equations."** *Journal of Computational Physics* **443**:110525 (2021).
    (Weak/integral formulation; recommended for noisy DAE discovery.)
16. R. Chartrand. **"Numerical differentiation of noisy, nonsmooth data."**
    *ISRN Applied Mathematics* (2011). (Robust derivative estimation for SINDy.)
