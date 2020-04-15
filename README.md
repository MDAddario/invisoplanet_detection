__Abstract__

Using Bayesian inference, the detection of the unknown masses of planets and stars was attempted by simulating the trajectory of N known bodies in the presence of m unknown bodies in two dimensions. 
The invisible planets were then detected by applying a Markov-Chain Monte Carlo (MCMC) algorithm to a posterior distribution for the mass of the unknowns built from the probability that their presence with a given mass gave rise to the trajectory simulated. 
This probability was estimated from the Gaussian difference between the simulated trajectory and the trajectories obtained with the unknowns at each mass. 
The program was found to be quite sensitive to unknown masses that caused only small perturbations in the trajectories of the known masses, but gave incorrect results for prior distributions which resulted in instability. 
Provided with the Sun and a few other known planets from the solar system, it was found to be possible to get a good approximation on the mass of a planet in the system made "invisible" for the purpose of the simulation. 
Given knowledge of the planets in the solar systems, and reasonable constraints on the prior distribution, the mass of the sun was determined to **0.3% of the actual**.

![Alt Text](assets/solar_system_animation.gif)

__Program dependencies__

In addition to the default packages provided by Anaconda (tested 2020/04/07), the following packages must be installed to ensure complete compatitbility.
`Pyglet` is a package used to animate the simulations, and `Corner` is a package used to generate corner plots of the MCMC runs.

```
conda install -c conda-forge pyglet
conda install -c astropy corner
```

__Description of file directories__

The following is a breakdown of all the files present in the GitHub repository. 
All of the files in the repository are intended for the final submission.

+ `assets/` - _Folder containing sample animation .gif for the README.md_

+ `final_report/` - _Contains the manuscript for the final report submission_

+ `invisoplanet_detection/` - _Module containing all of the source code_

    + `animations/` - _Submodule for running animations_

    + `data/` - _Subdirectory containing initial conditions for various systems_

    + `simulations/` - _Submodule for conducting simulations_

        + `tests/` - _Unit tests for simulations submodule_

    + `statistics/` - _Submodule for performing statistics_

        + `tests/` - _Unit tests for statistics submodule_

+ `preplanning/` - _Contains all of the brainstorms and initial project planning_

+ `Running_MCMC.ipynb` - _Main code used to conduct the MCMC analyses_

+ `contributions.txt` - _Breakdown of contributions per member_

+ `generate_animation.py` - _Main code used to generate animations_

The remaining files `.gitignore` and `README.md` are typical.

__Authors__

The three authors for this work are Delaney Dunne (DD), Michael Lindner-D'Addario (MLD), and Gabriella Morin (GM).