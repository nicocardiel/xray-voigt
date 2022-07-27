# xray-voigt

This repository contains some Python scripts used to
simulate and fit X-ray line complexes.

See the Jupyter notebook `test_simulate_voigt_complex.ipynb`
for an example using the Mn Kalpha line complex.

It is also possible to use it from the command line:
```
$ mkdir testsimul
$ cd testsimul
$ python ../simulate_voigt_complex.py \
  --name MnKa \
  --nphotini_min 4000 \
  --nphotini_max 16000 \
  --nphotini_nstep 7 \
  --nsimulations 2 \
  --fwhm_g 2.2 \
  --xmin 5860 \
  --xmax 5920 \
  --seed 123456 
Creating 2 simulations with 4000 photons
Creating 2 simulations with 6000 photons
Creating 2 simulations with 8000 photons
Creating 2 simulations with 10000 photons
Creating 2 simulations with 12000 photons
Creating 2 simulations with 14000 photons
Creating 2 simulations with 16000 photons
```