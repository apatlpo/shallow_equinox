## install

Install shenfun, see [doc](https://shenfun.readthedocs.io/en/latest/installation.html):

```
conda create -n shenfun_pip -c conda-forge python=3.6
conda activate shenfun_pip
conda install -c conda-forge mpi4py mpich fftw numpy cython sympy matplotlib
pip install shenfun
```

The following should work but is bugged at the moment:
```
conda create --name shenfun -c conda-forge -c spectralDNS shenfun
conda activate shenfun
conda install -c conda-forge matplotlib
```

To remove environment:
```
conda remove --name shenfun_pip --all
```
