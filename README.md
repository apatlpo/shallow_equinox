# 

Contains code that process mitgcm output for the EQUINOx project

- datarmor/ : code for launching

- shallow_equinox/ : utils, preprocessing and postprocessing

- doc/ : conda and git info

---
## install

*Needs update*

All scripts require python librairies that may be installed with conda according to the following instructions [here](https://github.com/apatlpo/shallow_equinox/blob/master/doc/CONDA.md)

Dependencies

---
## run on datarmor:

*Needs update*


After having installed all libraries, and cloned this repository, go into `mit_equinox/datarmor`.


```
./launch-jlab.sh
```

Follow instructions that pop up from there.

The spin up of dask relies on dask-jobqueue:
```
from dask_jobqueue import PBSCluster
local_dir = os.getenv('TMPDIR')
cluster = PBSCluster(local_directory=local_dir)
w = cluster.start_workers(10)

from dask.distributed import Client
client = Client(cluster)
```

Kill jobs once done with computations in  a notebook with:
```
cluster.stop_workers(cluster.jobs)
```
or in a shell with `python kill.py`.

Clean up after computations: `./clean.sh`

