# py_f2recom

**py_f2recom** is the toolbox for the evaluation of **FESOM2-REcoM2** outputs.

**RECOM2**'s documentation:  
https://recom.readthedocs.io/en/latest/intro.html  
**FESOM2**'s documentation:  
https://fesom2.readthedocs.io/en/latest/index.html  


**py_f2recom** repository:  
https://github.com/RECOM-Regulated-Ecosystem-Model/py_f2recom.git

**py_f2recom** is based on the **pyfesom2** [python3] structure:  
https://pyfesom2.readthedocs.io/en/latest/  
https://github.com/FESOM/pyfesom2

WARNING: If you are working on the previous FESOM1.4, this toolbox IS NOT for you.  
You can refer to the ancestor called **py_recom** working with the soon obsolete python2:  
https://gitlab.dkrz.de/py_recom/py_recom

**py_recom2** works on its own environment whcih contains all dependencies.
For example, **pyfesom2** is included as a submodule.

HOW-TO-INSTALL:
-> git clone --recurse-submodules https://github.com/RECOM-Regulated-Ecosystem-Model/py_f2recom.git
-> conda create -n pyf2recom python=3.10 pip wheel
-> conda activate pyf2recom
-> cd py_f2recom/
-> pip install -e .
OPTIONAL, EXPORT KERNEL TO JUPYTERHUB:
-> python -m ipykernel install --name pyf2recom --user


[![asciicast](https://asciinema.org/a/FsP6HiP3yEklfbWj.svg)](https://asciinema.org/a/FsP6HiP3yEklfbWj)

(c) REcoM development team (MarESys Judith Hauck's group). No warranty. 
Main developper : Laurent Oziel, Tanvi Nagwekar
Contact: laurent.oziel@awi.de, tanvi.nagwekar@awi.fr, judith.hauck@awi.de
