# py_f2recom

**py_f2recom** is the toolbox for the evaluation of **FESOM2-REcoM2** outputs.

**RECOM2**'s documentation:  
https://recom.readthedocs.io/en/latest/intro.html  
**FESOM2**'s documentation:  
https://fesom2.readthedocs.io/en/latest/index.html  


**py_f2recom** repository:  
https://gitlab.dkrz.de/a270114/py_f2recom  
**py_f2recom** is based on the **pyfesom2** [python3] structure:  
https://pyfesom2.readthedocs.io/en/latest/  
https://github.com/FESOM/pyfesom2

WARNING: If you are working on the previous FESOM1.4, this toolbox IS NOT for you.  
You can refer to the ancestor called **py_recom** working with the soon obsolete python2:  
https://gitlab.dkrz.de/py_recom/py_recom

**py_recom2** works like **py_fesom2**, but adds some custom dependencies which are included (i.e. improved version of pyfesom2, skillmetrics, cmocean). All you need is to copy the directory (git clone https://gitlab.dkrz.de/a270114/py_f2recom.git), adapt your own paths, and a python3 distribution (e.g.anaconda3) that already contains the required mainstream librairies (e.g. numpy, xarray, matplotlib >= 3.3, pandas, cartopy, netcdf) and NOT an conda environment. For example, on the machine Ollie just type 'module load python', on HLRN machine  'module load anaconda3' (although you might need on HLRN to install/update locally some missing modules using 'pip install module_name --user').
 
**Note1**: Using Robinson projection needs more time than the standard Plate Carree projection (default).  
**Note2**: in order to avoid memory issue and restart kernels, most functions do not output variables.  
However, they could output the variables by assiggning an object (e.g., output = PHC3tempcomp()). 
**Note3**: Use the MASTER script in the GlobalAssessment folder, to get a complete overview.

FESOM2 documentation. https://fesom2.readthedocs.io/en/latest/geometry.html  
REcoM2 documentation: https://recom.readthedocs.io/en/latest/intro.html

py_recom allows in some analysis the masking/comparison of oceans, for example:

    Available regions:
            Ocean Basins:
                "Atlantic_Basin"
                "Pacific_Basin"
                "Indian_Basin"
                "Arctic_Basin"
                "Southern_Ocean_Basin"
                "Mediterranean_Basin"
                "Global Ocean"
                "Global Ocean 65N to 65S"
                "Global Ocean 15S to 15N"
            MOC Basins:
                "Atlantic_MOC"
                "IndoPacific_MOC"
                "Pacific_MOC"
                "Indian_MOC"
            Nino Regions:
                "Nino 3.4"
                "Nino 3"
                "Nino 4"
            Arctic Ocean regions:
                "Amerasian basin"
                "Eurasian basin"

The user can refer to the 'get_mask' funstion in '/modules/pyseom2/ut.py'

(c) REcoM development team (MarESys Judith Hauck's group). No warranty. contact: laurent.oziel@awi.de
