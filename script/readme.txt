Readme txt by Dr. Tanda Li The University of Birmingham

Any questions please contact Tanda Li < litanda@hotmail.com >

Date: 20211207

This script folder includes main codes used in the paper darft named "Modelling stars with Gaussian Process Regression: Augmenting Stellar Model Grid" by Tanda Li et al. 

There are four scripts in the folder. They have been used to train GP models to map from five stellar fundamental parameters (Mass, EEP, metallicity, helium fraction, and mixing-length parameters) onto output quantities (effective T, gravity, radius, surface metallicity, and stellar age). The idea of the work is applying GP as an intelligent interpolator for sparse stellar model grid. 

These four scripts were only for the final GP models which are mentioned in Section 5 - 7 of the paper. Scripts for preliminary studies are not included.  


"section_gp-5d-train-validate.py" is the script for training GP models and validating them (at every iteration). 

"section_gp-5d-test.py" is for testing the GP model performance with a holdout testing dataset. Results in Table 3 and Figure 5 are all from the testing. 

"SVI_gp-5d-sys-train.py" is for mapping the GP model systematic uncertainty as mentioned in Section 5.2. 

"section_gp-5d-predict.py" is the script for generating GP-prediction stellar models as described in section 6.1. 


To test any scripts or predict with learned GP models, you will need to download the GP models on 

https://drive.google.com/drive/folders/1USZE3wTjvVaS0VbCdhXFEOl9XsCJ2clw?usp=sharing

A GP-predicted model dataset (contents 500,000 models) is also available on 

https://drive.google.com/file/d/1JwHMHjIHUJmC9Kt7VgK2FVDokC7dNSWK/view?usp=sharing








 
