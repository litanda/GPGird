#!/usr/bin/env python
# coding: utf-8

# # Adopted training inputs:
# 
# Accroding to the tests, below are selected for traning GP models:
# 
# -  EEP factor: f = 0.18                      This number gives uniform distributions
# 
# -  Kernel: Mat32                              
# 
# - Normalization: p = (p - p.mean)/p.std      for both inputs and outputs
# 
# - NN mean funcition 6x128 elu                elu is smooth Relu is spicky
# 
# - optimizer: Adam with amsgrad=True          AdamW or Adamx are similar.
# 
# - likelihood function set noise = 0 
# 

# # The chose of EEP factor
# 
# For the data points on the same evolutionary track, we calculated the displacement of 
# a model (with the model number n) on keil diagram as 
# 
# d_n = ((Teff_{n} - Teff_{n-1})^2 + (logg_{n} - logg_{n-1})^2)^f
# 
# and then we calcaulate EEP as 
# 
# EEP_n = Sum(d_0 to d_n)
# 
# and we lastly normalized EP in the 0 - 1 range. 
# 
# The factor f is to adjust the EEP. When f is 0.5, we found a obvious gap in the parameter space between tracks
# with and without hooks. The gap leads poor precictions in this area. We adjusted f to squeeze tracks and found that 
# f = 0.18 gives the best predictions on 2D data. 

# In[1]:




# # S0: Loading packages

# In[2]:


import math
from matplotlib import pyplot as plt
import pandas as pd
import numpy as np
import os


# ! pip install ipywidgets 
# ! pip install jupyterlab
# ! jupyter nbextension enable --py widgetsnbextension
# ! jupyter labextension install jupyterlab-manager

# ! pip install tqdm

# In[3]:


import torch
import tqdm
import gpytorch
from torch.nn import Linear
from gpytorch.means import ConstantMean, LinearMean
from gpytorch.kernels import RBFKernel, ScaleKernel
from gpytorch.variational import VariationalStrategy, CholeskyVariationalDistribution
from gpytorch.distributions import MultivariateNormal
from gpytorch.models import ApproximateGP, GP
from gpytorch.mlls import VariationalELBO, AddedLossTerm
from gpytorch.likelihoods import GaussianLikelihood

from gpytorch.models.deep_gps import DeepGPLayer, DeepGP
from gpytorch.mlls import DeepApproximateMLL



#--------- def functions --------------

#normalizations
def normalize_a_df(gdf = None, df0= None, xcolumns = None, ycolumns = None):
    
    gg = gdf.copy()
    
    for cc in xcolumns + ycolumns:
        gg[cc] = (gdf[cc] - np.mean(df0[cc]))/np.std(df0[cc])
    return gg

# A seven layers NN for the mean function
class MLPMean(gpytorch.means.Mean):
    def __init__(self, dim):
        super(MLPMean, self).__init__()
        self.mlp = torch.nn.Sequential(
            torch.nn.Linear(dim, 128), 
            torch.nn.ELU(), 
            torch.nn.Linear(128, 128), 
            torch.nn.ELU(), 
            torch.nn.Linear(128, 128), 
            torch.nn.ELU(),    
            torch.nn.Linear(128, 128), 
            torch.nn.ELU(),  
            torch.nn.Linear(128, 128), 
            torch.nn.ELU(),  
            torch.nn.Linear(128, 1))

        count = 0
        for n, p in self.mlp.named_parameters():
            self.register_parameter(name = 'mlp' + str(count), parameter = p)
            count += 1
    
    def forward(self, x):
        m = self.mlp(x)
        return m.squeeze()
    
# Define a full GP model, we use same setting for all six parameters
class ExactGPModel(gpytorch.models.ExactGP):
    def __init__(self, train_x, train_y, likelihood):
        super(ExactGPModel, self).__init__(train_x, train_y, likelihood)
        
        input_dims = train_x.shape[-1]
        
        self.mean_module = MLPMean(input_dims)
        self.covar_module = gpytorch.kernels.ScaleKernel(
        gpytorch.kernels.MaternKernel(nu = 3/2, ard_num_dims=input_dims)
        )

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)
#----------------------------------------------------------


#----------------Here we star the main script--------------

os.system('nvidia-smi')
n_devices = torch.cuda.device_count()
print('Planning to run on {} GPUs.'.format(n_devices))

#set up the path
dr = 'subsections_GP_NNElu_Mat32_10sections/'
command = 'ls ' + dr
os.system(command)

#set up inputs and outputs
xcolumns = ["initial_mass","evol_scale","initial_feh","initial_Yinit","initial_MLT"]
ycolumns = ["effective_T","log_g","radius", "delta_nu_fit", "star_feh", "star_age"]
#ycolumns = ["effective_T"]

# load the reference data set for nomalization
dfref = pd.read_csv(dr + 'reference.csv')

# load a dataframe to predict

#filename = '../../gp_data_v3_scale/validation_5d.csv'
#validation = pd.read_csv(filename, index_col=0)
#nv = ( (validation.star_age <= 20.0) 
#      & (validation.star_feh >= -0.6) 
#      & (validation.evol_scale>=0.01) 
#      & (validation.evol_scale<=0.99)
#      & (validation.initial_mass<=1.19)
#      & (validation.initial_mass>=0.81)
#      & (validation.effective_T <=7000)
#     )
#tests = validation.sample(n = 20000, weights= 'hrgradient', random_state = 50000)
#tests.index = range(len(tests))
#tests.to_csv('5d_sys_data_n20K.csv')
#print('total input data size is ', len(validation[nv]), len(tests))
#exit()

tests = pd.read_csv(dr + 'testing/5d_tests_n50K.csv',index_col = 0)
tests = tests[25000:50000]
tests0 = tests.copy()

# for three demissions, divide into 5 sections:
nsections = 10  #5   #3
section = 0.1   #0.2   #0.333

for i in range(nsections):
    os.system('nvidia-smi')
    pptemp = []
    
    E0 = round( i*section, 1)
    E1 = round( (i+1)*section, 1)
    EEPrange = [E0, E1]
    print('Now working on EEP = ', EEPrange)
        
    label = 'EEP' + str(E0) + '-' + str(E1)
    savedr =  dr + label + '/'
    
    # select the section of tests
    nuse = (
        (tests.evol_scale >=EEPrange[0]) 
        & (tests.evol_scale <= EEPrange[1]) 
    )
    pred = tests[nuse]
    pred.index = range(len(pred))
    
    if len(pred) <= 10: 
        print(len(pred), ' testing data points')
        continue
    # normalized data for predicting
    pred_normal = normalize_a_df(gdf = pred, df0 = dfref, xcolumns = xcolumns, ycolumns = ycolumns)
    
    #gpmodels = [None]*len(ycolumns)
    #gplikelihoods = [None]*len(ycolumns)
    
    for ii in range(len(ycolumns)):
        yy = ycolumns[ii]
        print('loading models for ', yy)
        ylabel = label + yy
        
        # load tensors
        state_dict = torch.load(savedr + ylabel + '_model_state.pth')
        train_x = torch.load(savedr + ylabel + '_train_x.pt')
        train_y = torch.load(savedr + ylabel + '_train_y.pt') #torch.linspace(0, 1, 20000) # better to save and load true train_y
        
        # define exactly same model as training
        noise_constraint = gpytorch.constraints.Interval(torch.tensor(0.0001), 
                                                         torch.tensor(0.2),
                                                         initial_value=None)
        gplikelihoods = gpytorch.likelihoods.GaussianLikelihood(noise_constraint = noise_constraint)
        gpmodels = ExactGPModel(train_x, train_y,gplikelihoods)
        
        # load state dict
        gpmodels.load_state_dict(state_dict)
        
        # predicting results
        gpinputs = torch.tensor((pred_normal[xcolumns].values)).to(dtype=torch.float32)
        knownvalues = torch.tensor((pred_normal[yy].values)).to(dtype=torch.float32)
        
        if torch.cuda.is_available():
            train_x = train_x.cuda()
            train_y = train_y.cuda()
            gpinputs = gpinputs.cuda()
            knownvalues = knownvalues.cuda()
            gpmodels = gpmodels.cuda()
            gplikelihoods = gplikelihoods.cuda()
        
        gpmodels.eval()
        gplikelihoods.eval()
        with torch.no_grad(), gpytorch.settings.fast_pred_var():
            observed_pred = gplikelihoods(gpmodels(gpinputs))
            gpoutputs = observed_pred.mean
        
        #replace pred[yy]
        gpoutputs = gpoutputs.cpu().detach().numpy()
        
        ddy = gpoutputs - knownvalues.cpu().detach().numpy()
        print( np.percentile(np.abs(ddy*np.std(dfref[yy])), [68,95,99.8,100]) )
        
        gpoutputs = gpoutputs*np.std(dfref[yy]) + np.mean(dfref[yy])
        ddy = ddy*np.std(dfref[yy])
        tests.loc[nuse, yy] = ddy #gpoutputs
    
        if torch.cuda.is_available():
            train_x = train_x[0]
            train_y = train_y[0]
            gpinputs = train_x[0]
            knownvalues =  train_x[0]
            gpmodels = None
            gplikelihoods = None
            
        
        state_dict = train_x = train_y = gpinputs = gpoutputs =knownvalues = gpmodels = gplikelihoods = observed_pred = gpoutputs = None
        torch.cuda.empty_cache()
        
tests.to_csv(dr + 'testing/5d_tests_n50K_errors-2.csv')

exit()

for yy in ycolumns:
    print('validating ' , yy)
    
    dif = tests[yy] - tests0[yy]
    print( np.percentile(np.abs(dif), [68,95,99.8,100]) )
    
    fig, ax = plt.subplots()
    cp = ax.scatter(tests.initial_mass, tests.evol_scale, c = dif, s = 1)
    ax.set_xlabel(r'$M (M_{\odot})$')
    ax.set_ylabel(r'EEP')
    cc = plt.colorbar(cp)
    cc.set_label(yy)
    fig.savefig(dr + 'tests_' + yy + '_png')
exit()

    
    

   
    



# In[ ]:





# In[ ]:





# # 3D GP Full  Version 2 (Constrained likelihood + amsgrad)
# 
# 
# 
# 
# # R1: Mat32 NN mean  Nomalization -- std   20K Adam ELu   （！Winer！）
# 
# 
# effective_T   [ 2.29646424  6.76057799 14.70505026 20.43564034]
# 
# log_g  [0.00141013 0.00378277 0.01155004 0.01726359]
# 
# radius  [0.00249971 0.00686117 0.01727212 0.02468476]
# 
# delta_nu_fit  [0.19860151 0.59042628 2.24528293 3.70184517]
# 
# star_feh  [0.00153985 0.00371011 0.02146512 0.03333323]
# 
# star_age  [0.02045885 0.06808621 0.22584867 0.36360192]
# 
# # R2: Mat32 NN mean  Nomalization -- std   20K Adam ReLu
# 
# 
# effective_T   [ 2.53164098  6.80726402 18.11548222 26.01394081]
# 
# log_g  [0.00143676 0.00412744 0.01270682 0.01921595]
# 
# radius  [0.00269905 0.00788009 0.01795666 0.02620659]
# 
# delta_nu_fit  [0.20842729 0.5935468  2.2643608  3.91968465]
# 
# star_feh  [0.0013125  0.00532967 0.02878173 0.04907673]
# 
# star_age  [0.02702187 0.07352434 0.26129112 0.35672244]
# 

# # 2D GP Full  Version 2 (elu, fixed noise,  amsgrad=True)
# 
# 
# # R1: Mat32 NN mean  Nomalization -- std   20K Adam
# 
# effective_T [ 0.93295757  5.01417756 12.20306314 13.61799717]
# 
# log_g [0.00095927 0.00309138 0.00984617 0.01178575]
# 
# radius [0.00204354 0.00565071 0.01356371 0.01539518]          Pick! 
# 
# delta_nu_fit [0.11897697 0.39261544 1.19743541 1.33957326]    Pick!
# 
# star_feh [0.00070427 0.00231722 0.01242407 0.02140774]
# 
# star_age 
# 
# # R2: Mat32 NN mean  Nomalization -- std   20K Adam 
# 
# effective_T  [ 1.13865767  4.70341504 12.10525291 13.81997299]  Pick!
# 
# log_g [0.00114722 0.00299164 0.00780284 0.01139687]             Pick!
# 
# radius [0.00204081 0.00605936 0.0146233  0.01698701]            
# 
# delta_nu_fit  [0.14389355 0.42876455 1.22692903 1.41739714]
# 
# star_feh [0.000399   0.00174528 0.01273863 0.02258373]          Pick!
# 
# star_age [0.01174926 0.03518691 0.09458795 0.11461608]          Pick!
# 
# 
# ----------------------------
# 
# 

# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# # 2D GP Full  Version 1 (Relu, gaussian noise,  amsgrad=False)
# 
# # First run: Mat32 NN mean  Nomalization -- std   20K Adam
# 
# effective_T [ 0.96759338  4.43236172 10.87957911 12.83617783]
# 
# log_g  [0.00094074 0.0028698  0.00843479 0.00992017]
# 
# radius [0.00183108 0.00618609 0.01478621 0.01765038]
# 
# delta_nu_fit [0.13196098 0.39072995 1.16657602 1.57352757]
# 
# star_feh [0.00037131 0.00196537 0.01285748 0.02233051]
# 
# star_age [0.0108337  0.03333055 0.09971912 0.11078747]
# 
# 
# # R2: Mat32 NN mean  Nomalization -- std   20K SGD  （Not stable）
# 
# star_age [0.05756118 0.08396784 0.14917123 0.17526181]
# 
# # R3 Mat32 NN mean  Nomalization -- std   10K Adam
# 
# effective_T [ 1.28079859  4.91566479 10.66677462 13.00879765]
# 
# log_g  [0.00103417 0.00301479 0.00933594 0.01178567]
# 
# radius  [0.00187024 0.0059354  0.01521754 0.01742246]
# 
# delta_nu_fit [0.12346228 0.36663527 1.15084557 1.51223099]
# 
# star_feh   [0.00040162 0.00188361 0.01321657 0.0229958 ] 
# 
# star_age   [0.01229311 0.03595101 0.10800961 0.12928981]
# 
# # R4 Mat32 NN mean  Nomalization -- std   5K Adam
# 
# effective_T [ 1.09721192  4.60156138 10.60890373 12.61343193]
# 
# log_g  [0.00115729 0.00297832 0.0089735  0.01120964]
# 
# radius  [0.00186831 0.00569319 0.01640638 0.01995684]
# 
# delta_nu_fit [0.12019951 0.38473792 1.3113087  1.55754125]
# 
# star_feh    [0.00038643 0.00185756 0.01336201 0.0226854 ]
# 
# star_age   [0.01347767 0.04198609 0.10294061 0.12372085]
# 
# # R5 Mat32 NN mean  Nomalization -- std   2.5K Adam
# 
# effective_T  [ 1.41829939  4.87456636 12.29124973 14.72623062]
# 
# log_g [0.00113555 0.00336804 0.01009623 0.01165517]
# 
# radius  [0.00203875 0.00606484 0.01685285 0.0213025 ]
# 
# delta_nu_fit  [0.13201557 0.41976399 1.46152447 1.92335689]
# 
# star_feh [0.00038929 0.00203275 0.01291176 0.0232384 ]
# 
# star_age [0.01206109 0.04138095 0.12683855 0.17370345]
# 
# # R6 Mat32 NN mean  Nomalization -- std   5K Adam test best loss
# 
# effective_T  [ 1.0650389   4.52233644 12.0893996  13.08435917]
# 
# # R7 Mat32 NN mean  Nomalization -- std   10K Adam test best loss
# 
# effective_T  [ 1.05395553  4.21078889 11.99635136 13.89216518]
# 
# 

# # 2D SVI GP
# 
# 
# _____________________
# 
# # 2nd run: Mat32  NN Mean  Nomalization -- std   5Kx4   (winner)
# 
# # output    loss            validations (68/95/99.8/100%)
# 
# --------------------
# 
# effective_T  -2.880  [ 1.10209811  4.78847613 10.40704091 13.32463932]
# 
# log_g -2.93  [0.00089775 0.00274551 0.00973068 0.01259972]
# 
# radius 2.7197 [0.00215979 0.00615649 0.0171424  0.02490166]
# 
# delta_nu_fit -2.869 [0.12148838 0.39335734 1.47558728 1.82840705]
# 
# star_feh  -2.53  [0.00055028 0.00240294 0.01279394 0.02472778]
# 
# star_age  -2.889 [0.01128489 0.0363209  0.08722457 0.12526932] 
# 
# _____________________
# 
# # 3rd run: Mat32  NN Mean  Nomalization -- std   5Kx4 (new tests)
# 
# # output    loss            validations (68/95/99.8/100%)
# 
# --------------------
# 
# effective_T  [ 1.12235659  5.69439952 16.61109818 22.52091217]
# 
# log_g  [0.00118676 0.00366934 0.00906133 0.01131113]
# 
# radius [0.00229564 0.00630343 0.01458555 0.01742631]
# 
# delta_nu_fit [0.13251803 0.37603404 1.1101479  1.63607013]
# 
# star_feh  [0.00035802 0.00159173 0.01290824 0.02260517]
# 
# star_age  [0.01038994 0.03171899 0.10269298 0.16331328]
# 

# In[ ]:





# #  # testing EEP v3 (full GP)
# 
# Kernel        n    epoch    correlation   mean    validation errors 
# 
# ----------------------------
# 
# RBF+Mat12(SGD)10K     200       N     L      [ 6.02174124  8.89345779 16.76398705 24.78442192]
# 
# RBF+Mat12(SGD)20K     200       N     L      [ 2.19133428  5.97482758 11.99727036 13.97822762]   (winner)
# 
# RBF+Mat12   20K     200        N     L        [ 2.9977169   6.3573169  14.14301355 17.93074799]   
# 

# #  # testing EEP v3 (SVI GP)
# 
# Kernel       opt      n       mean    epoch    validation errors 
# 
# Mat32(0-2)  Adam      10000x1K   L      1000   Not good
# 
# Mat32(0-1)  Adam      10000x1K   L      1000    [ 3.71154255 13.75717306 35.12444675 47.88844681]
# 
# Mat32(5sig) Adam      10000x1K     L     1000    [ 3.19696237 12.58400488 34.75011889 46.79726028]
# 
# Mat32(.5sig) Adam      10000x1K     L     1000    [ 0.85029869  4.99200447 16.72496518 20.423172  ] !!!
# 
# Mat32      Adam      10000x1K     L     1000    [ 1.52312168  7.55758741 23.58297765 30.97191811]
# 
# Mat32      Adam      5000Kx2     L     1000      [ 1.45902501  7.20362151 23.56091354 30.63788605]
# 
# Mat32      Adam      5000Kx4      L      1000   [ 1.32396966  6.9170069  23.63270625 31.34643364]
# 
# ----------------------------
# RBF+Mat12   Adam     10Kx2      L      1000   [ 1.13966137  5.89335251 18.85964389 25.51701164]
# 
# RBF+Mat12   Adam     5Kx4      L      1000    [ 1.0072931   5.31334407 16.40474388 21.66530991] !!!!!
# 
# RBF+Mat12(.6sig)   Adam     5Kx4      L      1000    [ 0.93515253  4.93021736 15.1582718  20.27057457]
# 
# RBF+Mat12(.5sig)   Adam     5Kx4      L      1000    [ 0.93237593  4.78563373 13.80759848 17.8200531 ]  !!!
# 
# RBF+Mat12(.4sig)   Adam     5Kx4      L      1000    [ 0.88178204  4.75576808 14.31413676 18.52410507]
# 
# RBF+Mat12(.25sig)   Adam     5Kx4      L      1000    [ 0.99262313  5.18757887 16.93955814 22.56874466]
# 
# RBF+Mat12   Adam     2Kx10     L      1000    [ 1.1415282   5.99054408 20.23959143 26.70519447] 
# 
# RBF+Mat12   SGD      5Kx4      L      1000    local minimum
# 
# RBF+Mat12   Adadelta 5Kx4      L      1000     [ 8.46006645 22.08966618 58.04372513 74.06304932]
# 
# RBF+Mat12   Adagrad 5Kx4      L      1000     [ 1.17297339  5.07752006 14.61492423 34.55992126]
# 
# RBF+Mat12   AdamW   5Kx4      L      1000     [ 1.21857575  5.79373729 15.11036374 18.8704586 ] !!!!!
# 
# RBF+Mat12  SparseAdam 5Kx4      L      1000    Not work
# 
# RBF+Mat12  Adamax 5Kx4      L      1000       [ 1.00683134  5.38465788 17.96592408 23.76205063]
# 
# RBF+Mat12  ASGD   5Kx4      L      1000        local minimum
# 
# ------------------------------
# 
# Mat12+Mat12    Adam     5Kx4      L      1000    [ 1.04214651  5.35429125 18.63718993 24.77742195]
# 
# Mat12+Mat12    SGD     5Kx4       L      1000    local minimum

# # testing EEP v0 (single layer)
# 
# Kernel           batch     factor      validation errors 
# 
# ----------------------------
# Mat12+Mat12     1000x20      1.0     [ 1.16455216  5.57278082 31.21558549 42.22875595]
# 
# Mat12+Mat12     1000x20      0.9     [ 1.132394    5.40005956 31.51196307 42.5899086 ]
# 
# Mat12+Mat12     1000x20      0.6     [ 0.98082321  5.07503567 31.3280181  43.22613907]
# 
# Mat12+Mat12     1000x20      0.5     [ 0.96657107  5.07524681 31.59414326 44.00315857] (best is 0.5)
# 
# Mat12+Mat12     5000x4       0.5     [ 2.54955319  5.4041183  17.9130042  31.37347603] (not finish)
# 
# Mat12+Mat12     1000x20      0.4     [ 0.98032125  4.97662184 32.98244475 44.81765747]
# 
# Mat12+Mat12     1000x20      0.3     [ 0.96261914  5.10110865 32.45037873 45.1966095 ]
# 
# Mat12+Mat12     1000x20      0.1     [ 1.08325098  4.94316452 33.66598858 48.1369133 ]
# 
# ------------------------------
# 
# Mat32           1000x20      1.0     [ 2.04635368  6.93017817 33.44620732 42.7049408 ]
# 
# Mat32           1000x20      0.9     [ 1.82103952  6.65440857 35.48153362 45.68404388]   
# 
# Mat32           1000x20      0.8     [ 1.78605962  6.72208891 35.2791497  45.40929794]
# 
# Mat32           1000x20      0.7     [ 1.63050709  6.57714787 35.12205529 46.14549255]
# 
# Mat32           1000x20      0.6     [ 1.51192863  6.58667305 34.24997946 45.02347183]
# 
# Mat32           1000x20      0.5     [ 1.45996469  6.58308055 34.88194438 45.91796875]
# 
# Mat32           5000x4       0.5     [ 0.90629964  5.60191102 29.23826544 41.38313293]

# In[ ]:





# In[ ]:





# In[ ]:





# # single layer
#  
# Kernel| batch size|   validation errors 68/95/99.8/100
# 
# ----------------------------------------------------------------------
# RBF          |1000x20| [ 3.92076566 14.96708269 33.54362242 41.78662109]
# 
# RBF          |5000x4|  [ 2.53184409 10.3714159  28.66574593 38.96194458]
# 
# ----------------------------------------------------------------------
# RQ           |1000x20| [ 3.68881349 15.3627049  32.43054482 36.69728088]
# 
# ----------------------------------------------------------------------
# Mat12        |1000x20| [ 2.11455879  7.14621909 23.32556133 26.28586578]
# 
# ----------------------------------------------------------------------
# Mat32        |1000x20| [ 2.13035681  7.70045154 23.73776591 31.42520332]
# 
# ---------------------------------------------------------------------
# Mat52        |1000x20| [ 2.2266041   7.84246294 26.44035172 32.7101059 ]
# 
# ----------------------------------------------------------------------    
# RBF+Mat12    |1000x20| [2.02574305  7.78611422 22.34349648 26.89947128]
# 
# ----------------------------------------------------------------------
# RBF+Mat32    |1000x20| [ 2.16959077  8.25448952 25.36939505 32.20012283]
# 
# ----------------------------------------------------------------------
# RQ+Mat12     |1000x20| [ 2.05316819  8.00381413 23.21213258 32.31229401]
# 
# ----------------------------------------------------------------------
# RQ+Mat32     |1000x20| [2.24384446  8.08762126 25.4185765  32.19676971]
# 
# ----------------------------------------------------------------------
# Mat12+Mat12  |1000x20| [ 1.87615819  7.02385864 22.41348807 28.40757561]
# 
# Mat12+Mat12  |5000x4|  [1.60563675  5.89649556 17.99046997 21.41035843]
# 
# Mat12+Mat12  |9000x2|  [ 1.34354727  4.97922111 15.02489474 19.35504532]
# 
# ----------------------------------------------------------------------
# 
# RBF*RBF      |1000| [3.17548359 13.98570042 31.57718209 37.13195419]
# 
# 
# # two layers
# 
# -----------------------
# 
# RBF/RBF        |1000x20| [ 3.17988528 10.79575248 28.4412952  40.07410431]
# 
# ----------------------
# 
# RQ/RQ          |1000x20| [ 2.3802446   7.56052713 23.01092806 30.33405495]
# 
# RQ/Mat12       |1000x20|
# 
# RQ/Mat32       |1000x20| 
# 
# ---------------------
# 
# Mat12/Mat12    |1000x20|  [ 2.23476944  6.1830411  18.55695063 25.54027557]
# 
# Mat12+Mat12/RBF   |1000x20|  [ 2.37181245  6.76484778 18.830227   25.1174221 ]
# 
# Mat12+Mat12/Mat12| 1000x20|   [ 2.42582632  6.7309927  19.44204641 25.721632  ]
# 
# ------------------
# 
# Mat32/Mat32    |1000x20|  ??
# 
# -------------------
# 
# 

# In[ ]:





# In[ ]:





# In[ ]:




