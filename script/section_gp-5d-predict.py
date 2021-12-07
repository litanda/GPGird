# # S0: Loading packages

import math
from matplotlib import pyplot as plt
import pandas as pd
import numpy as np
import os

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

# make a dataframe to GP
def make_a_flat_df(xcolumns = None, xranges= None, ycolumns = None, n = None):
    
    xt = []
    
    for ii in range(len(xcolumns)):
        
        xx = xcolumns[ii]
        rr = xranges[ii]

        data = np.random.uniform(low=rr[0], high=rr[1], size=n)
        dd = {xx: data}
        
        tdf = pd.DataFrame(data = dd)
        xt.append(tdf)
    
    for ii in range(len(ycolumns)):
        
        yy = ycolumns[ii]

        data = np.full(n, -9999.99)        
        dd = {yy: data}
        
        tdf = pd.DataFrame(data = dd)        
        xt.append(tdf)
    
    df = pd.concat(xt, axis=1)
    return df

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

# Define a SVI GP model for the systematical uncertainty
from gpytorch.models import ApproximateGP
from gpytorch.variational import CholeskyVariationalDistribution
from gpytorch.variational import VariationalStrategy

class GPModel(ApproximateGP):
    def __init__(self, inducing_points):
        variational_distribution = CholeskyVariationalDistribution(inducing_points.size(0))
        variational_strategy = VariationalStrategy(self, inducing_points, variational_distribution, learn_inducing_locations=True)
        super(GPModel, self).__init__(variational_strategy)
        
        input_dims = 3
        
        self.mean_module = gpytorch.means.ConstantMean()
        self.covar_module = gpytorch.kernels.ScaleKernel(
            gpytorch.kernels.RBFKernel(ard_num_dims=input_dims)
        )

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)
    
#----------------------------------------------------------




#----------------Here we star the main script--------------
os.system('nvidia-smi')

train_on = 'cuda'

device = torch.device(train_on)  #cuda cpu

#set up the path
dr = 'subsections_GP_NNElu_Mat32_10sections/'
command = 'ls ' + dr
os.system(command)

#set up inputs and outputs
xcolumns = ["initial_mass","evol_scale","initial_feh","initial_Yinit","initial_MLT"]
ycolumns = ["effective_T","log_g"]#,"radius", "star_feh", "star_age"]
xcolumns_sys = ["initial_mass","evol_scale","initial_feh"]
ycolumns_sys = ["effective_T_std"]#,"log_g_std","radius_std", "star_feh_std", "star_age_std"]

# load the reference data set for nomalization
dfref = pd.read_csv(dr + 'reference.csv')

#-----------------------------------------------------------
#make a dataframe to GP

saveto = 'subsections_10sections_predictions/m-demission.csv' 


xranges = [[0.8,1.2], [0.001,0.999], [0.0,0.0], [0.28,0.28], [2.1,2.1]]
tests = make_a_flat_df(xcolumns, xranges , ycolumns + ycolumns_sys, n = 40000)


#----------------------------------------------------------
# predict outputs
nsections = 10
section = 0.1

for i in range(nsections):
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
    
    for ii in range(len(ycolumns)):
        yy = ycolumns[ii]
        print('loading models for ', yy)
        ylabel = label + yy
        
        # load tensors
        state_dict = torch.load(savedr + ylabel + '_model_state.pth', map_location=device)
        train_x = torch.load(savedr + ylabel + '_train_x.pt', map_location=device)
        train_y = torch.load(savedr + ylabel + '_train_y.pt', map_location=device) 
        
        # define exactly same model as training
        noise_constraint = gpytorch.constraints.Interval(torch.tensor(0.0001), 
                                                         torch.tensor(0.2),
                                                         initial_value=None)
        gplikelihoods = gpytorch.likelihoods.GaussianLikelihood(noise_constraint = noise_constraint)
        gpmodels = ExactGPModel(train_x, train_y,gplikelihoods)
        gplikelihoods.to(device)
        gpmodels.to(device)
        
        # load state dict
        gpmodels.load_state_dict(state_dict)
        
        # predicting results
        gpinputs = torch.tensor((pred_normal[xcolumns].values)).to(dtype=torch.float32)
        gpinputs.to(device)
        
        if train_on == 'cuda': gpinputs = gpinputs.cuda()
               
        gpmodels.eval()
        gplikelihoods.eval()
        with torch.no_grad():
            observed_pred = gplikelihoods(gpmodels(gpinputs))
            gpoutputs = observed_pred.mean
        
        gpoutputs = gpoutputs.detach().cpu().numpy()
        gpoutputs = gpoutputs*np.std(dfref[yy]) + np.mean(dfref[yy])
        tests.loc[nuse, yy] = gpoutputs
    
        state_dict = train_x = train_y = gpinputs = gpoutputs =knownvalues = gpmodels = gplikelihoods = observed_pred = gpoutputs = None
        torch.cuda.empty_cache() 
########################

    print('working on systemtical uncertainty')

    sysdr =  dr +'systematics/'
    syslabel = '5d_sys_'
    dfsys = pd.read_csv(sysdr + '5d_sys_std.csv')
    
    pred_normal2 = normalize_a_df(gdf = pred, df0 = dfsys, xcolumns = xcolumns_sys, ycolumns = ycolumns_sys)
        
    for jj in range(len(ycolumns_sys)):
        yy = ycolumns_sys[jj]
        print('loading models for ', yy)
        ylabel = syslabel + yy
    
        inducing_points = torch.load(sysdr + ylabel + '_inducing_points.pt', map_location=device)   
        model_state = torch.load(sysdr + ylabel + '_model_state.pth', map_location=device) 
        likelihood_state = torch.load(sysdr + ylabel + '_likelihhod_state.pth', map_location=device) 
    
        model = GPModel(inducing_points=inducing_points)
        likelihood = gpytorch.likelihoods.GaussianLikelihood()
        model.load_state_dict(model_state)
        likelihood.load_state_dict(likelihood_state)
        likelihood.to(device)
        model.to(device)
    
        gpinputs = torch.tensor((pred_normal2[xcolumns_sys].values)).to(dtype=torch.float32) 
        if train_on == 'cuda': gpinputs = gpinputs.cuda()

        model.eval()
        likelihood.eval()
        with torch.no_grad(): #, gpytorch.settings.fast_pred_var():
            observed_pred = likelihood(model(gpinputs))
            gpoutputs = observed_pred.mean
            gpoutputs = gpoutputs.detach().cpu().numpy()
            gpoutputs = gpoutputs*np.std(dfsys[yy]) + np.mean(dfsys[yy])
            tests.loc[nuse, yy] = gpoutputs
            
        model_state =likelihood_state= inducing_points = gpinputs = gpoutputs =knownvalues = model = likelihood = observed_pred = gpoutputs = None
        torch.cuda.empty_cache() 
#--------------------------------
        
tests.to_csv(saveto)

exit()




