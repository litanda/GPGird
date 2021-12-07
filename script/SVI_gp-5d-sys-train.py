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

# In[4]:


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


# In[18]:

os.system('nvidia-smi')



n_devices = torch.cuda.device_count()
print('Planning to run on {} GPUs.'.format(n_devices))


# In[ ]:





# # S1: Set up saving path, Label the training, and set up datasets

# In[5]:


os.system('ls subsections_GP_NNElu_Mat32_10sections/systematics/')


# In[6]:


label = '5d_sys_'
savedr =  'subsections_GP_NNElu_Mat32_10sections/systematics/'


# In[7]:


ntraining = 100000
nvalidation = 20000

xcolumns = ["initial_mass","evol_scale","initial_feh"]
ycolumns = ["log_g_std","radius_std", "delta_nu_fit_std", "star_feh_std", "star_age_std"] #"effective_T_std",


# In[8]:


df = pd.read_csv(savedr + label + 'std.csv',index_col = 0)

gdf = df.sample(n = 32000, random_state = 1000000)
vdf = df.drop(gdf.index)
gdf.index = range(len(gdf))
vdf.index = range(len(vdf))

nuse = (df.initial_feh == 0)
pdf = df[nuse]
pdf.index= range(len(pdf))

batch_size = round(len(gdf)/10)

print(len(gdf), batch_size)

# In[11]:


#normalizations
def normalize_a_df(gdf = None, df0= None, xcolumns = None, ycolumns = None):

    gg = gdf.copy()

    for cc in xcolumns + ycolumns:
        gg[cc] = (gdf[cc] - np.mean(df0[cc]))/np.std(df0[cc])
    return gg


gdf = normalize_a_df(gdf = gdf, df0= df, xcolumns = xcolumns, ycolumns = ycolumns)
vdf = normalize_a_df(gdf = vdf, df0= df, xcolumns = xcolumns, ycolumns = ycolumns)
pdf = normalize_a_df(gdf = pdf, df0= df, xcolumns = xcolumns, ycolumns = ycolumns)

# In[16]:


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


# # Define a SVI GP model, we use same setting for all six parameters

# In[17]:


from gpytorch.models import ApproximateGP
from gpytorch.variational import CholeskyVariationalDistribution
from gpytorch.variational import VariationalStrategy

class GPModel(ApproximateGP):
    def __init__(self, inducing_points):
        variational_distribution = CholeskyVariationalDistribution(inducing_points.size(0))
        variational_strategy = VariationalStrategy(self, inducing_points, variational_distribution, learn_inducing_locations=True)
        super(GPModel, self).__init__(variational_strategy)

        input_dims = train_x.shape[-1]

        #self.mean_module = MLPMean(input_dims)
        self.mean_module = gpytorch.means.ConstantMean()
        self.covar_module = gpytorch.kernels.ScaleKernel(
            gpytorch.kernels.RBFKernel(ard_num_dims=input_dims)
        )

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)


# In[19]:


def traing_SVIGPModel(num_epochs = 1000,
                      lr0 = 0.01,
                      validate = False,
                      ylabel = 'yy',
                      best_loss = 9999,
                      loss_cut = -2.0,
                      model0 = False):

    #  here we define some paramters

    num_epochs = num_epochs
    lr0 = lr0
    t = 0
    tr = 0
    vali_error = 0.0
    dds0 = 0.0
    min_loss = 9999

    model.train()
    likelihood.train()

    # We use Adam here
    optimizer = torch.optim.Adam([
        {'params': model.parameters()},
        {'params': likelihood.parameters()},
    ], lr=lr0, amsgrad=True)

    # Our loss object. We're using the VariationalELBO
    mll = gpytorch.mlls.VariationalELBO(likelihood, model, num_data=train_y.size(0))

    #epochs_iter = tqdm.notebook.tqdm(range(num_epochs), desc="Epoch")

    for i in range(num_epochs):
        # Within each iteration, we will go over each minibatch of data
        #minibatch_iter = tqdm.notebook.tqdm(train_loader, desc="Minibatch", leave=False)

        #print(minibatch_iter)

        for x_batch, y_batch in train_loader:
            optimizer.zero_grad()
            output = model(x_batch)
            loss = -mll(output, y_batch)
            #minibatch_iter.set_postfix(loss=loss.item())
            loss.backward()
            optimizer.step()

        # for moniting the training process
        if ((i%1)==0 and (loss.item() < loss_cut)):

            t = t + 1

            if validate == True:
                model.eval()
                likelihood.eval()
                with torch.no_grad(), gpytorch.settings.fast_pred_var():
                    latent_pred = model(vali_x)
                    test_rmse = torch.sqrt(torch.mean(torch.pow(latent_pred.mean - vali_y, 2)))

                    plot_data = likelihood(model(plot_x))
                    plot_c = plot_data.mean
                    plot_c = plot_c.cpu()
                    fig, ax = plt.subplots(1,2, figsize = (20,10))
                    cp = ax[0].scatter(pdf.initial_mass, pdf.evol_scale, c = plot_c.detach().numpy(), s = 50, cmap = 'rainbow')
                    ax[1].scatter(pdf.initial_mass, pdf.evol_scale, c = pdf[yy], s = 50, cmap = 'rainbow')
                    plt.colorbar(cp)
                    fig.savefig(savedr + ylabel + '_quick_plot.png')
                    plt.close()

                model.train()
                likelihood.train()

                if ((best_loss - test_rmse.item()) >= 1.0e-3):
                    best_loss = test_rmse.item()
                    torch.save(model.state_dict(), savedr + ylabel + '_model_state.pth')
                    torch.save(likelihood.state_dict(), savedr + ylabel + '_likelihhod_state.pth')
                    t = 0

        if ((i%10)==0):
            print(i,loss.item(), best_loss, t, tr, lr0, likelihood.noise.item())

        # for adjusting learning rate

        if ((loss.item() < min_loss)):
            min_loss = loss.item()
            tr = 0
        else:
            tr = tr + 1

        if (tr>20):
            lr0 = lr0/2.0
            print('lr change to', lr0)
            optimizer = torch.optim.Adam([
                {'params': model.parameters()},
                {'params': likelihood.parameters()}, # Includes GaussianLikelihood parameters
            ], lr=lr0, amsgrad=True)
            tr = 0

        # terminate when no improvement for 400 steps

        if (tr > 50) or (lr0 < 1.0e-4):
            print('no improvements in the last 30 steps or learning rate are too small')

            torch.save(model.state_dict(), savedr + ylabel + '_model_stateX.pth')
            torch.save(likelihood.state_dict(), savedr + ylabel + '_likelihhod_stateX.pth')

            break

    return best_loss


# In[ ]:


# GP individual parameters:
#["effective_T","log_g","radius","delta_nu_fit", "star_feh", "star_age"]

yys = ycolumns

for ii in range(len(yys)):

    yy = yys[ii]
    loss_cut = 99.0

    print('start training ', yy, '| early stop activates at loss = ',  loss_cut)

    num_epochs = 5000
    lr0 = 0.01
    lr1 = 0.01
    validate = True
    ylabel = label + yy

    train_x = torch.tensor((gdf[xcolumns].values)).to(dtype=torch.float32)
    train_y = torch.tensor((gdf[yy].values)).to(dtype=torch.float32)

    vali_x = torch.tensor((vdf[xcolumns].values)).to(dtype=torch.float32)
    vali_y = torch.tensor((vdf[yy].values)).to(dtype=torch.float32)

    plot_x = torch.tensor((pdf[xcolumns].values)).to(dtype=torch.float32)

    if torch.cuda.is_available():
        train_x = train_x.cuda()
        train_y = train_y.cuda()
        vali_x = vali_x.cuda()
        vali_y = vali_y.cuda()
        plot_x = plot_x.cuda()


    from torch.utils.data import TensorDataset, DataLoader
    train_dataset = TensorDataset(train_x, train_y)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)


    inducing_points = train_x[:10000, :]
    model = GPModel(inducing_points=inducing_points)

    noise_constraint = gpytorch.constraints.Interval(torch.tensor(0.0),
                                                     torch.tensor(0.1),
                                                     initial_value=None)
    likelihood = gpytorch.likelihoods.GaussianLikelihood()#noise_constraint = noise_constraint)
    #likelihood = gpytorch.likelihoods.GaussianLikelihood()

    torch.save(inducing_points, savedr + ylabel + '_inducing_points.pt')


    if torch.cuda.is_available():
        model = model.cuda()
        likelihood = likelihood.cuda()

    # training GP model

    best_loss1 = traing_SVIGPModel(num_epochs = num_epochs,
                                   lr0 = lr0,
                                   validate = True,
                                   ylabel = ylabel,
                                   best_loss = 9999,
                                   loss_cut = loss_cut,
                                    model0 = False)


    #### remove all data

    if torch.cuda.is_available():
        train_x = None
        train_y = None
        vali_x = None
        vali_y = None
        model = None
        likelihood = None
        plot_x = None
        torch.cuda.empty_cache()

    train_x = None
    train_y = None
    vali_x = None
    vali_y = None
    model = None
    likelihood = None


# In[ ]:


train_x = None
train_y = None
vali_x = None
vali_y = None
model = None
likelihood = None


# # 2D SVI GP version2  (Constrained likelihood + amsgrad)
#
# _____________________
#
# # R1: Mat32  Linear Mean  Nomalization -- std   2Kx10   ELU
#
# # output    loss            validations (68/95/99.8/100%)
#
#
# effective_T
#
# log_g
#
# radius
#
# delta_nu_fit
#
# star_feh
#
# star_age
#

# In[ ]:





# In[ ]:
