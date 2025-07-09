"""
This script demonstrates the application of Principal Component Analysis (PCA) 
to analyze synchronization in coupled nonlinear oscillators, specifically
 Rossler systems. 

**Background:**  
Synchronization of chaotic and nonlinear systems is a fundamental topic in 
nonlinear dynamics, with numerous applications in neuroscience, physics, and 
engineering. Traditional measures of synchronization include phase coherence 
and correlation functions. 

In this work, we utilize PCA — a data-driven dimensionality reduction technique
— to analyze the joint behavior of two coupled Rossler oscillators. 
By simulating the systems over a range of coupling strengths, the explained 
variance ratio from PCA provides a quantitative metric of synchronization: as 
the oscillators become more synchronized, their collective dynamics are 
captured by fewer principal components, leading to a higher explained variance 
ratio.

This approach offers an alternative perspective for assessing synchronization, 
especially valuable in experimental data where traditional phase extraction 
may be challenging. The code attached demonstrates this method, which can be 
applied to various datasets.

**Note:**  
This code is part of a previous publication, and we are sharing it upon request 
to facilitate its application to your data.

@author: Henrique Castro
"""
#%%----------------------------------------------------------------------------
# Imports
#------------------------------------------------------------------------------
import numpy as np
import matplotlib.pyplot as plt
from numba import njit

import pandas as pd

from sklearn.decomposition import PCA

from multiprocessing import Pool

#%%----------------------------------------------------------------------------
# Auxiliary Functions
#------------------------------------------------------------------------------
# Mean Phase Coherence for comparative purposes
def mean_coherence(x):
    '''
    Calculates the mean phase coherence between two oscillators.

    Dependency:
        Requires the 'control' library to compute the 'unwrap' function for 
        phase unwrapping.

    Parameters:
    ----------
    x : ndarray (6 x n)
        State space time series for the two oscillators; first 3 rows for 
        oscillator 1, next 3 for oscillator 2.

    Returns:
    -------
    phis : ndarray (2 x n)
        Phase time series for each oscillator.
    mpc : float
        Mean phase coherence value between the two oscillators.
        
    Notes:
    ------
    The phase difference is computed from the arctangent of the derivatives of 
    the state variables.
    '''
    # Compute phase differences for each oscillator
    from control import unwrap
    def phase_diff(_x):
        p = np.arctan2(np.diff(_x[1,:]),np.diff(_x[0,:]))
        # Unwrap phase to avoid discontinuities
        phi = unwrap(p,period = np.pi*2)
        return phi
    
    # Calculate mean phase coherence between the two phase series
    def coherence(phi11):
        N = len(phi11)
        aux = np.exp(phi11*1j)
        R = np.abs(np.sum(aux)/N)
        return R
    
    # Compute phase time series for each oscillator
    phis = []
    for i in range(2):
        # Extract corresponding 3 variables for each oscillator
        phis.append(phase_diff(x[i*3:i*3+3]))
    phis = np.array(phis)
    
    return phis,coherence(phis[0]-phis[1])
#------------------------------------------------------------------------------
# Interquartile shaded plot
def plotGV(ms,gs_data,color,ax):
    '''
    Plots the median and interquartile range of a dataset over a dependent 
    variable.
 
    Parameters:
    ----------
    ms : array
        X-axis values (e.g., coupling strength).
    gs_data : DataFrame
        Descriptive statistics (mean, quartiles) for shading.
    color : str
        Color for lines and shading.
    ax : matplotlib.axes.Axes
        Axes object to plot on.
 
    Returns:
    -------
    None
    '''
    # Describe the data to extract quartiles
    gs_description = gs_data.describe()
    median = gs_description.loc['50%']
    # Plot median line
    ax.plot(ms, median, color=color)
    # Fill interquartile range
    ax.fill_between(ms, gs_description.loc['25%'], 
                      gs_description.loc['75%'], alpha=0.35,color=color)

#%%----------------------------------------------------------------------------
# Functions to Prepare Experimental Data Set
#------------------------------------------------------------------------------
# Numerical Integration
@njit
def rungeKutta(x0,h,dvFunc,*args):
    '''
    Performs a single integration step using the classical 4th order 
    Runge-Kutta method.
 
    Parameters:
    ----------
    x0 : array
        Current state variables of the system.
    h : float
        Integration step size.
    dvFunc : function
        Function defining the system's differential equations (dynamics).
    *args : additional parameters
        Parameters required by the differential equations function.
 
    Returns:
    -------
    x : array
        Updated state variables after one integration step.
    '''
    # 1a chamada
    xd=dvFunc(x0,*args)
    savex0=x0
    phi=xd
    x0=savex0+0.5*h*xd
    # 2a chamada
    xd=dvFunc(x0,*args)
    phi=phi+2*xd
    x0=savex0+0.5*h*xd
    # 3a chamada
    xd=dvFunc(x0,*args)
    phi=phi+2*xd
    x0=savex0+h*xd
    # 4a chamada
    xd=dvFunc(x0,*args)
    k = (phi+xd)/6
    x=savex0+k*h
    return x
#------------------------------------------------------------------------------
# Rossler Oscillator
@njit
def Rossler(x,w,a,b,c,u):
    '''
    Computes the derivatives for the Rossler system with coupling input.

    Parameters:
    ----------
    x : array
        Current state variables [x, y, z].
    w : float
        Mismatch parameter, affecting the system's frequency.
    a, b, c : float
        System parameters defining the Rossler oscillator dynamics.
    u : float
        External coupling signal influencing the x-variable.

    Returns:
    -------
    xd : ndarray
        Derivatives [dx/dt, dy/dt, dz/dt] at the current state.
    '''
    # Extract state variables for clarity
    _x = x[0]
    _y = x[1]
    _z = x[2]
    
    # Initialize the array for derivatives
    xd = np.empty(3)
    
    # Differential equations of the Rossler system with external coupling
    xd[0] = -w*_y -_z + u     # x' equation with coupling input
    xd[1] = w*_x + a*_y       # y' equation
    xd[2] = b + _z*(_x - c)   # z' equation
    return xd
#------------------------------------------------------------------------------
# System simulation
@njit
def simulate(dvFunc,h,x1,x2,kappa,delta,tf,a,b,c):
    '''
    Parameters
    ----------
    dvFunc : function - simulated system (Rossler)
    h      : float    - integration step
    x1,x2  : array    - initial conditions
    kappa  : float    - coupling strength
    delta  : float    - parameter mismatch
    tf     : integer  - simulation time (in steps)
    a,b,c  : float    - system parameters

    Returns
    -------
    _x1,_x2 : array(3xn) - state space time series

    '''
    # simulation steps
    n = tf+20000
    # initial conditions
    _x1=np.zeros((3,n))
    _x1[:,0] = x1
    _x2=np.zeros((3,n))
    _x2[:,0] = x2
    # integration loop
    for k in range(1,n):
        # calculate coupling signal
        u1 = kappa*(_x2[0,k-1]-_x1[0,k-1])
        u2 = kappa*(_x1[0,k-1]-_x2[0,k-1])
        # integration step
        _x1[:,k] = rungeKutta(_x1[:,k-1],h,dvFunc,1+delta,a,b,c,u1)
        _x2[:,k] = rungeKutta(_x2[:,k-1],h,dvFunc,1-delta,a,b,c,u2)
    # remove transient
    _x1 = _x1[:,20000:]
    _x2 = _x2[:,20000:]
    return _x1,_x2
#%%----------------------------------------------------------------------------
# Set up the experiment
#------------------------------------------------------------------------------
def experiment(kappa):
    '''
    Runs a single simulation experiment for given coupling strength 'kappa'.
    The simulation involves generating the time series of two coupled Rossler 
    oscillators, performing PCA on their combined state data, and computing 
    synchronization metrics.

    Parameters:
    ----------
    kappa : float
        The coupling strength between the two oscillators. Determines the level 
        of interaction.

    Returns:
    -------
    tuple:
        kappa : float
            The input coupling strength, for reference.
        cohe : float
            The average phase coherence between the oscillators, indicating 
            their phase synchronization.
        eVar : float
            The total explained variance ratio from PCA, serving as a measure 
            of the shared dynamics/synchronization.

    Notes:
    ------
    - The function initializes system parameters and simulates the coupled oscillators over a fixed duration.
    - Performs PCA on the combined trajectories; the resulting explained variance ratio reflects the degree of shared dynamics.
    - Computes mean phase coherence for an alternative synchronization measure.
    '''
    # Define system parameters: for Rossler oscillators
    a, b, c = 0.160, 0.2, 10
    
    # Integration step size
    h = 0.01
    
    # Total simulation time (number of steps)
    tf = 40000
    
    # Parameter mismatch between oscillators
    delta = 0.02
    
    # Initialize the oscillators' state with a random value
    x1 = np.random.rand()
    
    # Run the system simulation:
    # - Simulates two coupled Rossler oscillators with specified parameters
    # - The coupling strength 'kappa' influences the interaction terms
    x1,x2 = simulate(dvFunc=Rossler,h=h,x1=x1,x2=x1,
                      kappa=kappa,delta=delta,
                      tf=tf,a=a,b=b,c=c)
    
    # Combine the two oscillator trajectories into a single matrix
    x = np.vstack((x1,x2))
    # 'x' now contains the state variables over time for both oscillators
    
    # Set up PCA for dimensionality reduction in the combined data
    # - n_components refers to the dimension of the latent subspace.
    # - 'n_components=3' since each oscillator has 3 state variables
    _pca = PCA(n_components=3,random_state=0)
    
    # Fit PCA model to the data and transform to latent space
    # - Transpose 'x' because PCA expects samples as rows
    X_lat = _pca.fit_transform(x.T).T
    # 'X_lat' contains principal components (latent variables) over time
    
    # Calculate total explained variance ratio across selected components
    # - Acts as an indicator of how much synchronized are the oscillators
    eVar = np.sum(_pca.explained_variance_ratio_)
    
    # Compute mean phase coherence as an alternative measure to PCA
    # - Provides a phase-based synchronization metric
    phis,cohe = mean_coherence(x)
    
    # Return the results:
    # - 'kappa' for reference
    # - 'cohe': mean phase coherence between oscillators
    # - 'eVar': PCA explained variance ratio
    return kappa,cohe,eVar
#%%----------------------------------------------------------------------------
# Main execution block
#------------------------------------------------------------------------------
if __name__ == '__main__':
    # Define the range of coupling strengths (kappas) to evaluate
    # Using a rounded array from 0 to 0.07 in steps of 0.003
    kappas = np.round(np.arange(0,0.073,0.003),3)
    
    # Initialize a list to store results from experiments
    result = []
    
    # Use multiprocessing Pool to parallelize simulations over different kappas
    with Pool() as p:
        # Run multiple batches; here, 5 iterations
        for i in range(5):
            print('=== Batch', i+1, '===')
            # Map the experiment function over all kappa values in parallel
            result.extend(p.map(experiment, kappas))
            
        # Convert the results list into a pandas DataFrame for easier handling    
        result = pd.DataFrame(result)
        # Name the columns for clarity
        result.columns = ['kappa','Coherence','eVar']
        
        # Prepare DataFrames to organize results for plotting
        MCohe = pd.DataFrame() # To store mean phase coherence results
        EVar = pd.DataFrame()  # To store PCA explained variance ratios
        
        # Loop through each coupling strength to extract corresponding results
        for kappa in kappas:
            # Filter results for current kappa and extract 'Coherence' values
            MCohe[kappa] = result[result['kappa']==kappa]['Coherence'].values
            # Extract 'eVar' values
            EVar[kappa]  = result[result['kappa']==kappa]['eVar'].values
    
        # Plotting section
        fig = plt.figure(figsize=(9,5))
        ax = fig.add_subplot()
        
        # Set axis labels with LaTeX formatting for clarity
        ax.set_xlabel(r'$\kappa$') # Coupling strength
        ax.set_ylabel(r'$R$')      # Coherence measure
        
        # Hide the top spine for a cleaner look
        ax.spines['top'].set_visible(False)
        
        # Create a secondary y-axis to plot explained variance
        ax2 = ax.twinx()
        ax2.spines['top'].set_visible(False)         # Hide top spine
        ax2.set_ylabel(r'Explained Var.',color='r')  # Expl. variance label
        ax2.spines['right'].set_color('red')
        ax2.tick_params(axis='y', colors='red')      # Match tick color
        
        # Plot mean phase coherence across kappas with black line
        plotGV(ms=kappas,gs_data=MCohe,color='k',ax=ax)
        # Plot explained variance with red line on secondary y-axis
        plotGV(ms=kappas,gs_data=EVar,color='r',ax=ax2)
        
        # Adjust layout for neatness
        fig.tight_layout()
        
        