#######################################################################################################################
import numpy as np
import matplotlib.pyplot as plt
import scipy.special as sci
from matplotlib import rc
from scipy.interpolate import interp1d
from scipy.interpolate import InterpolatedUnivariateSpline
from scipy.integrate import nquad
from sympy import Eq, Symbol, solve, nsolve, DiracDelta
from mpl_toolkits.mplot3d import Axes3D
from functools import reduce
import numpy.random as rd
from scipy.stats import norm, uniform
from scipy.optimize import curve_fit
import scipy as sp
import functions
import sys
from scipy import stats
#rc('font',**{'family':'sans-serif','sans-serif':['Helvetica']})
## for Palatino and other serif fonts use:
rc('font',**{'family':'serif','serif':['Times New Roman']})
rc('text', usetex=True)
#######################################################################################################################


### CODE V2: calculate analytically the convolution of the posterior distribution and the prior ###
#### Define cosmological parameters #####
Omega_m = 0.3
Omega_r = 5.5e-5
Omega_Lambda = 1-Omega_m

#### Read and interpolate constraints #####
Omega_at_zc = np.loadtxt("Omega_at_zc.dat")
interp_constraints_log_extrap = InterpolatedUnivariateSpline(np.log10(Omega_at_zc[:,0]),np.log10(Omega_at_zc[:,1]),k=1)
interp_constraints_log_interp = interp1d(np.log10(Omega_at_zc[:,0]),np.log10(Omega_at_zc[:,1]))


#######################################################################################################################
#1 = Scale invariant measure
#2 = M-theory effective model
#3 = Unimodal RMT
#######################################################################################################################

log_ula_mass = -30
log_ula_alpha = -1
ula_theta = 1.1

Model_control = 3
distributions = ['uniform','normal','beta','gamma','lognormal','poisson','weibull','laplace','exponential','chisquare']
if Model_control == 1:

#######################################################################################################################
##### String axiverse conjecture scale inveriance
#######################################################################################################################

	def uniform_prior_functions(hyperparameters):
		log_ulasector_lower_mass, log_ulasector_upper_mass, log_ulasector_lower_alpha, log_ulasector_upper_alpha, ula_theta_range = hyperparameters
		rv_mass = uniform(np.abs(log_ulasector_upper_mass),np.abs(log_ulasector_lower_mass))
		rv_decay = uniform( np.abs(log_ulasector_upper_alpha),np.abs(log_ulasector_lower_alpha))
		rv_theta = uniform(0,ula_theta_range)
		return rv_mass, rv_decay, rv_theta

	def	uniform_prior(ula_decay_constant,ula_mass,ula_theta):
		global prior_mass, prior_decay, prior_theta
		prior_ula_mass = prior_mass.pdf(np.abs(ula_mass))
		prior_ula_decay = prior_decay.pdf(np.abs(ula_decay_constant))
		prior_ula_phi = prior_theta.pdf(ula_theta)
		return prior_ula_mass,prior_ula_decay,prior_ula_phi

	def posterior_times_prior(ula_decay_constant,ula_mass,ula_theta):
		print('Performing integration....[Log-flat Model]')
		H0_eV = 1.4e-33
		value = ula_decay_constant,ula_mass,ula_theta,10**ula_mass*functions.posterior(10**ula_decay_constant,(10**ula_mass)/H0_eV,ula_theta,3)*uniform_prior(ula_decay_constant,ula_mass,ula_theta)[0]*uniform_prior(ula_decay_constant,ula_mass,ula_theta)[1]*uniform_prior(ula_decay_constant,ula_mass,ula_theta)[2]
		st=str(ula_decay_constant)
		st2=str(ula_mass)
		st3=str(ula_theta)
		st4=str(value[3])
		with open("output.txt", "a") as myfile:
			myfile.write(st+','+st2+','+st3+','+st4+'\n' )

		return 10**ula_mass*functions.posterior(10**ula_decay_constant,(10**ula_mass)/H0_eV,ula_theta,3)*uniform_prior(ula_decay_constant,ula_mass,ula_theta)[0]*uniform_prior(ula_decay_constant,ula_mass,ula_theta)[1]*uniform_prior(ula_decay_constant,ula_mass,ula_theta)[2]

	
	hyperprior_vector = [-35,-10,-2,-0.5,3.14]
	hyperparameters=functions.hyperpriors(distributions,hyperprior_vector)
	prior_mass, prior_decay, prior_theta = uniform_prior_functions(hyperparameters)
	test = posterior_times_prior(log_ula_alpha, log_ula_mass,ula_theta)
	print(test)

if Model_control == 2:
#######################################################################################################################
##### M theory model
#######################################################################################################################
	def func(x, a, b, c, d):
		return a*np.exp(-c*(x-b))+d

	def gaussian(mu,sigma,x):
		f = np.exp(-(x-mu)**2/(2*sigma**2))/(2*np.pi*sigma**2)**0.5
		return f

	def decay_fit_mtheory(ula_decay_constant):
		decay_array = lf_array
		accuracy = 1000
		n, bin, patches  = plt.hist(decay_array,accuracy,density=True,alpha=0.0)
		x_space = np.linspace((decay_array.min()),decay_array.max(),accuracy)
		popt, pcov = curve_fit(func, x_space, n, p0=(0.020,0.1,10,10))
		#function = (popt[0]*np.exp(-popt[2]*(x_space-popt[1]))+popt[3])*np.heaviside(decay_array.max()-x_space,decay_array.max())*np.heaviside(x_space-decay_array.min(),decay_array.min())
		if ula_decay_constant > decay_array.min() and ula_decay_constant < decay_array.max():
			function = (popt[0]*np.exp(-popt[2]*(ula_decay_constant-popt[1]))+popt[3])
		else:
			function = 0
		#function = [0 if i < 0 else i for i in function]
		function = np.abs(function)
		return function

	def prior_mu(fit_params_mass,ula_mass):
		return gaussian(fit_params_mass[0],fit_params_mass[1],ula_mass)

	def prior_theta(fit_params_phi,ula_theta):
		return gaussian(fit_params_phi[0],fit_params_phi[1],ula_theta)

	def posterior_times_prior(ula_decay_constant,ula_mass,ula_theta):
		print('Performing integration....[M-theory Model]')
		print(ula_decay_constant)
		global fit_params_mass, fit_params_phi
		fit_params_mass = fit_params_mass
		fit_params_phi = fit_params_phi
		accuracy =  1000
		H0_eV = 1.4e-33

		return 10**ula_mass*functions.posterior(10**ula_decay_constant,(10**ula_mass)/H0_eV,ula_theta,3)*decay_fit_mtheory(ula_decay_constant)*prior_mu(fit_params_mass,ula_mass)*prior_theta(fit_params_phi,ula_theta)

	#hyperparameterisation
	accuracy=100
	hyperprior_vector = [10,1.0,1.,2.,10.,21.,10**105,1.,10.,100.,0.8,3.14159265359]
	#hyperprior_vector = [10,[1,0,1],[1,0,1],[1,0,1],[1,0,1],[1,0,1],[1,0,1],[1,0,1],[1,0,1],[1,0,1],[1,0,1],np.pi]
	hyperparameters=functions.hyperpriors(distributions,hyperprior_vector)
	#ma_array, fef, phiin_array, vol = functions.diag_mtheory(hyperprior_vector)
	count, totalcount, lma_array, lf_array, volumelist, initial_phi = functions.spectra(hyperparameters,accuracy)
	fit_params_vol, fit_params_mass, fit_params_phi = functions.pdf_fit_mtheory(volumelist,lma_array,initial_phi)

	##### Determination of prior values
	prior_alpha_function = functions.decay_fit_mtheory(lf_array, 1000, log_ula_alpha)
	prior_mu_function = functions.prior_mu(fit_params_mass, log_ula_mass)
	prior_theta_function = functions.prior_theta(initial_phi,ula_theta)

	posterior_times_prior(log_ula_alpha,log_ula_mass,ula_theta)



#######################################################################################################################

if Model_control == 3:
#######################################################################################################################
############## RMT  Model
#######################################################################################################################

	#n,betaK,betaM,a0,b0,phi_range = functions.matrix_theory_hyperpriors(distributions,10,[1,1,1],[1,1,1],[1,1,1],[1,1,1],np.phi)
	accuracy = 100
	hyperprior_vector = [20,1.0,1.0,0.1,0.1,3.14159265359]
	hyperparameters = functions.hyperpriors(distributions, hyperprior_vector)
	#######################################################################################################################
	############## Dependant Analysis
	#######################################################################################################################
	lma_array, lf_array, biglistphi, stack = functions.spectra_matrix_theory(hyperparameters,accuracy)
	kde = stats.gaussian_kde(stack,bw_method=0.075)
	#######################################################################################################################

	#######################################################################################################################
	############## Independant Analysis
	#######################################################################################################################
	'''
	fun1m,fun2m,fun3m,fun4m,fun1f,fun2f,fun3f,fun4f,fun1p,fun2p,fun3p,fun4p  = functions.rmt_input()
	l1m,l2m,l3m,l4m,l1f,l2f,l3f,l4f,l1p,l2p,l3p,l4p = functions.gld_params(betaK,betaM,fun1m,fun2m,fun3m,fun4m,fun1f,fun2f,fun3f,fun4f,fun1p,fun2p,fun3p,fun4p)
	gld_x_mass,gld_y_mass = functions.gld_pdf(l1m,l2m,l3m,l4m)
	gld_x_decay,gld_y_decay = functions.gld_pdf(l1f,l2f,l3f,l4f)
	gld_x_phi,gld_y_phi = functions.gld_pdf(l1p,l2p,l3p,l4p)
	##### Determination of prior values
	gld_prior_mass = functions.gld_prior(gld_x_mass,gld_y_mass,0.8)
	gld_prior_decay = functions.gld_prior(gld_x_decay,gld_y_decay,0.8)
	gld_prior_phi = functions.gld_prior(gld_x_phi,gld_y_phi,0.8)
	'''
	#######################################################################################################################

	def posterior_times_prior(ula_decay_constant,ula_mass,ula_theta):
		print('Performing integration....[RMT Model]')
		print(ula_decay_constant,ula_mass,ula_theta)
		H0_eV = 1.4e-33
		value = ula_decay_constant,ula_mass,ula_theta,10**ula_mass*functions.posterior(10**ula_decay_constant,(10**ula_mass)/H0_eV,ula_theta,3)*kde.evaluate([ula_mass,ula_decay_constant,ula_theta])

		st=str(ula_decay_constant)
		st2=str(ula_mass)
		st3=str(ula_theta)
		st4=str(value[3])
		with open("output.txt", "a") as myfile:
			myfile.write(st+','+st2+','+st3+','+st4+'\n' )

		return 10**ula_mass*functions.posterior(10**ula_decay_constant,(10**ula_mass)/H0_eV,ula_theta,3)*kde.evaluate([ula_mass,ula_decay_constant,ula_theta])



	test = posterior_times_prior(log_ula_alpha, log_ula_mass,ula_theta)

#######################################################################################################################
############## Integration
#######################################################################################################################
#decay constant, mass, initial field value
A = -20
B = -18
quad = (nquad(posterior_times_prior, [[-3,-1],[A, B],[0.,3.14]]))
#######################################################################################################################

























'''
log_ula_mass = -30
log_ula_alpha = -1
ula_theta = 0.01
#gaussian(0.0,sigma_zc(alpha,mu,Theta_i,n),np.log10(Omega_zc(alpha,mu,Theta_i, n)))
#functions.posterior_times_prior(log_ula_alpha,log_ula_mass,ula_theta)
##We calculate the probability of the axiverse hyperparameter here.
##The last 3 entries are the range of log10alpha (in mpl), log10mu (in eV), theta_i
#print(nquad(posterior_times_prior_mtheory, [[-1.75,-1],[-25, -15],[1.,1.1]]),'thisssssssssssss')
'''
###OLD CODE###
#### Define cosmological parameters #####

'''
#### Define model parameters #####
n=1
F=7./8
p=1./2 ##will be checked later

#### Read and interpolate constraints #####
Omega_at_zc = np.loadtxt("Omega_at_zc.dat")
interp_constraints_log_extrap = InterpolatedUnivariateSpline(np.log10(Omega_at_zc[:,0]),np.log10(Omega_at_zc[:,1]),k=1)
interp_constraints_log_interp = interp1d(np.log10(Omega_at_zc[:,0]),np.log10(Omega_at_zc[:,1]))

#### Define some tables for writing #####
mu_excluded = []
alpha_excluded = []
Theta_i_excluded = []
mu_allowed = []
alpha_allowed = []
Theta_i_allowed = []

#### Define distribution properties #####
###parameters of a log flat distribution (currently centered on 0 +- 2)
log_mu_min = -2 ##minimal value
log_mu_size = 18 ##size of the interval.
log_alpha_min = -4.5
log_alpha_size = 4

##for a future gaussian distribution
##var_log_mu =
## mean_log_mu = 3
## log_mu = var_log_mu*np.abs(np.random.rand())*np.pi+mean_log_mu

#### some additional checks ####
print_in_file = True
print_in_screen = True
plot_results = True

#### How many iterations? #####
max_iterations = 10000


#### The code really starts here! #####
total = 0
allowed = 0
excluded = 0

while total < max_iterations:
	total+=1 ##increment the total number of points
	##Draw from log flat distribution##
	log_mu = np.random.rand()*log_mu_size+log_mu_min #np.random.rand() draws in [0,1) ##log_mu_min is negative!
	mu = 10**log_mu
	log_alpha = np.random.rand()*log_alpha_size+log_alpha_min
	alpha = 10**log_alpha
	##Draw from flat distribution
	Theta_i = np.random.rand()*np.pi ##draws from [0;pi)

	##Calculate Omega_at_zc and zc
	Omega_zc = 1./6*alpha*alpha*mu*mu*(1-np.cos(Theta_i))**n
	xc = (1-np.cos(Theta_i))**((1-n)/2)/mu*np.sqrt((1-F)*(6*p+2)*Theta_i/(n*np.sin(Theta_i)))
	zc = Symbol('zc')
	Results = solve(Omega_m*(1+zc)**3.0+Omega_r*(1+zc)**4+Omega_Lambda-(p/xc)**2,zc) ##nb: this assumes negligible Omega_zc. What if not??
	zc_found = 0
	##Results is a table with 4 entries: the 2 firsts are real, one of which is positive: this is our zc.
	if Results[0] > 0:
		 zc_found = Results[0]
	if Results[1] > 0:
		zc_found = Results[1]

	##Check that our hypothesis of zc>z_eq was correct. Otherwise we recalculate.
	if zc_found < Omega_m/Omega_r:
		p=2./3
		xc = (1-np.cos(Theta_i))**((1-n)/2)/mu*np.sqrt((1-F)*(6*p+2)*Theta_i/(n*np.sin(Theta_i)))
		zc = Symbol('zc')
		Results = solve(Omega_m*(1+zc)**3.0+Omega_r*(1+zc)**4+Omega_Lambda-(p/xc)**2,zc)
		if Results[0] > 0:
			 zc_found = Results[0]
		if Results[1] > 0:
			zc_found = Results[1]
		if zc_found == 0:
			if print_in_screen is True:
				print("Weird! zc is negative: it might mean that the field is a dark energy candidate.")
				print("mu:", mu, "alpha:",  alpha, "Theta_i:",  Theta_i, "Omega_zc:", Omega_zc, "xc:", xc, "zc:", Results[1])
				#print Results
			## Currently compares to Lambda: if = or less, assume it is viable. TO BE IMPROVED.
			if Omega_zc <= Omega_Lambda:
				mu_allowed.append(log_mu)
				alpha_allowed.append(log_alpha)
				Theta_i_allowed.append(Theta_i)
				allowed+=1
			else:
				mu_excluded.append(log_mu)
				alpha_excluded.append(log_alpha)
				Theta_i_excluded.append(Theta_i)
				excluded+=1
		else:
			if print_in_screen is True:
				print("mu:", mu, "alpha:",  alpha, "Theta_i:",  Theta_i, "Omega_zc:", Omega_zc, "xc:", xc, "log10_ac:", np.log10(float(1/zc_found)))
			##interp1d is more stable but cannot extrapolate.
			if 0 > np.log10(float(1/zc_found)) > -6:
				# print "interpolation",np.log10(float(1/zc_found))
				if interp_constraints_log_interp(np.log10(float(1/zc_found))) < np.log10(Omega_zc) :
					if print_in_screen is True:
						print("EXCLUDED",interp_constraints_log_interp(np.log10(float(1/zc_found))), "<",np.log10(Omega_zc), "at ac=",1/zc_found)
					mu_excluded.append(log_mu)
					alpha_excluded.append(log_alpha)
					Theta_i_excluded.append(Theta_i)
					excluded+=1 ##increment the number of points excluded
				else:
					if print_in_screen is True:
						print("allowed",interp_constraints_log_interp(np.log10(float(1/zc_found))), ">", np.log10(Omega_zc), "at ac=",1/zc_found)
					mu_allowed.append(log_mu)
					alpha_allowed.append(log_alpha)
					Theta_i_allowed.append(Theta_i)
					allowed+=1 ##increment the number of points allowed
			else:
				# print "extrapolation"
				if interp_constraints_log_extrap(np.log10(float(1/zc_found))) < np.log10(Omega_zc) :
					if print_in_screen is True:
						print("EXCLUDED",interp_constraints_log_extrap(np.log10(float(1/zc_found))), "<",np.log10(Omega_zc), "at ac=",1/zc_found)
					mu_excluded.append(log_mu)
					alpha_excluded.append(log_alpha)
					Theta_i_excluded.append(Theta_i)
					excluded+=1 ##increment the number of points excluded
				else:
					if print_in_screen is True:
						print("allowed",interp_constraints_log_extrap(np.log10(float(1/zc_found))), ">", np.log10(Omega_zc), "at ac=",1/zc_found)
					mu_allowed.append(log_mu)
					alpha_allowed.append(log_alpha)
					Theta_i_allowed.append(Theta_i)
					allowed+=1 ##increment the number of points allowed

#### if needed, write to output file ####
if print_in_file is True:
	f = open('allowed_models_axiverse.dat', 'w')
	f.write('# mu \t\t alpha \t\t Theta_i \n') # column titles
	for i in range(allowed):
		f.write(str(mu_allowed[i]) + '\t\t' + str(alpha_allowed[i]) + '\t\t' + str(Theta_i_allowed[i]) +'\n') # writes the array to file
	f.close()
#### if needed, plot the allowed parameters ####
if plot_results is True:
	fig = plt.figure()
	ax = fig.add_subplot(111, projection='3d')
	ax.set_xlim(log_mu_min,log_mu_min+log_mu_size)
	ax.set_ylim(log_alpha_min,log_alpha_min+log_alpha_size)
	ax.set_zlim(0,np.pi)
	ax.set_xlabel(r"$\mu$", fontsize=16)
	ax.set_ylabel(r"$\alpha$", fontsize=16)
	ax.set_zlabel(r"$\Theta_i$", fontsize=16)
	ax.scatter(mu_allowed,alpha_allowed,Theta_i_allowed,c='b')
	plt.show()
#### calculate the fraction of allowed models ####
fraction = allowed*1.0/total
print("fraction of models allowed = ", fraction)
'''
