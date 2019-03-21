import numpy as np
from functools import reduce 
import numpy.random as rd
from scipy.stats import norm
from scipy.optimize import curve_fit
import scipy as sp
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
from sympy import Eq, Symbol, solve, nsolve, DiracDelta
from scipy.interpolate import interp1d
from scipy.interpolate import InterpolatedUnivariateSpline


Omega_m = 0.3
Omega_r = 5.5e-5
Omega_Lambda = 1-Omega_m

#### Read and interpolate constraints #####
Omega_at_zc = np.loadtxt("Omega_at_zc.dat")
interp_constraints_log_extrap = InterpolatedUnivariateSpline(np.log10(Omega_at_zc[:,0]),np.log10(Omega_at_zc[:,1]),k=1)
interp_constraints_log_interp = interp1d(np.log10(Omega_at_zc[:,0]),np.log10(Omega_at_zc[:,1]))

def rmt_input():
	############################# Load RMT Data
	rmt_data_mass=[]
	rmt_data_decay=[]
	rmt_data_phi=[]
	for i in range(0,40):
		array_mass = (np.load('mass_data/nparray{0}.npy'.format(i)))
		rmt_data_mass.append(array_mass)
		array_decay = (np.load('decay_data/nparray{0}.npy'.format(i)))
		rmt_data_decay.append(array_decay)
		array_phi = (np.load('phi_data/nparray{0}.npy'.format(i)))
		rmt_data_phi.append(array_phi)
	#############################

	############################# Define RMT Paramter Space
	beta_k = np.linspace(0.01,1.0,len(rmt_data_mass[0]))
	beta_m = np.linspace(0.01,1.0,len(rmt_data_mass))
	x_sp = []
	y_sp = []
	for j in range(0,len(rmt_data_mass)):
		temp = [beta_m[j]]*len(rmt_data_mass[0])
		y_sp.append(temp)
		for i in range(0,len(rmt_data_mass[0])):
			x_sp.append(beta_k[i])
	#############################
	y_sp = np.reshape(y_sp,(len(rmt_data_mass)*len(rmt_data_mass[0])))
	
	gld_one_mass=[]
	gld_two_mass=[]
	gld_three_mass=[]
	gld_four_mass=[]
	gld_one_decay=[]
	gld_two_decay=[]
	gld_three_decay=[]
	gld_four_decay=[]
	gld_one_phi=[]
	gld_two_phi=[]
	gld_three_phi=[]
	gld_four_phi=[]
		
	for i in range(0,len(rmt_data_mass)):
		for j in range(0,len(rmt_data_mass[0])):
			gld_one_mass.append(rmt_data_mass[i][j][0])
			gld_one_decay.append(rmt_data_decay[i][j][0])
			gld_one_phi.append(rmt_data_phi[i][j][0])
			gld_two_mass.append(rmt_data_mass[i][j][1])
			gld_two_decay.append(rmt_data_decay[i][j][1])
			gld_two_phi.append(rmt_data_phi[i][j][1])
			gld_three_mass.append(rmt_data_mass[i][j][2])
			gld_three_decay.append(rmt_data_decay[i][j][2])
			gld_three_phi.append(rmt_data_phi[i][j][2])
			gld_four_mass.append(rmt_data_mass[i][j][3])
			gld_four_decay.append(rmt_data_decay[i][j][3])
			gld_four_phi.append(rmt_data_phi[i][j][3])
			
	gld_one_mass = np.reshape(gld_one_mass,(len(rmt_data_mass),len(rmt_data_mass[0])))
	gld_two_mass = np.reshape(gld_two_mass,(len(rmt_data_mass),len(rmt_data_mass[0])))
	gld_three_mass = np.reshape(gld_three_mass,(len(rmt_data_mass),len(rmt_data_mass[0])))
	gld_four_mass = np.reshape(gld_four_mass,(len(rmt_data_mass),len(rmt_data_mass[0])))
	
	gld_one_decay = np.reshape(gld_one_decay,(len(rmt_data_decay),len(rmt_data_decay[0])))
	gld_two_decay = np.reshape(gld_two_decay,(len(rmt_data_decay),len(rmt_data_decay[0])))
	gld_three_decay = np.reshape(gld_three_decay,(len(rmt_data_decay),len(rmt_data_decay[0])))
	gld_four_decay = np.reshape(gld_four_decay,(len(rmt_data_decay),len(rmt_data_decay[0])))
	
	gld_one_phi = np.reshape(gld_one_phi,(len(rmt_data_phi),len(rmt_data_phi[0])))
	gld_two_phi = np.reshape(gld_two_phi,(len(rmt_data_phi),len(rmt_data_phi[0])))
	gld_three_phi = np.reshape(gld_three_phi,(len(rmt_data_phi),len(rmt_data_phi[0])))
	gld_four_phi = np.reshape(gld_four_phi,(len(rmt_data_phi),len(rmt_data_phi[0])))

	spline1m = sp.interpolate.Rbf(x_sp,y_sp,gld_one_mass,function='multiquadric',smooth=2, episilon=2)
	spline2m = sp.interpolate.Rbf(x_sp,y_sp,gld_two_mass,function='multiquadric',smooth=2, episilon=2)
	spline3m = sp.interpolate.Rbf(x_sp,y_sp,gld_three_mass,function='multiquadric',smooth=2, episilon=2)
	spline4m = sp.interpolate.Rbf(x_sp,y_sp,gld_four_mass,function='multiquadric',smooth=2, episilon=2)
	
	spline1f = sp.interpolate.Rbf(x_sp,y_sp,gld_one_decay,function='multiquadric',smooth=2, episilon=2)
	spline2f = sp.interpolate.Rbf(x_sp,y_sp,gld_two_decay,function='multiquadric',smooth=2, episilon=2)
	spline3f = sp.interpolate.Rbf(x_sp,y_sp,gld_three_decay,function='multiquadric',smooth=2, episilon=2)
	spline4f = sp.interpolate.Rbf(x_sp,y_sp,gld_four_decay,function='multiquadric',smooth=2, episilon=2)
	
	spline1p = sp.interpolate.Rbf(x_sp,y_sp,gld_one_phi,function='multiquadric',smooth=2, episilon=2)
	spline2p = sp.interpolate.Rbf(x_sp,y_sp,gld_two_phi,function='multiquadric',smooth=2, episilon=2)
	spline3p = sp.interpolate.Rbf(x_sp,y_sp,gld_three_phi,function='multiquadric',smooth=2, episilon=2)
	spline4p = sp.interpolate.Rbf(x_sp,y_sp,gld_four_phi,function='multiquadric',smooth=2, episilon=2)
	
	return(spline1m,spline2m,spline3m,spline4m,spline1f,spline2f,spline3f,spline4f,spline1p,spline2p,spline3p,spline4p)


def hyperpriors(distributions,hyperprior_vector):
	hyperparameter_vector=[]	
	for i in range(len(hyperprior_vector)):	
		if isinstance(hyperprior_vector[i],(list,)):
			dis, *vec = hyperprior_vector[i]
			hyperparameter_vector.append(getattr(rd, distributions[dis])(*vec))
		else:
			hyperparameter_vector.append(hyperprior_vector[i])	
	print(hyperparameter_vector)		
	return(hyperparameter_vector)

				
def gld_params(betaK,betaM,fun1,fun2,fun3,fun4,fun5,fun6,fun7,fun8,fun9,fun10,fun11,fun12):
	
	l1=fun1(betaK,betaM)
	l2=fun2(betaK,betaM)
	l3=fun3(betaK,betaM)
	l4=fun4(betaK,betaM)
	l5=fun5(betaK,betaM)
	l6=fun6(betaK,betaM)
	l7=fun7(betaK,betaM)
	l8=fun8(betaK,betaM)
	l9=fun9(betaK,betaM)
	l10=fun10(betaK,betaM)
	l11=fun11(betaK,betaM)
	l12=fun12(betaK,betaM)
	
	return(l1,l2,l3,l4,l5,l6,l7,l8,l9,l10,l11,l12)	

def A(l3,l4):
	a = (1./(1.+l3)) - (1./(1+l4))
	return a
	
def B(l3,l4):	
	b = (1./(1.+(2*l3))) + (1./(1+(2*l4))) - 2*sp.special.beta(1+l3,1+l4)
	return b 
	
def C(l3,l4):	
	c = (1./(1.+(3.*l3))) - (1./(1.+(3.*l4))) - 3.*sp.special.beta(1.+2.*l3,1.+l4)+ 3.*sp.special.beta(1.+l3,1.+2.*l4)
	return c
	
def D(l3,l4):
	d = (1./(1.+(4*l3))) + (1./(1+(4*l4))) - 4*sp.special.beta(1+3*l3,1+l4) + 6*sp.special.beta(1+2*l3,1+2*l4) + 4*sp.special.beta(1+l3,1+3*l4)	
	return d
	
def GLD_central_moments(l1,l2,l3,l4):
	
	print(A(l3,l4),B(l3,l4),C(l3,l4),D(l3,l4))
	
	m1 = l1+(A(l3,l4))/l2
	m2 = (B(l3,l4)-A(l3,l4))/l2**2
	m3 = (C(l3,l4)-3.*A(l3,l4)*B(l3,l4)+2.*A(l3,l4)**2.)/((l2**3.*m2**1.5)) 
	m4 = (D(l3,l4)-4.*A(l3,l4)*C(l3,l4)+6.*A(l3,l4)**2.*B(l3,l4)-3.*A(l3,l4)**4.)/(l2**4.*m2**2.)
	
	return(m1,m2,m3,m4)

def gld_pdf(l1,l2,l3,l4):
	y = np.linspace(0.0000001,0.9999999999,500000)
	x = l1 + (1./l2)*(((y**l3)-1)/l3 - (((1-y)**l4)-1)/l4)
	f_x = l2/(y**(l3-1)+(1-y)**(l4-1))
	f = interp1d(x, f_x,bounds_error=None)
	try:
		a = f(-5)
	except ValueError:
		a = 0
	return(x,f_x)
	
def gld_prior(x,f_x,x_point):
	f = interp1d(x, f_x)
	try:
		a = f(x_point)
	except ValueError:
		a = 0
	return(a)	

def marcenko_pastur_pdf(beta, a0, decay):
	#x_space = np.logspace(-5,0.74345267648,10000)*a0**2
	x_space = np.logspace(np.log10(decay.min()),np.log10(decay.max()),100000)
	ub = a0**2*(1 + np.sqrt(beta))**2
	lb = a0**2*(1 - np.sqrt(beta))**2 
	mp = np.zeros(len(x_space))
	lbidx = np.where(x_space > lb)
	ubidx = np.where(x_space < ub)  
	a = lbidx[0][0]
	b = ubidx[-1][-1]
	xh = x_space[a:b+1]
	mp[a:b+1] = np.sqrt((xh - lb)*(ub - xh))/(2*np.pi*beta*xh*a0**2)  
	f_mp = interp1d(x_space, mp, kind='linear')
	return  (f_mp)

def diag_matrix_theory(hyperparameters):
	n,betaK,betaM,a0,b0,phi_range = hyperparameters
	
	print(betaK)
	
	LK=int(n/betaK)
	LM=int(n/betaM)
	k  = a0*(np.random.randn(n, LK))
	k2 = np.dot(k,(k.T))/LK # Factor of L
	ev,p = np.linalg.eig(k2) 
	fef = np.sqrt(np.abs(ev))
	fmat = np.zeros((n,n))
	np.fill_diagonal(fmat,fef)	 
	kD = reduce(np.dot, [p.T, k2, p]) 
	kD[kD < 1*10**-13] = 0 
	kDr = np.zeros((n, n)) 
	np.fill_diagonal(kDr, 1./(fef))

	m = b0*(np.random.randn(n, LM)) 
	M = np.dot(m,(m.T))/LM # Factor of L
	mn = reduce(np.dot, [kDr,p,M,p.T,kDr.T]) 
	eigs,mv = np.linalg.eig(mn) 
	ma_array=np.sqrt(eigs)
	
	return(ma_array, fef, mv, ev)


def spectra_matrix_theory(hyperparameters,accuracy): 
	biglistm = []
	biglistf = []
	biglistf2 = []
	biglistphi = []
	count = 0
	totalcount = 0
	print('Hyperparameters inputted.')
	print('Calculating....')
	while count < accuracy and totalcount < 100000000:
		print(count)
		flag = 0
		totalcount = totalcount+1
		ma_array, fef, mv, ev= diag_matrix_theory(hyperparameters)
		phiin_array = phi_array(fef, hyperparameters[0], hyperparameters[-1], mv)
		
		if flag == 0:
			count = count + 1
			biglistm = np.concatenate((biglistm,ma_array))			
			biglistf = np.concatenate((biglistf,fef))
			biglistphi = np.concatenate((biglistphi,phiin_array))
	lma_array = np.log10(biglistm)
	lf_array = np.log10(biglistf)	
	stack = np.vstack([lma_array, lf_array, biglistphi])
	return(lma_array, lf_array, biglistphi,stack)		
			
	

def diag_mtheory(hyperparameters):
	n,beta,a0,sa,b0,sb,F,Lambda,smin,smax,Ntildemax,phi_range = hyperparameters
	s = np.random.uniform(smin,smax,n)
	k = np.zeros((n,n))
	np.fill_diagonal(k,a0*a0/s/s)
	ev,p = np.linalg.eig(k) 
	fef = np.sqrt(np.abs(2.*ev)) 
	fmat = np.zeros((n,n))
	np.fill_diagonal(fmat,fef)
	kDr = np.zeros((n, n))
	np.fill_diagonal(kDr, (1./fef))
	
	L = int(n/beta)
	Ntilde = np.random.uniform(0,Ntildemax,size=(n,L))
	Sint = np.dot(s,Ntilde)
	Esint = np.exp(-Sint/2.)
	Idar = n*[1.]
	Cb = np.sqrt(np.dot(Idar,Ntilde))
	A = 2*np.sqrt(F*Lambda*Lambda*Lambda)*reduce(np.multiply,[Cb,Esint,Ntilde]) 
	m = np.dot(A,A.T)/L
	mn = 2*reduce(np.dot, [kDr,p,m,p.T,kDr.T]) 
	ma_array,mv = np.linalg.eigh(m)

	remove_tachyons=True	
	#if remove_tachyons:
	#	ma_array[ma_array<0]=0.
	ma_array = np.sqrt(np.abs(ma_array))

	vol = np.dot(s,Ntilde)/(2*np.pi)

	return(ma_array, fef, mv, vol)

def phi_array(fef,n,phi_range,mv):
	phiin_array = rd.uniform(-phi_range,phi_range,n) 
	for i in range (0,n):
		phiin_array[i] = phiin_array[i]*fef[i]  
	phiin_array=np.dot(mv,phiin_array) 
	return(phiin_array)


def spectra(hyperparameters,accuracy):
	biglistm = []
	biglistf = []
	biglistphi = []
	volumelist = []
	count = 0
	totalcount = 0
	print('Hyperparameters inputted.')
	print('Calculating....')
	while count < accuracy and totalcount < 100000000:
		print(count)
		flag = 0
		totalcount = totalcount+1
		ma_array, fef, mv, vol = diag_mtheory(hyperparameters )
		phiin_array = phi_array(fef, hyperparameters[0], hyperparameters[-1], mv)
		#if flag == 0 and any(x < 30 and x > 20 for x in vol):
		if flag == 0:
			count = count + 1
			biglistm = np.concatenate((biglistm,ma_array))			
			biglistf = np.concatenate((biglistf,fef))
			biglistphi = np.concatenate((biglistphi,phiin_array))
			volumelist = np.concatenate((volumelist,vol))

	lma_array = np.log10(biglistm) - 33
	lf_array = np.log10(biglistf)
	return(count, totalcount, lma_array, lf_array, volumelist, biglistphi)


def pdf_fit_matrix_theory(lma_array,biglistphi):
	fit_params_mass=norm.fit(lma_array)
	fit_params_phi=norm.fit(biglistphi)
	return(fit_params_mass,fit_params_phi)
	
def pdf_fit_mtheory(volumelist,lma_array,biglistphi):
	fit_params_vol=norm.fit(volumelist)
	fit_params_mass=norm.fit(lma_array)
	fit_params_phi=norm.fit(biglistphi)
	return(fit_params_vol, fit_params_mass,fit_params_phi)	

def gaussian(mu,sigma,x):
	f = np.exp(-(x-mu)**2/(2*sigma**2))/(2*np.pi*sigma**2)**0.5
	return f

def Omega_zc(alpha,mu,Theta_i, n):
	return 1./6*alpha*alpha*mu*mu*(1-np.cos(Theta_i))**n

def return_zc(alpha,mu,Theta_i, n):
	
	##We fist assume zc>z_eq.
	p=1./2
	F=7./8
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
		zc_found = 0
		Results = solve(Omega_m*(1+zc)**3.0+Omega_r*(1+zc)**4+Omega_Lambda-(p/xc)**2,zc)
		if Results[0] > 0:
			 zc_found = Results[0]
		if Results[1] > 0:
			zc_found = Results[1]
		if zc_found == 0:
			print("Weird! zc is negative: it might mean that the field is a dark energy candidate.")     
	return zc_found

def sigma_zc(alpha,mu,Theta_i, n):
	zc_found = return_zc(alpha,mu,Theta_i, n)
	if 0 > np.log10(float(1/zc_found)) > -6:
		return interp_constraints_log_interp(np.log10(float(1/zc_found)))
	else:
		return interp_constraints_log_extrap(np.log10(float(1/zc_found)))
	
def posterior(alpha,mu,Theta_i,n):
	return gaussian(0.0,sigma_zc(alpha,mu,Theta_i,n),np.log10(Omega_zc(alpha,mu,Theta_i, n)))

def posterior3(alpha,mu,Theta_i):
	return posterior(alpha,mu,Theta_i,3)#*DiracDelta(alpha-1)*DiracDelta(mu-10**5)*DiracDelta(Theta_i-3)

def func(x, a, b, c, d):
	return a*np.exp(-c*(x-b))+d
##TEMPORARY: define prior for a fixed axiverse hyperparameter and approximate fit of Matt's code 
##TO DO: use histograms from Matt's code

def decay_fit_mtheory(decay_array,accuracy,ula_decay_constant):
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
	
def posterior_times_prior_mtheory(ula_decay_constant,ula_mass,ula_theta,                                         fit_params_mass,     fit_params_phi,decay_array):
	print('doing this...')
	accuracy =  1000
	H0_eV = 1.4e-33
	return decay_fit_mtheory(decay_array,accuracy,ula_decay_constant)*prior_mu(fit_params_mass,ula_mass)*prior_theta(fit_params_phi,ula_theta)	

def posterior_times_prior(log10alpha,log10mu,Theta_i):
	print('doing this...')
	H0_eV = 1.4e-33
	return 10**log10mu*posterior(10**log10alpha,(10**log10mu)/H0_eV,Theta_i,3)*prior_alpha(log10alpha)*prior_mu(log10mu)*prior_theta(Theta_i)
	
