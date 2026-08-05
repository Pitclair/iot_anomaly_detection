###########################################
#   Copyright (C) 2024
###########################################
#   Authors:
#	- Sherenaz Al-Haj Baddar (s.baddar@ju.edu.jo)
#	- Alessandro Languasco (alessandro.languasco@unipd.it)
#	- Mauro Migliardi (mauro.migliardi@unipd.it)
#  This program fits a count dataset using the Dirichlet Multinomial Distribution while comparing
#  The LM technique (see below) to Python's math and scipy implementations of the log-likelihood function
###########################################
#   This software is licensed under the GNU General Public License version 3 (GPL-3), as published by the Free Software Foundation.
#   You can find a copy of the license at https://www.gnu.org/licenses/gpl-3.0.en.html
 ##########################################
#   LM: stands for the Languasco-Migliardi paper
###########################################
#   MAIN PROCEDURE
#	USAGE: call it with python3 LogL-global-v5.py &1 &2
#       USAGE: first parameter &1: precision for the mpmath computation
#       USAGE: second parameter &2: requested accuracy for the mantissa in LM
#
###########################################
#
#   GOAL:
#       We compute  the log-likelihood logL functions 
#       with an internal check on the (guessed) size of logL to be sure that the logL-result will be
#       reliable using the float-type (C double precision) with the correct number of required mantissa digits
#
############################################
#
#   Results computed using:
#
# 	mpmath: to verify the other results
#
#	math: standard approach with the float type (it fails for large overdisperesed datasets)
#
#	scipy: standard approach with the float type (it fails for large overdisperesed datasets)
#
#	LM: the approach in Languasco-Migliardi (2021); it uses the Euler-Maclaurin formula;
#		automatic search of the sum-level m and the horizontal shift to handle the required 
#		mantissa accuracy. It requires six text files with 100 precomputed coefficients:
#		- err_coeff-100; err_coeff-100_digamma; err_coeff-100_trigamma: first 100 coefficients for
#			the error formula of logGamma, digamma, trigamma in [ 1/x , 1/x + y ] (see LM-paper)
#		- bernreal-norm-100; bernreal-norm-100-digamma; bernreal-100: first 100 coefficients for
#			the Euler-Maclaurin formula of logGamma, digamma, trigamma in [ 1/x , 1/x + y ] (see LM-paper)
#
########################################### 



# import the needed python packages
import sys

defaultprecision = float(sys.argv[1]) # mpmath accuracy requested
prec = float(sys.argv[2]) # number of correct decimal digits for the mantissa of the whole problem
import numpy as np
np.set_printoptions(precision=15)

import math 
import mpmath as mp 
from time import perf_counter
import scipy as scipy
import scipy.special as sp
from  scipy.stats import wasserstein_distance 
from scipy.spatial import distance
from scipy.spatial.distance import cdist

from scipy import stats
from scipy.stats import dirichlet
from numpy import linalg as LA
from scipy.stats import chi2_contingency
from scipy.stats import chi2
from scipy.stats import chisquare
from scipy.stats import ks_2samp


import ctypes
from ctypes import *
import os


lib_path = os.getcwd()+"/LM_time_lib.so"
LM_horshift = ctypes.CDLL(lib_path)


# definitions of math functions used

def lngamma64(x): # C or FORTRAN compiled
    y = math.lgamma(x)
    return y
    
def log(x): # C or FORTRAN compiled
    y = math.log(x)
    return y

def log10(x): # C or FORTRAN compiled
    y = math.log10(x)
    return y

def floor(x): 
    y = math.floor(x)
    return y
    
def ceil(x): 
    y = math.ceil(x)
    return y

        
# function logGamma scipy-float64
def scipylngamma64(x): # C or FORTRAN compiled
    y = sp.loggamma(x)
    return y

# function logGamma scipy-float64
def scipylngamma64(x): # C or FORTRAN compiled
    y = sp.loggamma(x)
    return y

def scipydigamma64(x): # C or FORTRAN compiled
    y = sp.digamma(x)
    return y

def scipytrigamma64(x): # C or FORTRAN compiled
    y = sp.polygamma(1,x)
    return y
    
# slower than **    
def pow_int( b,  a):
  retval = np.float64(1.0) 
    
  if a == 0:
    return retval
  #endif
  		
  flag = int(0) 
  if  a < 0 :
    a = -a
    flag = 1
  #endif
    	
  while a > 0 :
    if  a & 1 : # execute when the least significant bit is 1
        retval *= b
    #endif    
    b *= b;
    a >>= 1 # divide by 2 and take the remainder (shift 1 bit on the right)
  #endwhile   

  if flag == 1:
    retval = 1.0/retval
  #endif
    
  return retval


###############################################################
####  ----------  BEGIN LM functions implementation
###############################################################

def LM_initBern():
        # read the precomputed Bernoulli numbers; already weighted if needed
        # the output are four arrays of floats (64 bits; C double type)
        # vec_evenbernoulli: for computing the trigamma differences
	# vec_evenbernoullinorm: for computing the logGamma differences 
	# vec_evenbernoullinormdigamma: for computing the digamma differences 	
	# vec_errcoeff: for computing the error in the Euler-Maclaurin formula for logGamma differences 
    
    lines_1 = []
    lines_2 = []
    lines_3 = []
    lines_4 = []
	
    vec_list = 100*[] 
    vec_evenbernoullinorm = []*100 
    vec_evenbernoullinormdigamma = []*100 
    vec_errcoeff = []*100 

            
    with open('bernreal-100.txt') as f:
        lines_1 = f.readlines()
    f.close()
    vec_evenbernoulli = np.arange(0, len(lines_1), dtype=float)
    
    with open('bernreal-norm-100.txt') as f:
        lines_2 = f.readlines()
    f.close()
    vec_evenbernoullinorm = np.arange(0, len(lines_2), dtype=float) 

    with open('bernreal-norm-100-digamma.txt') as f:
         lines_3 = f.readlines()
    f.close()
    vec_evenbernoullinormdigamma = np.arange(0, len(lines_3), dtype=float) 

    with open('err_coeff-100.txt') as f:
        lines_4 = f.readlines()
    f.close()
    vec_errcoeff = np.arange(0, len(lines_4), dtype=float) 

    i = 0
    for line in lines_1:
        vec_evenbernoulli[i] = float(line)
        i += 1 
    #endfor
    
    i = 0
    for line in lines_2:
        vec_evenbernoullinorm[i] = float(line)
        i += 1 
    #endfor
    
    i = 0
    for line in  lines_3:
        vec_evenbernoullinormdigamma[i] = float(line) 
        i += 1 
    #endfor
    
    i = 0
    for line in lines_4:
        vec_errcoeff[i] = float(line) 
        i += 1 
    #endfor

    return  vec_evenbernoulli,  vec_evenbernoullinorm,   vec_evenbernoullinormdigamma, vec_errcoeff

# aux function


def KL_distance(P, Q):
    #computes the LK distance
    #input: vector of real probabilities P, of size K
    #input: vector of proposed proobabilities Q, of size K
    #output: KL(P||Q)

    _sum = np.float64(0)
    K = len(P)
    for k in range (0, K):
        _sum = _sum + P[k]*math.log(P[k]/Q[k])
    return _sum

    
    
def the_Z_test(mean_1, variance_1,N1, mean_2, variance_2, N2):
    #computes the Z test value
    #input: mean of first sample
    #input: variance of first sample
    #N1: size of first sample
    #input: mean of second sample
    #input: variance of second sample
    #N2: size of second sample
    
    #returns : the Z test value (less than 2 -> the two samples are the same
    #between 2.0 and 2.5 -> the two samples are marginally different
    #between 2.5 and 3.0-> the two samples are significantly different
    #more then 3.0 -> the two samples are highly signficantly different
    
    standard_deviation_1 = math.sqrt(variance_1)
    standard_deviation_2 = math.sqrt(variance_2)
    
    standard_error_1 = standard_deviation_1/math.sqrt(N1)
    standard_error_2 = standard_deviation_2/math.sqrt(N2)

    standard_error_1_squared = standard_error_1 * standard_error_1
    standard_error_2_squared = standard_error_2 * standard_error_2

    stanrard_error_sum = standard_error_1_squared + standard_error_2_squared
    standard_error_root = math.sqrt(stanrard_error_sum)

    z_test =  abs(mean_1 - mean_2)/standard_error_root
    if z_test < 2.5:
        print("Z-test: Accept.  Same dist.");
    else:
        print("Z-test: Reject. Different dist.");

            
def the_mahalanobis_distance(array_1, array_2):
    #calculates the Mahalanobis distance between two vectors
    #input: array_1: first vector
    #input: array_2: second vector
    #output: prints the Mahalanobis distance between array_1 and array_2 according to distance.mahalanobis function
    
    V = np.cov(np.array([array_1, array_2]).T)
    IV = np.linalg.inv(V)
    print(distance.mahalanobis(array_1, array_2, IV))
            
               
def tvd(probs_1, probs_2):
    #calculates the total variation distance between two probability vectors (i.e. distributions)
    #input: probs_1: first vector of probabilities
    #input: probs_2: second vector of probabilities
    #assuming both vectors are of length K
    _sum = np.float64(0)
    K = len(probs_1)
    for k in range(K):
        _sum = _sum + math.fabs(probs_1[k] - probs_2[k])
    _sum = 0.5*_sum
    return _sum
        
    
def overdispersion_index(logL, K):
    #input: logL of the data
    #input: number of classes
    #returns  overdispersion index defined as -2*logL/(k-1)
    # if index>1, data is overdispersed
    res = np.float64(0)
    res = -2*logL*(K-1)
    return res
    
def mean_random_variable(prob_vector):
    #input: vector of K probabilities ( a random variable)
    #ouput: the mean of prob_vector
    K = len(prob_vector)
    _sum_prob  = np.float64(0)
    for k in range(K):
        _sum_prob = _sum_prob + (k+1)*prob_vector[k]
    return _sum_prob

def mean_random_variable_counts(_vector):
    #input: vector of K counts ( a random variable)
    #ouput: the mean of prob_vector
    K = len(_vector)
    _sum  = np.float64(0)
    _sum_counts = np.float64(0)
    _sum_counts = sum(_vector)
    for k in range(K):
        _sum= _sum + (k+1)*_vector[k]
    return _sum/_sum_counts

def variance_random_variable(prob_vector):
    #input: vector of K probabilities ( a random variable)
    #N: number of observations
    #ouput: the variance of prob_vector (variance of random variable)
    K = len(prob_vector)
    _sum_prob  = np.float64(0)
    mu = mean_random_variable(prob_vector)
    for k in range(K):
        squared_diff = (mu -(k+1))*(mu -(k+1))*prob_vector[k]
        _sum_prob = _sum_prob + squared_diff
    return _sum_prob

def variance_random_variable_counts(_vector):
    #input: vector of K counts ( a random variable)
    #N: number of observations
    #ouput: the variance of _vector (variance of random variable)
    K = len(_vector)
    _sum  = np.float64(0)
    _n = np.float64(0)
    _n = sum(_vector)
    mu = mean_random_variable_counts(_vector)
    for k in range(K):
        squared_diff = (mu -(k+1))*(mu -(k+1))*_vector[k]
        _sum = _sum + squared_diff
    return _sum/_n

def _mean_square_error(real_vector, expected_vector):
    #returns the MSE between real and expected probabilities
    _sum_prob  = np.float64(0)
    _diff = np.float64(0)
    squared_diff = np.float64(0)
    K = len(real_vector)
    
    for k in range(K):
        _diff = real_vector[k] - expected_vector[k]
        squared_diff = _diff * _diff
        _sum_prob = _sum_prob + squared_diff

    return _sum_prob/K
        

def vet_counts(D_values):
     R, K = np.shape(D_values)
     for row in range(R):
         for col in range(K):
             if(D_values[row][col] <= 0):
                 print("bad count at row ", row, " and col ", col, "it is ", D_values[row][col], "Quitting, go fix the counts first!!");
                 exit(0);
                 
def increment_by_one(D_vector):
    #input: vector of K values
    #output: vector of K values after increasing each element in it by one

    K = len(D_vector)
    for k in range(K):
        D_vector[k] = D_vector[k] + 1
    return D_vector

def find_negative(D_values):
    #first occurence of negative value in matrix with R rows and K columns
    #input: matrix with R rows and K columns
    #output: 1 if matrix has a value less than 0 at row r and column k, and -1 otherwise
    R, K = np.shape(D_values)
    
    for r in range(R):
        for k in range(K):
            if D_values[r][k] < 0:
                return 1
    return -1

def find_zero(D_vector):
    #first occurence of a zero  in D_vector
    #input: vector of K values
    #output: index k if D_vector has a 0 at k, and -1 if D_vector has no zeros
    K = np.shape(D_vector)[0]
    for k in range(K):
        if(D_vector[k] == 0):
            return k
    return -1

def fix_zero_counts(D_values):
    # eliminate zero counts from the count matrix, D_values
    # input: count matrix D_values with R rows and K columns
    # output: D_values after increasing each category by one, in each row where there is at least one category with value zero. 
    R, K = np.shape(D_values)
    
    
    for row in range(0,R):
        k = find_zero(D_values[row])
        if(k == -1):
            continue
        increment_by_one(D_values[row])                   
    return D_values       

            
def counts_to_probs(D):
    # input: D, matrix of counts, with R rows and K cols
    #output: D_probs: matrix of corresponding probablities, same dimensions 
    
    N, K = np.shape(D)
    D_probs = np.zeros((N, K))
    for n in range(N):
        _sum = int(0)
        _sum_prob  = np.float64(0)
        for k in range(K):
            _sum = _sum + D[n][k]
        for k in range(K-1):
            if(_sum == 0):
                D_probs[n][k]  = 0
            else:
                D_probs[n][k] = D[n][k]/_sum
            _sum_prob = _sum_prob + D_probs[n][k]
        D_probs[n][K-1] = 1 - _sum_prob
    return D_probs

def probs_to_counts(probs_vector):
    # generate count vector from probabilities vector, of K values
    # assume values in probs_vector to be of 2 decimal places
    # input: probabilities vector, of K values, its values should add up to 1
    # returns: corresponding count vector, with cardinality of 100
    factor = 10000
    K = len(probs_vector) 
    counts_vector = np.zeros(K, np.float64)
    _sum = int(0)
    for k in range(K):
        counts_vector[k] = math.floor(probs_vector[k]*factor)
    return counts_vector

def find_ni(D_counts, i):
    # input: D_counts: a matrix of integer counts, with R rows, and K cols
    # input: i: a row in D_counts 
    # output: the sum of row i in D_counts
    n_i= int(0)
    N, K = D_counts.shape
    for k in range(K):
        n_i = n_i + D_counts[i][k]
    return n_i

def find_X(D):
    # input: a matrix of integer counts, with R rows, and K columns
    # output: a vector summary of the input matrix, has K components,  each is summation of a column
    
    factor = 1
    rows = np.size(D, 0)
    cols = np.size(D, 1)
    X = np.zeros(cols, dtype=np.float64) 
    for c in range(cols):
        _sum = int(0)
        for r in range(rows):
            _sum  = _sum + D[r][c]
        X[c] = (int)(_sum * factor)
    return X

def load_dataset_from_file(_file, delim=','):
    # input: _file: the file name
    # input: _delimiter inside file, defualt file format is csv

    # output: two-dimensional matrix of data corresponding to the entries in the file, and N: total observations (summations of all values in matrix)
    with open(_file, 'r') as f:
        D_list = [[int(num) for num in line.split(',')] for line in f]

    D_values = np.array(D_list)

    R, K = D_values.shape
    ### fix issues with counts, if any ######
    ##if there are negative counts, halt the program
    if find_negative(D_values)==1:
        print("unusable dataset, it has negatives, halt it all");
        exit()
    ##resolve all zero counts problems    
    D_values = fix_zero_counts(D_values) 
    #if something is still zero or negative, you have a messy count matrix, go fix it!!
    vet_counts(D_values);
    X = find_X(D_values)
    N = find_N(X)
    return D_values, N

def find_N(X): 
    # adapted from Sherenaz code
    # input: an array of integers
    # output: the sum of its components
    
    N = int(0)
    for x in X:
        N = N + x
    #endfor
    return N        

def dirichlet_pdf(probs, alpha):
    #returns the pdf of Dirichlet Distribution using parameter alpha
    #input: probs: vector of K probabilities, summation of values in probs must be one, K number of categories
    #input: alpha: vector of K values that designate the Dirichlet parameters, K number of categories
    
    return dirichlet.pdf(probs, alpha)

def find_probs(_dataset_file, delim=','):
    #returns the  probabilities for each categorty in the  dataset stored in _dataset_file
    #input: file that contains a dataset, a matrix of counts of R rows and K columns, R is the number of instances in the dataset, and K is the number of categories

    D_values, N = load_dataset_from_file(_dataset_file, delim=',')
    R,K = D_values.shape
    #summation of each column dividied by R
    
    _probs = np.zeros(K, dtype=np.float64)

    counts_vector = find_X(D_values)

    N = find_N(counts_vector)

    _probs =  counts_vector/N

    return _probs

def find_counts(_dataset_file, delim=','):
    #returns the  aggregate counts for each categorty in the  dataset stored in _dataset_file
    #input: file that contains a dataset, a matrix of counts of R rows and K columns, R is the number of instances in the dataset, and K is the number of categories

    D_values, N = load_dataset_from_file(_dataset_file, delim=',')  
    counts_vector = find_X(D_values)

    return counts_vector

def find_relative_error(v_real, v_expected):
    # calculates and returns the relative error in actual value denoted by v_real
    #input: v_real: actual value
    #input: v_expected: expected value

    return 100.0*(abs((v_real - v_expected))/v_expected)

def my_chisquare_test(real_counts, expected_counts):
    #simple calculation of the chi-square goodness of fit value, assumes all chi-square test conditions are met, by default
    #input: real_counts: vector of observed counts, of length K
    #input: expected_counts: vector of expected counts, of length K
    
    K = len(real_counts)
    diff_vector = real_counts - expected_counts
    diff_vector_squared = np.zeros(K, dtype =np.float64)
    for k in range (K):
        diff_vector_squared[k]= diff_vector[k] * diff_vector[k]
    ratio_vector =  diff_vector_squared/expected_counts

    return sum(ratio_vector)
        
    
def chi_square_test(real_counts_vector, expected_counts_vector, dof, conf):
    #input: real_counts_vector: 
    #input: expected_counts_vector:

    #null hypothesis: real and expected frequencies belong to the same distribution
    #alternative hyppothesis: real and expected frequencies DO NOT belong to the same distribution

    #steps: 1. get calcuated chisquare value using real_counts_vector,  and expected_counts_vector
    #       2. get critical value from table using significance level and dof
    #       3. if calculated > critical, reject null, otherwise accept it.  

    critical = chi2.ppf(conf, dof)
    _chivalue, _pvalue = chisquare(real_counts_vector, expected_counts_vector)#, dof)
    if _pvalue < conf: 
        print("Chi-test: Reject, Not same dist.");
    else:
        print("Chi-test: Accept, Same dist.");

def check_for_5(counts_vector):
    # checks if any value in counts vector is less than 5, if so returns 1, else returns 0
    # input: vector_of_counts: a vector of K values
    # output: 1,, if there is at least one value less than 5 in count_vector, 0, otherwise
    K = len(counts_vector)
    for k in range(K):
        if counts_vector[k] < 5:
            return 1
    return 0

##############################################################################    
###### logL functions withthe Euler-Maclaurin formula; see LM-paper and UL-paper
##############################################################################
# main logL-LM function  
def main_LM_logL(probs, psi, K, X, vec_evenbernoullinorm, vec_errcoeff, LM_accuracy):
	#initialization
	res_LM = np.float64(0) 
	# calls the error/hor_shift eval function
	[LMm, hor_shift] = opt_err_LM_logL(probs, psi, K, X, vec_errcoeff, LM_accuracy)
	if LMm == 0 :
		res_LM = 0	# error code for the Euler-Maclaurin formula not accurate enough
	else:
		# calls the logl-eval function that uses the LM-technique
		res_LM = LM_logL_dataset(probs, psi, K, X, LMm, vec_evenbernoullinorm, hor_shift) 
	#endif
	return res_LM
	
# logL function  
def LM_logL(x, y, m, vec_evenbernoullinorm, hor_shift):
	# computes the Euler-Maclaurin formula for the logGamma differences
	# in the interval [1/x, 1/x + y]
	# with a repeated products strategy
	# m: sum-index for the Euler-Maclaurin sum
	# vec_evenbernoullinorm: array of precomputed and normed values for the EM-sum of logGamma
	# hor_shift: value of the horizontal shift (integer)
	# output: retval: value of the log likelihood function in [1/x, 1/x + y]; horizontal shift contribution included

	# types and initializations
	hor_shift_contrib = np.float64(0)
	oneoverx = np.float64(0)
	retval = np.float64(0)	
	d = np.float64(0)	
	stepx = np.float64(0)	
	stepd = np.float64(0)			
	startx = np.float64(0)	
	startd = np.float64(0)			
	fattorex = np.float64(0)	
	fattored = np.float64(0)			

	# special values
	if  y == 1 :
		return ( -log(x) )
	#endif
	if  y == 2 :
		return ( -2 * log(x) + log(1+x) )
	#endif
		
	# handling the horizontal shift contribution; see LM-paper, eq. (12)	
	hor_shift_contrib = 0
	if hor_shift != 0 :
		oneoverx = 1/x
		# computing the horizontal shift contribution
		for j in range (0, hor_shift):
			hor_shift_contrib += log(oneoverx + j) 	
		#endfor 
		# performing the horizontal shift
		y = y - hor_shift	
		x = x/(1 + hor_shift * x)
    #endif	

	yminusone = y-1
	# computing the Euler-Maclaurin formula; see LM-paper, eq. (8)
	d = 1/(1 + x * yminusone )
	# building the repeated product strategy
	stepx = x * x
	stepd = d * d
	startx = x
	startd = d
	fattorex = startx
	fattored = startd
 
	retval = -y * log(x) - yminusone - (1/x + y - 0.5) * log(d) # main term in the LM formula
	
	# first term in the Euler-Maclaurin formula
	retval = retval + vec_evenbernoullinorm[0] * fattorex * ( - 1 + fattored )

	for i in range (1, m):        
		fattorex = fattorex * stepx
		fattored = fattored * stepd
		# next term in the Euler-Maclaurin formula
		retval = retval + vec_evenbernoullinorm[i] * fattorex * ( - 1 + fattored )
	#endfor
    
    # adding the horizontal shift contribution
	retval = retval + hor_shift_contrib    
	return retval

# logL function of a dataset
def LM_logL_dataset(probs, psi, K, X, m, vec_evenbernoullinorm, hor_shift):
	# computes logL on a dataset having:
	# probs: array of the probabilities for each category
	# psi: overdisposition parameter
	# K: number of categories 
	# X: counts of each category 
	# m: sum-index for the Euler-Maclaurin sum
	# vec_evenbernoullinorm: array of precomputed and normed values for the EM-sum of logGamma
	# output: retval: value of the log likelihood function of the dataset; horizontal shift contribution included	
	
	# types and initializations
	N = int(0)
	retval = np.float64(0)		
	
	N = find_N(X)
	psioverprobs = np.zeros(K, dtype=float)
	for k in range(K):
		psioverprobs[k] = psi/probs[k]
	#endfor

	# first interval has a minus sign
	retval = - LM_logL(psi, N, m, vec_evenbernoullinorm, hor_shift)    

	for i in range(K):
		retval = retval + LM_logL(psioverprobs[i], X[i], m, vec_evenbernoullinorm, hor_shift)
	#endfor
        	
	return retval
    
# error functions for logL
def LMg(x, y, m, vec_errcoeff) :
	# computes the error function for the EM formula for logGamma differences
	# in the interval [1/x, 1/x + y]
	# m: sum-index for the Euler-Maclaurin sum
	# vec_errcoeff: array of precomputed and normed values for the error in the EM-sum for logL	
	# output: retval: value of the error estimate in [1/x, 1/x + y]
		
	# types and initializations
	retval = np.float64(0)		
	d = np.float64(0)			

	# error formula; see LM-paper, eq. (9)
	if y>0 and y<3: 
		return(0)
	else:	
		d = 1/( 1 + x * (y-1) )
		# coeff m of the vec_errcoeff is stored in position m-1
		expo = 2*m+1;
		retval = vec_errcoeff[m-1] * (x ** expo) * (1 - (d ** expo) )
		retval = abs(retval)
		return retval
	#endif	   

# computes the total error estimates for LM-logL-dataset
def err(probs, psi, K, X, N, m, vec_errcoeff, j):
	# computes the total error for the EM formula for logGamma differences
	# on a dataset having:
	# probs: array of the probabilities for each category
	# psi: overdisposition parameter
	# K: number of categories 
	# X: counts of each category 
	# N: sum of the counts in X
	# m: sum-index for the Euler-Maclaurin sum
	# vec_errcoeff: array of precomputed and normed values for the error in the EM-sum for logL
	# j: to compute the horizontal shift
	# output: retval: value of the error estimate for the whole dataset
	
	# types and initializations
	aux1 = np.float64(0)		
	aux2 = np.float64(0)		
	
	psioverprobs = np.zeros(K, dtype =float)
	for k in range(K):
		psioverprobs[k] = psi/probs[k]
	#endfor

	# for the horizontal shift
	aux1 = psi/(1 + j * psi)
	aux2 = N - j
		 
	retval = LMg(aux1, aux2, m, vec_errcoeff) 
	
	for i in range(K):
		# for the horizontal shift
		aux1 =  psioverprobs[i]/( 1 + j * psioverprobs[i])
		aux2 =  X[i]-j
		retval = retval + LMg(aux1, aux2, m, vec_errcoeff)
	#endfor	
	return retval

def opt_err_LM_logL(probs, psi, K, X, vec_errcoeff, LM_accuracy):	
	# it computes the optimal m and the horizontal shift to get results with at least LM_accuracy mantissa digits
	# probs: array of the probabilities for each category
	# psi: overdisposition parameter
	# K: number of categories 
	# X: counts of each category 
	# vec_errcoeff: array of precomputed and normed values for the error in the EM-sum for logL
	# LM_accuracy: required accuracy for the mantissa
	# output: m_opt: m value to obtain LM_accuracy
	# output: hor_shift: horizontal shift value to obtain LM_accuracy	

	# types and initializations
	N = int(0)
	hor_shift = int(0)
	max_hor_shift = int(1000)
	ok = int(0)
	j = int(0)
	m = int(0)
	toterr = np.float64(0)		
	toterr1 = np.float64(0)		

	N = find_N(X)	
	
	# LOOKING FOR m_opt AND THE HORIZONTAL SHIFT
	# starts with j=0 and look if there exists m such that the error in the Euler-Maclaurin formula
	# is less than the required accuracy. If not, j is incremented by 1, and it repeats the 
	# evaluation for the shifted case until it gets m, or it has used all the available j (<1000).
	# If we have a success (ok = 1): m becomes m_opt and j becomes the horizontal shift attached.
	# If an m like this does not exist with j up to 1000 (ok = 0): it returns an error code [0,0] 
	# that will be handled by the calling function
	# 
	while ok == 0 and j < max_hor_shift :
		m = 1 
		toterr = err(probs, psi, K, X, N, m , vec_errcoeff, j)
		m += 1
		toterr1 = err(probs, psi, K, X, N, m, vec_errcoeff, j) 
	
		while toterr1 < toterr and toterr > LM_accuracy and m < 100:
			# we have 100 precomputed coefficients; so m +1 < 100
			toterr = toterr1
			m += 1  
			toterr1 = err(probs, psi, K, X, N, m, vec_errcoeff, j) 
		#endwhile
			
		if toterr < LM_accuracy :
			ok = 1
			hor_shift = j # horizontal shift for this accuracy
			if m <= 100: 
				m_opt=m-1 # optimal m for this accuracy
			else:
				m_opt=100 # optimal m for this precision
			#endif
		#endif
		j += 1
	#endwhile
	
	if ok == 0: 
		print('Horizontal shift too large (>1000) for the required accuracy; ask for a smaller accuracy')		
		return [0, 0]			
	#endif	
		
	# minimal positive number in float
	epsilon  = math.ldexp(1.0, -53)
	min_err = max(abs((K+1)*toterr), epsilon )

	return [m_opt, hor_shift]
	


# main digamma-LM function  
def main_LM_digamma(probs, psi, K, X, vec_evenbernoullinormdigamma, LM_accuracy):
	#initialization
	resdigamma_LM = np.float64(0) 
	# calls the error/hor_shift eval function
	[LMm_digamma, hor_shift_digamma] = opt_err_LM_digamma(probs, psi, K, X, vec_errcoeff_digamma, LM_accuracy)
	print('LM_digamma optimal m for the requested mantissa accuracy =', LMm_digamma)	
	print('LM_digamma horizontal shift for the requested mantissa accuracy =', hor_shift_digamma)	
	if LMm_digamma == 0 :
		resdigamma_LM = 0	# error code for the Euler-Maclaurin formula not accurate enough
	else:
		# calls the digamma-eval function that uses the LM-technique
		resdigamma_LM = LM_digamma_dataset(probs, psi, K, X, LMm_digamma, vec_evenbernoullinormdigamma, hor_shift_digamma) 
	#endif
	return resdigamma_LM
	


###############################################################
####  ----------  END LM functions implementation   
############################################################### 


    
# IMPLEMENTATION of BERNOULLI POLYNOMIALS using functions; 
# further improved using an array of precomputed powers of the main variable (upow)


def phi_1(upow):
    return upow[1]
def phi_2(upow):
    return upow[2] - upow[1]
def phi_3(upow):
    return upow[3] - 3/2 * upow[2] + 1/2 * upow[1]
def phi_4(upow):
    return upow[4] - 2 * upow[3] + upow[2]
def phi_5(upow):
    return upow[5] - 5/2 * upow[4] + 5/3 * upow[3] - 1/6 * upow[1]
def phi_6(upow):
    return upow[6] - 3 * upow[5] + 5/2 * upow[4] - 1/2 * upow[2]
def phi_7(upow):
    return upow[7] - 7/2 * upow[6] + 7/2 * upow[5] - 7/6 * upow[3] + 1/6 * upow[1]
def phi_8(upow):
    return upow[8] - 4 * upow[7] + 14/3 * upow[6] - 7/3 * upow[4] + 2/3 * upow[2]
def phi_9(upow):
    return upow[9] - 9/2 * upow[8] + 6 * upow[7] - 21/5 * upow[5] + 2 * upow[3] - 3/10 * upow[1]
def phi_10(upow):
    return upow[10] - 5 * upow[9] + 15/2 * upow[8] - 7 * upow[6] + 5 * upow[4] - 3/2 * upow[2]
def phi_11(upow):
    return  upow[11] - 11/2 * upow[10] + 55/6 * upow[9] - 11 * upow[7] + 11 * upow[5] - 11/2 * upow[3] + 5/6 * upow[1]
def phi_12(upow):
    return upow[12] - 6 * upow[11] + 11 * upow[10] - 33/2 * upow[8] + 22 * upow[6] - 33/2 * upow[4] + 5 * upow[2]
def phi_13(upow):
    return upow[13] - 13/2 * upow[12] + 13 * upow[11] - 143/6 * upow[9] + 286/7 * upow[7] - 429/10 * upow[5] + 65/3 * upow[3] - 691/210 * upow[1]
def phi_14(upow):
    return upow[14] - 7 * upow[13] + 91/6 * upow[12] - 1001/30 * upow[10] + 143/2 * upow[8] - 1001/10 * upow[6] + 455/6 * upow[4] - 691/30 * upow[2]
def phi_15(upow):
    return upow[15] - 15/2 * upow[14] + 35/2 * upow[13] - 91/2 * upow[11] + 715/6 * upow[9] - 429/2 * upow[7] + 455/2 * upow[5] - 691/6 * upow[3] + 35/2 * upow[1]
def phi_16(upow):
    return upow[16] - 8 * upow[15] + 20 * upow[14] - 182/3 * upow[12] + 572/3 * upow[10] - 429 * upow[8] + 1820/3 * upow[6] - 1382/3 * upow[4] + 140 * upow[2]
def phi_17(upow):
    return upow[17] - 17/2 * upow[16] + 68/3 * upow[15] - 238/3 * upow[13] + 884/3 * upow[11] - 2431/3 * upow[9] + 4420/3 * upow[7] - 23494/15 * upow[5] + 2380/3 * upow[3] - 3617/30 * upow[1]
def phi_18(upow):
    return upow[18] - 9 * upow[17] + 51/2 * upow[16] - 102 * upow[14] + 442 * upow[12] - 7293/5 * upow[10] + 3315 * upow[8] - 23494/5 * upow[6] + 3570 * upow[4] - 10851/10 * upow[2]
def phi_19(upow):
    return upow[19] - 19/2 * upow[18] + 57/2 * upow[17] - 646/5 * upow[15] + 646 * upow[13] - 12597/5 * upow[11] + 20995/3 * upow[9] - 446386/35 * upow[7] + 13566 * upow[5] - 68723/10 * upow[3] + 43867/42 * upow[1]
def phi_20(upow):
    return  upow[20] - 10 * upow[19] + 95/3 * upow[18] - 323/2 * upow[16] + 6460/7 * upow[14] - 4199 * upow[12] + 41990/3 * upow[10] - 223193/7 * upow[8] + 45220 * upow[6] - 68723/2 * upow[4] + 219335/21 * upow[2]
def phi_21(upow):
	 return  upow[21] - 21/2 * upow[20] + 35 * upow[19] - 399/2 * upow[17] + 1292 * upow[15] - 6783 * upow[13] + 293930/11 * upow[11] - 223193/3 * upow[9] + 135660 * upow[7] - 1443183/10 * upow[5] + 219335/3 * upow[3] - 1222277/110 * upow[1]
def phi_22(upow):
	 return  upow[22] - 11 * upow[21] + 77/2 * upow[20] - 1463/6 * upow[18] + 3553/2 * upow[16] - 10659 * upow[14] + 146965/3 * upow[12] - 2455123/15 * upow[10] + 373065 * upow[8] - 5291671/10 * upow[6] + 2412685/6 * upow[4] - 1222277/10 * upow[2]
def phi_23(upow):
	 return  upow[23] - 23/2 * upow[22] + 253/6 * upow[21] - 1771/6 * upow[19] + 4807/2 * upow[17] - 81719/5 * upow[15] + 260015/3 * upow[13] - 5133439/15 * upow[11] + 2860165/3 * upow[9] - 17386919/10 * upow[7] + 11098351/6 * upow[5] - 28112371/30 * upow[3] + 854513/6 * upow[1]
def phi_24(upow):
	 return  upow[24] - 12 * upow[23] + 46 * upow[22] - 1771/5 * upow[20] + 9614/3 * upow[18] - 245157/10 * upow[16] + 148580 * upow[14] - 10266878/15 * upow[12] + 2288132 * upow[10] - 52160757/10 * upow[8] + 22196702/3 * upow[6] - 28112371/5 * upow[4] + 1709026 * upow[2]
def phi_25(upow):
	 return  upow[25] - 25/2 * upow[24] + 50 * upow[23] - 1265/3 * upow[21] + 12650/3 * upow[19] - 72105/2 * upow[17] + 742900/3 * upow[15] - 51334390/39 * upow[13] + 5200300 * upow[11] - 86934595/6 * upow[9] + 554917550/21 * upow[7] - 28112371 * upow[5] + 42725650/3 * upow[3] - 1181820455/546 * upow[1]
def phi_26(upow):
	 return  upow[26] - 13 * upow[25] + 325/6 * upow[24] - 1495/3 * upow[22] + 16445/3 * upow[20] - 312455/6 * upow[18] + 2414425/6 * upow[16] - 51334390/21 * upow[14] + 33801950/3 * upow[12] - 226029947/6 * upow[10] + 3606964075/42 * upow[8] - 365460823/3 * upow[6] + 277716725/3 * upow[4] - 1181820455/42 * upow[2]
def phi_27(up):
	 return  upow[27] - 27/2 * upow[26] + 117/2 * upow[25] - 585 * upow[23] + 49335/7 * upow[21] - 148005/2 * upow[19] + 1278225/2 * upow[17] - 30800634/7 * upow[15] + 23401350 * upow[13] - 184933593/2 * upow[11] + 3606964075/14 * upow[9] - 469878201 * upow[7] + 499890105 * upow[5] - 3545461365/14 * upow[3] + 76977927/2 * upow[1]
def phi_28(upow):
	 return  upow[28] - 14 * upow[27] + 63 * upow[26] - 1365/2 * upow[24] + 8970 * upow[22] - 207207/2 * upow[20] + 994175 * upow[18] - 15400317/2 * upow[16] + 46802700 * upow[14] - 431511717/2 * upow[12] + 721392815 * upow[10] - 3289147407/2 * upow[8] + 2332820490 * upow[6] - 3545461365/2 * upow[4] + 538845489 * upow[2]
def phi_29(upow):
	 return  upow[29] - 29/2 * upow[28] + 203/3 * upow[27] - 7917/10 * upow[25] + 11310 * upow[23] - 286143/2 * upow[21] + 1517425 * upow[19] - 26271129/2 * upow[17] + 90485220 * upow[15] - 962603061/2 * upow[13] + 1901853785 * upow[11] - 10598363867/2 * upow[9] + 9664542030 * upow[7] - 20563675917/2 * upow[5] + 5208839727 * upow[3] - 23749461029/30 * upow[1]
def phi_30(upow):
	 return  upow[30] - 15 * upow[29] + 145/2 * upow[28] - 1827/2 * upow[26] + 28275/2 * upow[24] - 390195/2 * upow[22] + 4552275/2 * upow[20] - 43785215/2 * upow[18] + 339319575/2 * upow[16] - 2062720845/2 * upow[14] + 9509268925/2 * upow[12] - 31795091601/2 * upow[10] + 72484065225/2 * upow[8] - 102818379585/2 * upow[6] + 78132595905/2 * upow[4] - 23749461029/2 * upow[2]
def phi_31(upow):
	 return  upow[31] - 31/2 * upow[30] + 155/2 * upow[29] - 6293/6 * upow[27] + 35061/2 * upow[25] - 525915/2 * upow[23] + 6720025/2 * upow[21] - 71439035/2 * upow[19] + 618759225/2 * upow[17] - 4262956413/2 * upow[15] + 22675948975/2 * upow[13] - 985647839631/22 * upow[11] + 249667335775/2 * upow[9] - 3187369767135/14 * upow[7] + 484422094611/2 * upow[5] - 736233291899/6 * upow[3] + 8615841276005/462 * upow[1]
def phi_32(upow):
	 return  upow[32] - 16 * upow[31] + 248/3 * upow[30] - 3596/3 * upow[28] + 21576 * upow[26] - 350610 * upow[24] + 53760200/11 * upow[22] - 57151228 * upow[20] + 550008200 * upow[18] - 4262956413 * upow[16] + 181407591800/7 * upow[14] - 1314197119508/11 * upow[12] + 399467737240 * upow[10] - 6374739534270/7 * upow[8] + 1291792252296 * upow[6] - 2944933167596/3 * upow[4] + 68926730208040/231 * upow[2]
def phi_33(upow):
	 return  upow[33] - 33/2 * upow[32] + 88 * upow[31] - 1364 * upow[29] + 79112/3 * upow[27] - 2314026/5 * upow[25] + 7012200 * upow[23] - 628663508/7 * upow[21] + 955277400 * upow[19] - 140677561629/17 * upow[17] + 399096701960/7 * upow[15] - 303276258348 * upow[13] + 1198403211720 * upow[11] - 23374044958990/7 * upow[9] + 6089877760824 * upow[7] - 32394264843556/5 * upow[5] + 68926730208040/21 * upow[3] - 84802531453387/170 * upow[1]
def phi_34(upow):
	 return  upow[34] - 17 * upow[33] + 187/2 * upow[32] - 23188/15 * upow[30] + 672452/21 * upow[28] - 3026034/5 * upow[26] + 9933950 * upow[24] - 971570876/7 * upow[22] + 1623971580 * upow[20] - 15630840181 * upow[18] + 848080491665/7 * upow[16] - 736528055988 * upow[14] + 3395475766540 * upow[12] - 79471752860566/7 * upow[10] + 25881980483502 * upow[8] - 550702502340452/15 * upow[6] + 585877206768340/21 * upow[4] - 84802531453387/10 * upow[2]
def phi_35(upow):
	 return  upow[35] - 35/2 * upow[34] + 595/6 * upow[33] - 5236/3 * upow[31] + 115940/3 * upow[29] - 2353582/3 * upow[27] + 13907530 * upow[25] - 211211060 * upow[23] + 2706619300 * upow[21] - 28793652965 * upow[19] + 249435438725 * upow[17] - 1718565463972 * upow[15] + 9141665525300 * upow[13] - 36123524027530 * upow[11] + 100652146324730 * upow[9] - 550702502340452/3 * upow[7] + 585877206768340/3 * upow[5] - 593617720173709/6 * upow[3] + 90219075042845/6 * upow[1]
def phi_36(upow):
	 return  upow[36] - 18 * upow[35] + 105 * upow[34] - 3927/2 * upow[32] + 46376 * upow[30] - 1008678 * upow[28] + 19256580 * upow[26] - 316816590 * upow[24] + 4429013400 * upow[22] - 51828575337 * upow[20] + 498870877450 * upow[18] - 3866772293937 * upow[16] + 23507139922200 * upow[14] - 108370572082590 * upow[12] + 362347726769028 * upow[10] - 826053753510678 * upow[8] + 1171754413536680 * upow[6] - 1780853160521127/2 * upow[4] + 270657225128535 * upow[2]
def phi_37(upow):
	 return  upow[37] - 37/2 * upow[36] + 111 * upow[35] - 4403/2 * upow[33] + 55352 * upow[31] - 1286934 * upow[29] + 79165940/3 * upow[27] - 2344442766/5 * upow[25] + 7124934600 * upow[23] - 91317013689 * upow[21] + 18458222465650/19 * upow[19] - 8415916169157 * upow[17] + 57984278474760 * upow[15] - 4009711167055830/13 * upow[13] + 1218805990041276 * upow[11] - 10187996293298362/3 * upow[9] + 43354913300857160/7 * upow[7] - 65891566939281699/10 * upow[5] + 3338105776585265 * upow[3] - 26315271553053477373/51870 * upow[1]
def phi_38(upow):
	 return  upow[38] - 19 * upow[37] + 703/6 * upow[36] - 4921/2 * upow[34] + 131461/2 * upow[32] - 8150582/5 * upow[30] + 107439490/3 * upow[28] - 44544412554/65 * upow[26] + 11281146450 * upow[24] - 157729387281 * upow[22] + 1845822246565 * upow[20] - 17766934134887 * upow[18] + 137712661377555 * upow[16] - 76184512174060770/91 * upow[14] + 3859552301797374 * upow[12] - 193571929572668878/15 * upow[10] + 205935838179071510/7 * upow[8] - 417313257282117427/10 * upow[6] + 63424009755120035/2 * upow[4] - 26315271553053477373/2730 * upow[2]
def phi_39(upow):
	 return  upow[39] - 39/2 * upow[38] + 247/2 * upow[37] - 27417/10 * upow[35] + 155363/2 * upow[33] - 10253958/5 * upow[31] + 48162530 * upow[29] - 14848137518/15 * upow[27] + 17598588462 * upow[25] - 267454178433 * upow[23] + 23995689205345/7 * upow[21] - 36468970066347 * upow[19] + 315929046689685 * upow[17] - 15236902434812154/7 * upow[15] + 11578656905392122 * upow[13] - 228766825858608674/5 * upow[11] + 2677165896327929630/21 * upow[9] - 2325031004857511379/10 * upow[7] + 494707276089936273/2 * upow[5] - 26315271553053477373/210 * upow[3] + 38089920879940267/2 * upow[1]
def phi_40(upow):
	 return  upow[40] - 20 * upow[39] + 130 * upow[38] - 9139/3 * upow[36] + 91390 * upow[34] - 5126979/2 * upow[32] + 192650120/3 * upow[30] - 29696275036/21 * upow[28] + 27074751480 * upow[26] - 445756964055 * upow[24] + 43628525827900/7 * upow[22] - 72937940132694 * upow[20] + 702064548199300 * upow[18] - 38092256087030385/7 * upow[16] + 33081876872548920 * upow[14] - 457533651717217348/3 * upow[12] + 10708663585311718520/21 * upow[10] - 2325031004857511379/2 * upow[8] + 1649024253633120910 * upow[6] - 26315271553053477373/21 * upow[4] + 380899208799402670 * upow[2]
def phi_41(upow):
	 return  upow[41] - 41/2 * upow[40] + 410/3 * upow[39] - 10127/3 * upow[37] + 749398/7 * upow[35] - 6369883/2 * upow[33] + 254795320/3 * upow[31] - 41984388844/21 * upow[29] + 123340534520/3 * upow[27] - 3655207105251/5 * upow[25] + 77772589519300/7 * upow[23] - 142402645020974 * upow[21] + 1514981393482700 * upow[19] - 91869558798132105/7 * upow[17] + 90423796784967048 * upow[15] - 1442990747723531636/3 * upow[13] + 439055206997780459320/231 * upow[11] - 31775423733052655513/6 * upow[9] + 9658570628422565330 * upow[7] - 1078926133675192572293/105 * upow[5] + 15616867560775509470/3 * upow[3] - 261082718496449122051/330 * upow[1]
def phi_42(upow):
	 return  upow[42] - 21 * upow[41] + 287/2 * upow[40] - 3731 * upow[38] + 374699/3 * upow[36] - 7868679/2 * upow[34] + 222945905/2 * upow[32] - 41984388844/15 * upow[30] + 61670267260 * upow[28] - 5904565323867/5 * upow[26] + 19443147379825 * upow[24] - 2990455545440454/11 * upow[22] + 3181460926313670 * upow[20] - 30623186266044035 * upow[18] + 237362466560538501 * upow[16] - 1442990747723531636 * upow[14] + 219527603498890229660/33 * upow[12] - 222427966131368588591/10 * upow[10] + 101414991598436935965/2 * upow[8] - 1078926133675192572293/15 * upow[6] + 54659036462714283145 * upow[4] - 1827579029475143854357/110 * upow[2]
def phi_43(upow):
	 return  upow[43] - 43/2 * upow[42] + 301/2 * upow[41] - 12341/3 * upow[39] + 435461/3 * upow[37] - 48336171/10 * upow[35] + 9586673915/66 * upow[33] - 58236410332/15 * upow[31] + 91442120420 * upow[29] - 28210700991809/15 * upow[27] + 33442213493299 * upow[25] - 5590851671910414/11 * upow[23] + 6514419991975610 * upow[21] - 69305105759994395 * upow[19] + 600387415417832679 * upow[17] - 62048602152111860348/15 * upow[15] + 726129765419406144260/33 * upow[13] - 869491140331713573583/10 * upow[11] + 1453614879577596082165/6 * upow[9] - 46393823748033280608599/105 * upow[7] + 470067713579342835047 * upow[5] - 78585898267431185737351/330 * upow[3] + 1520097643918070802691/42 * upow[1]
def phi_44(upow):
	 return  upow[44] - 22 * upow[43] + 473/3 * upow[42] - 135751/30 * upow[40] + 504218/3 * upow[38] - 177232627/30 * upow[36] + 563921995/3 * upow[34] - 160150128413/30 * upow[32] + 402345329848/3 * upow[30] - 44331101558557/15 * upow[28] + 56594515142506 * upow[26] - 931808611985069 * upow[24] + 13028839983951220 * upow[22] - 152471232671987669 * upow[20] + 1467613682132479882 * upow[18] - 170633655918307615957/15 * upow[16] + 1452259530838812288520/21 * upow[14] - 9564402543648849309413/30 * upow[12] + 3197952735070711380763/3 * upow[10] - 510332061228366086694589/210 * upow[8] + 10341489698745542371034/3 * upow[6] - 78585898267431185737351/30 * upow[4] + 16721074083098778829601/21 * upow[2]
def phi_45(upow):
	 return  upow[45] - 45/2 * upow[44] + 165 * upow[43] - 9933/2 * upow[41] + 193930 * upow[39] - 14370213/2 * upow[37] + 241680855 * upow[35] - 14559102583/2 * upow[33] + 194683224120 * upow[31] - 4585976023299 * upow[29] + 282972575712530/3 * upow[27] - 8386277507865621/5 * upow[25] + 586297799277804900/23 * upow[23] - 2287068490079815035/7 * upow[21] + 3475927141892715510 * upow[19] - 30111821632642520463 * upow[17] + 1452259530838812288520/7 * upow[15] - 2207169817765119071403/2 * upow[13] + 4360844638732788246495 * upow[11] - 510332061228366086694589/42 * upow[9] + 22160335068740447937930 * upow[7] - 235757694802293557212053/10 * upow[5] + 83605370415493894148005/7 * upow[3] - 83499808737903072705069/46 * upow[1]
def phi_46(upow):
	 return  upow[46] - 23 * upow[45] + 345/2 * upow[44] - 10879/2 * upow[42] + 446039/2 * upow[40] - 17395521/2 * upow[38] + 1852886555/6 * upow[36] - 19697609377/2 * upow[34] + 559714269345/2 * upow[32] - 35159149511959/5 * upow[30] + 3254184620694095/21 * upow[28] - 14837260206223791/5 * upow[26] + 48858149939817075 * upow[24] - 4782052297439613255/7 * upow[22] + 7994632426353245673 * upow[20] - 230857299183592656883/3 * upow[18] + 4175246151161585329495/7 * upow[16] - 7252129401228248377467/2 * upow[14] + 33433142230284709889795/2 * upow[12] - 11737637408252419993975547/210 * upow[10] + 254843853290515151286195/2 * upow[8] - 1807475660150917271959073/10 * upow[6] + 1922923519556359565404115/14 * upow[4] - 83499808737903072705069/2 * upow[2]
def phi_47(upow):
	 return  upow[47] - 47/2 * upow[46] + 1081/6 * upow[45] - 11891/2 * upow[43] + 511313/2 * upow[41] - 20963833/2 * upow[39] + 2353666705/6 * upow[37] - 925787640719/70 * upow[35] + 797168807855/2 * upow[33] - 53305807324583/5 * upow[31] + 5274023350780085/21 * upow[29] - 77483469965835353/15 * upow[27] + 91853321886856101 * upow[25] - 9772019912159209695/7 * upow[23] + 17892748763742978411 * upow[21] - 571068055875202888079/3 * upow[19] + 11543327594387912381545/7 * upow[17] - 113616693952575891246983/10 * upow[15] + 120873668063337028063105/2 * upow[13] - 50151723471623976337895519/210 * upow[11] + 3992553701551404036817055/6 * upow[9] - 12135908003870444540296633/10 * upow[7] + 18075481083829779914798681/14 * upow[5] - 1308163670227148139046081/2 * upow[3] + 596451111593912163277961/6 * upow[1]
def phi_48(upow):
	 return  upow[48] - 24 * upow[47] + 188 * upow[46] - 6486 * upow[44] + 2045252/7 * upow[42] - 62891499/5 * upow[40] + 495508780 * upow[38] - 1851575281438/105 * upow[36] + 562707393780 * upow[34] - 159917421973749/10 * upow[32] + 8438437361248136/21 * upow[30] - 44276268551905916/5 * upow[28] + 169575363483426648 * upow[26] - 19544039824318419390/7 * upow[24] + 39038724575439225624 * upow[22] - 2284272223500811552316/5 * upow[20] + 92346620755103299052360/21 * upow[18] - 340850081857727673740949/10 * upow[16] + 207212002394292048108180 * upow[14] - 100303446943247952675791038/105 * upow[12] + 3194042961241123229453644 * upow[10] - 36407724011611333620889899/5 * upow[8] + 72301924335319119659194724/7 * upow[6] - 7848982021362888834276486 * upow[4] + 2385804446375648653111844 * upow[2]
def phi_49(upow):
	 return  upow[49] - 49/2 * upow[48] + 196 * upow[47] - 105938/15 * upow[45] + 332948 * upow[43] - 75163011/5 * upow[41] + 1867686940/3 * upow[39] - 350298026218/15 * upow[37] + 787790351292 * upow[35] - 237453141718597/10 * upow[33] + 1905453597701192/3 * upow[31] - 74811626173909996/5 * upow[29] + 923243645631989528/3 * upow[27] - 27361655754045787146/5 * upow[25] + 83169456704196611112 * upow[23] - 15989905564505680866212/15 * upow[21] + 34022439225564373335080/3 * upow[19] - 16701654011028656013306501/170 * upow[17] + 676892541154687357153388 * upow[15] - 702124128602735668730537266/195 * upow[13] + 14228009554619548931202596 * upow[11] - 594659492189651782474535017/15 * upow[9] + 72301924335319119659194724 * upow[7] - 384600119046781552879547814/5 * upow[5] + 116904417872406784002480356/3 * upow[3] - 39265823582984723803743892829/6630 * upow[1]
def phi_50(upow):
	 return  upow[50] - 25 * upow[49] + 1225/6 * upow[48] - 23030/3 * upow[46] + 378350 * upow[44] - 17895955 * upow[42] + 2334608675/3 * upow[40] - 92183691110/3 * upow[38] + 3282459797050/3 * upow[36] - 1187265708592985/34 * upow[34] + 5954542492816225/6 * upow[32] - 74811626173909996/3 * upow[30] + 1648649367199981300/3 * upow[28] - 136808278770228935730/13 * upow[26] + 173269701467076273150 * upow[24] - 7268138892957127666460/3 * upow[22] + 85056098063910933337700/3 * upow[20] - 27836090018381093355510835/102 * upow[18] + 4230578382216795982208675/2 * upow[16] - 501517234716239763378955190/39 * upow[14] + 177850119432744361640032450/3 * upow[12] - 594659492189651782474535017/3 * upow[10] + 451887027095744497869967025 * upow[8] - 641000198411302588132579690 * upow[6] + 1461305223405084800031004450/3 * upow[4] - 196329117914923619018719464145/1326 * upow[2]

def find_phi(upow,m):
    match m:
        case 1: return phi_1(upow)
        case 2: return phi_2(upow)
        case 3: return phi_3(upow)
        case 4: return phi_4(upow)
        case 5: return phi_5(upow)
        case 6: return phi_6(upow)
        case 7: return phi_7(upow)
        case 8: return phi_8(upow)
        case 9: return phi_9(upow)
        case 10: return phi_10(upow)
        case 11: return phi_11(upow)
        case 12: return phi_12(upow)
        case 13: return phi_13(upow)
        case 14: return phi_14(upow)
        case 15: return phi_15(upow)
        case 16: return phi_16(upow)
        case 17: return phi_17(upow)
        case 18: return phi_18(upow)
        case 19: return phi_19(upow)
        case 20: return phi_20(upow)
        case 21: return phi_21(upow)
        case 22: return phi_22(upow)
        case 23: return phi_23(upow)
        case 24: return phi_24(upow)
        case 25: return phi_25(upow)
        case 26: return phi_26(upow)
        case 27: return phi_27(upow)
        case 28: return phi_28(upow)
        case 29: return phi_29(upow)
        case 30: return phi_30(upow)
        case 31: return phi_31(upow)
        case 32: return phi_32(upow)
        case 33: return phi_33(upow)
        case 34: return phi_34(upow)
        case 35: return phi_35(upow)
        case 36: return phi_36(upow)
        case 37: return phi_37(upow)
        case 38: return phi_38(upow)
        case 39: return phi_39(upow)
        case 40: return phi_40(upow)    
        case 41: return phi_41(upow)
        case 42: return phi_42(upow)
        case 43: return phi_43(upow)
        case 44: return phi_44(upow)
        case 45: return phi_45(upow)
        case 46: return phi_46(upow)
        case 47: return phi_47(upow)
        case 48: return phi_48(upow)
        case 49: return phi_49(upow)
        case 50: return phi_50(upow)            
    return -1
    



###############################################################		
##  BEGIN COMPARISON FUNCTIONS
###############################################################

##  they directly use the lngamma function of mpmath or math
##  with no mesh of points or the Euler-Maclaurin formula

# paired difference of logGammas 64 bits (math package; C compiled)
# prone to obtain wrong results for large overdispersed datasets
def Diff64(x, y): 
	z = math.lgamma(1/x+y)-math.lgamma(1/x)
	return z

def ver_logL_64(N, psi, psioverprobs, X, K):
    z =  - Diff64(psi,N)
    for k in range (0, K):
        z += Diff64(psioverprobs[k], X[k])
	#endfor
    return z
	
# paired difference of logGammas 64 bits (scipy package; C compiled)
# prone to obtain wrong results for large overdispersed datasets
def Diffscipy64(x, y): 
	z = sp.loggamma(1/x+y)-sp.loggamma(1/x)
	return z

def ver_logLscipy64(N, psi, psioverprobs, X, K):
	z =  - Diffscipy64(psi,N) 
	for k in range (0, K):  
		z += Diffscipy64(psioverprobs[k], X[k])
	#endfor	
	return z	


# paired difference of logGammas mpmath (multiprecision)
def Diff(x, y): 
	z = lngamma(1/x+y)-lngamma(1/x)
	return z

def ver_logL(N, psi, psioverprobs, X, K):
	z =  - Diff(psi,N) 
	for k in range (0, K):  
		z += Diff(psioverprobs[k], X[k])
	#endfor	
	return z

###############################################################		
##  END COMPARISON FUNCTIONS
###############################################################
	
#######################################################################         
#################  FUNCTIONS FOR ESTIMATING PSI USING #################
#################  MINKA'S FIXED-POINT PROCEDURES     #################
#######################################################################

def find_psi(alpha, K):
    # input: Dirchilet parameter vector alpha, number of components is K
    # input: K: number of categories
    
    # output: overdispersion parameter psi
 
    _psi = 1.0/np.sum(alpha)

    if(_psi < 0): #_alpha_fail was the input
        print("invalid alpha, returning psi of -1");
        return -1
    return _psi

def find_P_est(alpha, psi, K):
    # input: Dirchilet parameter vector alpha, number of components is K
    # input: overdispersion parameter psi
    # input: K: number of categories
    
    # output: estimated probabilities vector

    P = np.zeros(K, dtype=np.float64) #[0]*K
    
    P = np.multiply(alpha, psi)

    return P

def _init_a(D_counts, delta, YS_fails):
    # Initial guess for Dirichlet alpha parameters given counts matrix D_counts
    # input: d_counts, matrix of counts, with R rows and K columns
    # input: delta from YS algorithm (if a component in alpha < 1/delta, we set it to 1/delta, to make YS executable!
    # input: one-value vector YS_fails, incremented by one for each time delta-mesh error happens in an _init_a call

    # output: result vector of K columns, the initial guess for Dirichlet alpha
    
    R, K = D_counts.shape
    alpha_fail = np.empty(K, dtype=float)  # initialization
    for k in range(K):
        alpha_fail[k] = -1.0
    # this function is from Minka's python code online, it assumes D_counts is a probablity (ratios) not counts matrix
    D_probs = counts_to_probs(D_counts)
    factor = 1
    E = D_probs.mean(axis=0)
    E2 = (D_probs ** 2).mean(axis=0)
    
    result = factor*(((E[0] - E2[0]) / (E2[0] - E[0] ** 2)) * E)


    first_time = 0
    for k in range (K):
        if (result[k] < 1.0/delta):
            if(first_time == 0):
                print('*********** ERROR from _init_a: delta-mesh not applicable ***********')
                first_time = 1
                YS_fails[0] = YS_fails[0]+1
            result[k] = 1/delta        

   
    return result

def _init_a_LM(D_counts):
    # Initial guess for Dirichlet alpha parameters given counts matrix D_counts
    # input: d_counts, matrix of counts, with R rows and K columns

    # output: result vector of K columns, the initial guess for Dirichlet alpha
    
    R, K = D_counts.shape
   
    D_probs = counts_to_probs(D_counts)
    factor = 1
    E = D_probs.mean(axis=0)
    E2 = (D_probs ** 2).mean(axis=0)
    
    result = factor*(((E[0] - E2[0]) / (E2[0] - E[0] ** 2)) * E)


    return result

#check if requested precision can be accommodated using asymp
def check_with_asym(bound, X, psival,probs, K ):
    #input: bound: number of mantissa decimal digits to represent logL(SIGDIG - precision)
    #input: X: vector of K counts
    #input: psival: value of psi
    #input: probs: vector of K probabilities, they must add up to 1, obviously

    #output: returns 1 if check passes, and -1 otherwise
    
    asymp = np.float64(0)
    psioverprobs = np.zeros(K, dtype=np.float64)
    alpha = np.zeros(K, dtype=np.float64)

    for k in range (0, K):  
        psioverprobs[k] = psival/probs[k] # needed for the D_m sum
        alpha[k] = probs[k]/psival  # alpha vector          
        asymp += X[k]*log(probs[k]) # computing the asymptotic state of the system as psi->0+   
        #endfor
    if log10(abs(asymp)) > bound :
        print('*********** ERROR (check_with_asym): LogL too large to ensure the desired precision in double; switch to multiprecision')
        print('*********** ABORTING')
        return -1
        #endif
    return 1

#check if requested precision can be accommodated using asymp1
def check_with_asym1(bound, X, K ):
    #input: bound: number of mantissa decimal digits to represent logL(SIGDIG - precision)
    #input: X: vector of K counts
    #input: psival: value of psi
    #input: probs: vector of K probabilities, they must add up to 1, obviously


    #output: returns 1 if check passes, and -1 otherwise

   N = int(0)
   for k in range (0, K):
       N = N + X[k]
       #endfor
       
   asymp1 = np.float64(0)
   for k in range (0, K):  
        asymp1 = asymp1 + X[k]*log(X[k]/N)  # computing the asymptotic state of the system as psi->0+ with the frequencies
        #endfor

   if log10(abs(asymp1)) > bound :
        print('*********** ERROR (check_with_asym1): LogL too large to ensure the desired precision in double; switch to multiprecision')
        print('*********** ABORTING')
        return -1
        #endif
   return 1



# new C-warrped minka_procedure_LM_mode
def minka_procedure_LM_c_wrapped_mode(prec, tolerance, D_counts, maxiter,  R, K):
        #adapted from Alessandro's run_experiment function
        #input: prec is the requested number of decimal digits of accuracy
        #tolerance: upper bound on diff between two consecutive alphas (equivalent to LM_accuracy from UL notes)
        #input: D_counts: the matrix of counts, with K columns (i.e. categories), R rows(i.e. instances)
        #input: K : the number of categories
        #input: R : number of rows in D_counts
        #input: maxiter: maximum number of ietrations for Minka's procedure
        

        #output: alpha, estimated Dirichlet parameter vector, or a vector of K (-1) values if the procedure fails to converge, or fails to compute logL

   


    # finish settings for wrapping
    LM_horshift.loggamma_LM.restype = ctypes.c_double#ctypes.c_int
    LM_horshift.loggamma_LM.argtypes = [ ctypes.POINTER(ctypes.c_double*K), ctypes.c_double]#, ctypes.POINTER(ctypes.c_double*100), ctypes.POINTER(ctypes.c_double*100)]

    LM_horshift.initBern_logL.restype = None
    LM_horshift.initBern_logL.argtypes = []

    LM_horshift.init_params.restype = ctypes.c_long
    LM_horshift.init_params.argtypes = [ctypes.c_int, ctypes.POINTER(ctypes.c_double*K), ctypes.c_int]
    
    print("using the LM C-wrapped version ");
    
  
    
    start_time = perf_counter()
    print("precision is ", prec, " tolerance (i.e. LM_accuracy) is ", tolerance, " maxiter is ", maxiter, " R is ", R, " K is ", K);
    res = mp.mpf(0) # mpmath variable
    res64 = np.float64(0)
    
    resdigamma = mp.mpf(0) # mpmath variable
    restrigamma = mp.mpf(0) # mpmath variable
    
    res_LM = np.float64(0)  
    LM_accuracy = tolerance 
    LMm = int(0)
    horshift = int(0)
    resdigamma_LM = np.float64(0)
    LMm_digamma = int(0)
    horshift_digamma = int(0)
    restrigamma_LM = np.float64(0)
    LMm_trigamma = int(0)
    horshift_trigamma = int(0)
    #initialize the LM vectors
    [vec_evenbernoulli, vec_evenbernoullinorm, vec_evenbernoullinormdigamma, vec_errcoeff] = LM_initBern() 
    common_err = np.float64(0)
    
    N = np.float64(0)   
    mplusone = int(0)
    SIGDIG  = int(15) # number of significative decimal digits in C-double (float)
    bound  = int(0)

    
 
    # find vector of counts X using D_counts
    X = np.zeros(K, dtype=np.float64)
    X = find_X(D_counts)
    N = find_N(X)



    #initialize Bern. numbers for the C-wrapped loggamma-LM
    LM_horshift.initBern_logL();
    
    
    #initialize params
    X_ctypes =  X.ctypes.data_as(ctypes.POINTER(ctypes.c_double*K))
    NN = np.float64(0)  
    NN = LM_horshift.init_params(prec, X_ctypes, K)
    

   
    #alpha vectors
    #alpha_0: a K-value vector of initially guessed alpha values, using Minka's original method
    alpha_1 = np.zeros(K, dtype=np.float64)
    alpha_0 = _init_a_LM(D_counts)
    alpha_fail = np.empty(K, dtype=float)  # initialization
    for k in range(K):
        alpha_fail[k] = -1.0

    # checking if the problem can be handled with the C double precision (float)
    SIGDIG = 15 # number of available digits in float
    bound = SIGDIG - prec - 2 # 2 is experimentally determined on gcc on INTEL i7
    
    if (check_with_asym1(bound, X, K ) == -1) :
        print('*********** ERROR FROM MINKA by LM PROCEDURE: LogL too large to ensure the desired precision in double; switch to multiprecision')
        return alpha_fail
       

    diff = np.float64(0)

    #totoal loggamma_LM time accumulator
    tot_LM_time = np.float64(0)
    
 
    
    # Start Minka's procedure (LM mode)
    if maxiter is None:
        maxiter = MAXINT
    for j in range(maxiter):
          sum_a0 = np.float64(0)
          
          for k in range(K):
            sum_a0 = sum_a0 + alpha_0[k]

          for k in range(K):
              
            sum_i = np.float64(0)
            for i in range(R):
                y = alpha_0[k]
                x = D_counts[i][k]
                diff = scipydigamma64(x+y)- scipydigamma64(y)
                sum_i = sum_i + diff 
                
            sum_i_2 = np.float64(0)
            for i in range(R):
                n_i = find_ni(D_counts, i)
                x = n_i 
                y = sum_a0
                diff = scipydigamma64(x+y)- scipydigamma64(y)
                sum_i_2 = sum_i_2 + diff
               
            alpha_1[k] = alpha_0[k] * (sum_i/sum_i_2)
         

        
          psi_1_LM = find_psi(alpha_1, K)
          P_est_1_LM = find_P_est(alpha_1, psi_1_LM, K)

          psi_0_LM = find_psi(alpha_0, K)
          P_est_0_LM = find_P_est(alpha_0, psi_0_LM, K)

     
        
          psioverprobs_0 = np.zeros(K, dtype=np.float64)
          psioverprobs_1 = np.zeros(K, dtype=np.float64)
          for k in range(K):
              psioverprobs_0[k] = psi_0_LM/P_est_0_LM[k]
              psioverprobs_1[k] = psi_1_LM/P_est_1_LM[k]
              
         
          
          #casting to ctypes before calling C-wrapped LM (loggamma_LM after calling pass_params )    

          
          P_est_0_LM_ctypes = P_est_0_LM.ctypes.data_as(ctypes.POINTER(ctypes.c_double*K))
          P_est_1_LM_ctypes = P_est_1_LM.ctypes.data_as(ctypes.POINTER(ctypes.c_double*K)) 
          
          
          # LM logL computation
          start_time_0 = perf_counter()
          log_L_0 = LM_horshift.loggamma_LM( P_est_0_LM_ctypes,psi_0_LM)
          if(log_L_0 == 99):
              print("loggamma in C crashed, cannot achieve precision");
              sys.exit()
          end_time_0 = perf_counter()
          LMexec_time_0 = end_time_0 - start_time_0 


          start_time_1 = perf_counter()
          log_L_1 = LM_horshift.loggamma_LM(P_est_1_LM_ctypes,psi_1_LM)
          if(log_L_1 == 99):
              print("loggamma in C crashed, cannot achieve precision");
              sys.exit()
          end_time_1 = perf_counter()
          LMexec_time_1 = end_time_1 - start_time_1

          tot_LM_time = tot_LM_time + LMexec_time_0 + LMexec_time_1

        

          _abs = abs((log_L_0) - (log_L_1))
  
          if  (_abs < tolerance):
              end_time = perf_counter()
              duration = tot_LM_time
              print("time = ", duration, " seconds, and ", duration/60, " minutes");
              print("iterations ", j);
              return alpha_1

          #distance between consecutive alphas not close enough, prep for next iteration
          for k in range(K):
             alpha_0[k] = alpha_1[k]
            
    #raise NotConvergingError(
    print("Failed to converge after {} iterations, values are {}.".format(maxiter, alpha_1));

    
def run_minka_LM(_file, _delimiter, prec,tolerance,  maxiter, wrapped):

     # input: _file containing the dataset
     # input: _delimiter: delimiter inside _file
     # input: prec: precision of deciaml digits (after decimal dot)
     # input: tolerance: upper bound on diff between two consecutive alphas (equivalent to LM_accuracy)
     # input: maxiter: number of iterations for the fixed-point procedure
     # input: wrapped: boolean that is equal to one if the C-wrapped version of LM-loggamma is to be used, and zero if its Python counterpart is to be used instead
   
     #output: estimated alpha using the LM logL function( if computable, or alpha_fail otherwise)
    D_counts, N = load_dataset_from_file(_file, _delimiter)
    R,K = D_counts.shape

    if(wrapped == 1):
         _alpha =  minka_procedure_LM_c_wrapped_mode(prec, tolerance, D_counts, maxiter,  R, K)
    else:
         _alpha = minka_procedure_LM_mode(prec, tolerance, D_counts, maxiter, R, K)
    if(sum(_alpha)== -1.0*K): # minka failed
        print("Error in Minka by LM: alpha is not computable, try a different dataset and/or a smaller precision");
    
    return _alpha   
      
def minka_LM_experiment_from_file(_file, _delimiter, prec, tolerance, maxiter, wrapped): #trump_biden_2020_training.csv
     # executes  run_minka_LM on using training dataset stored in _file
     # input: prec: precision of deciaml digits (after decimal dot)
     # input: tolerance: upper bound on diff between two consecutive alphas (equivalent to LM_accuracy)
     # input: maxiter: number of iterations for the fixed-point procedure
     # input: wrapped: boolean that is equal to one if the C-wrapped version of LM-loggamma is to be used, and zero if its Python counterpart is to be used instead


     
     # returns respective alpha and psi using the LM logL function( if computable, or alpha_fail and -1 otherwise)

    _alpha = run_minka_LM(_file, _delimiter, prec, tolerance, maxiter, wrapped)
    K = _alpha.shape
    _psi = find_psi(_alpha, K)

    
    if(_psi== -1):
         print("Minka by LM error: invalid psi, terminate the program")
         exit()
    print("psi = ",  _psi);
    return _alpha, _psi

################ end of Minka by LM #############################################

def minka_procedure_DEFAULT_mode(prec, tolerance, D_counts, maxiter, R, K, scipy_mode):
        #adapted from Alessandro's run_experiment function
        #input: prec is the requested number of decimal digits of accuracy
        #tolerance: upper bound on diff between two consecutive alphas (equivalent to LM_accuracy from UL notes)
        #input: D_counts: the matrix of counts, with K columns (i.e. categories), R rows(i.e. instances)
        #input: K : the number of categories
        #input: R : number of rows in D_counts
        #input: maxiter: maximum number of ietrations for Minka's procedure
        #input: scipy_mode:  0 means we use math loggamma,  1 means we use scipy loggamma
        

        #output: alpha, estimated Dirichlet parameter vector, or a vector of K (-1) values if the procedure fails to converge, or fails to compute logL

   


    start_time = perf_counter()
    print("precision is ", prec, " tolerance (i.e. LM_accuracy) is ", tolerance, " maxiter is ", maxiter, " R is ", R, " K is ", K);
    res = mp.mpf(0) # mpmath variable
    res64 = np.float64(0)
    
    resdigamma = mp.mpf(0) # mpmath variable
    restrigamma = mp.mpf(0) # mpmath variable
  
    common_err = np.float64(0)
    
    N = np.float64(0)   
    mplusone = int(0)
    SIGDIG  = int(15) # number of significative decimal digits in C-double (float)
    bound  = int(0)

    
 
    # find vector of counts X using D_counts
    X = find_X(D_counts)
    N = find_N(X)
  
          

    #alpha vectors
    #alpha_0: a K-value vector of initially guessed alpha values, using Minka's original method
    alpha_1 = np.zeros(K, dtype=np.float64)
    alpha_0 = _init_a_LM(D_counts) 
    alpha_fail = np.empty(K, dtype=float)  # initialization
    for k in range(K):
        alpha_fail[k] = -1.0

    # checking if the problem can be handled with the C double precision (float)
    bound = SIGDIG - prec  #
    
    if (check_with_asym1(bound, X, K ) == -1) :
        print('*********** ERROR FROM MINKA by DEFAULT PROCEDURE: LogL too large to ensure the desired precision in double; switch to multiprecision')
        return alpha_fail
       

    diff = np.float64(0)
    tot_default_time = np.float64(0)
 
    
    # Start Minka's procedure (DEFAULT mode)
    if maxiter is None:
        maxiter = MAXINT
    for j in range(maxiter):
          sum_a0 = np.float64(0)
          
          for k in range(K):
            sum_a0 = sum_a0 + alpha_0[k]

          for k in range(K):
              
            sum_i = np.float64(0)
            for i in range(R):
                y = alpha_0[k]
                x = D_counts[i][k]
                diff = scipydigamma64(x+y)- scipydigamma64(y)
                sum_i = sum_i + diff 
                
            sum_i_2 = np.float64(0)
            for i in range(R):
                n_i = find_ni(D_counts, i)
                x = n_i 
                y = sum_a0
                diff = scipydigamma64(x+y)- scipydigamma64(y)
                sum_i_2 = sum_i_2 + diff
               
            alpha_1[k] = alpha_0[k] * (sum_i/sum_i_2)
          

          psi_1_DEFAULT = find_psi(alpha_1, K)
          P_est_1_DEFAULT = find_P_est(alpha_1, psi_1_DEFAULT, K)

          psi_0_DEFAULT = find_psi(alpha_0, K)
          P_est_0_DEFAULT = find_P_est(alpha_0, psi_0_DEFAULT, K)

          if (check_with_asym(bound, X, psi_1_DEFAULT,P_est_1_DEFAULT, K ) == -1) :
              #failed to compute logL, it is too large for available memory and desired precision
              return alpha_fail
            
          if (check_with_asym(bound, X, psi_0_DEFAULT,P_est_0_DEFAULT, K ) == -1) :
              #failed to compute logL, it is too large for available memory and desired precision
              return alpha_fail  

        
          psioverprobs_0 = np.zeros(K, dtype=np.float64)
          psioverprobs_1 = np.zeros(K, dtype=np.float64)
          for k in range(K):
              psioverprobs_0[k] = psi_0_DEFAULT/P_est_0_DEFAULT[k]
              psioverprobs_1[k] = psi_1_DEFAULT/P_est_1_DEFAULT[k]
              

          # logL computation
          if(scipy_mode == 0):
              start_time_0 = perf_counter()
              log_L_0 = ver_logL_64(N, psi_0_DEFAULT, psioverprobs_0, X, K) 
              end_time_0 = perf_counter()
              DEFAULTexec_time_0 = end_time_0 - start_time_0
              start_time_1 = perf_counter()
              log_L_1 = ver_logL_64(N, psi_1_DEFAULT, psioverprobs_1, X, K)
              end_time_1 = perf_counter()
              DEFAULTexec_time_1 = end_time_1 - start_time_1
          else:
              start_time_0 = perf_counter()
              log_L_0 = ver_logLscipy64(N, psi_0_DEFAULT, psioverprobs_0, X, K) 
              end_time_0 = perf_counter()
              DEFAULTexec_time_0 = end_time_0 - start_time_0
              start_time_1 = perf_counter()
              log_L_1 = ver_logLscipy64(N, psi_1_DEFAULT, psioverprobs_1, X, K)
              end_time_1 = perf_counter()
              DEFAULTexec_time_1 = end_time_1 - start_time_1

          # add up time    
          tot_default_time = tot_default_time + DEFAULTexec_time_0 + DEFAULTexec_time_1
          

          _abs = abs((log_L_0) - (log_L_1))
        
          if(_abs < tolerance):
              end_time = perf_counter()
              duration = tot_default_time
              print("time = ", duration, " seconds, and ", duration/60, " minutes");
              print("iterations ", j);
              return alpha_1
 
          #distance between consecutive alphas not close enough, prep for next iteration
          for k in range(K):
             alpha_0[k] = alpha_1[k]
            
    #raise NotConvergingError(
    print("Failed to converge after {} iterations, values are {}.".format(maxiter, alpha_1));

def run_minka_DEFAULT(_file, _delimiter, prec,tolerance,  maxiter, scipy_mode):

     # input: _file containing the dataset
     # input: _delimiter: delimiter inside _file
     # input: prec: precision of deciaml digits (after decimal dot)
     # input: tolerance: upper bound on diff between two consecutive alphas 
     # input: maxiter: number of iterations for the fixed-point procedure
     # input: scipy_mode: either 0 or 1, see minka_procedure_DEFAULT_mode
   
     #output: estimated alpha using the Python default logL function( if computable, or alpha_fail otherwise)
    D_counts, N = load_dataset_from_file(_file, _delimiter)
    R,K = D_counts.shape

    _alpha = minka_procedure_DEFAULT_mode(prec, tolerance, D_counts, maxiter, R, K, scipy_mode)
    if(sum(_alpha)== -1.0*K): # minka failed
        print("Error in Minka by DEFAULT: alpha is not computable, try a different dataset and/or a smaller precision");
    
    return _alpha   
      
def minka_DEFAULT_experiment_from_file(_file, _delimiter, prec, tolerance, maxiter, scipy_mode): #trump_biden_2020_training.csv
     # executes  run_minka_DEFAULT on using training dataset stored in _file
     # input: prec: precision of deciaml digits (after decimal dot)
     # input: tolerance: upper bound on diff between two consecutive alphas 
     # input: maxiter: number of iterations for the fixed-point procedure
     # input: scipy_mode: either 0 or 1, see minka_procedure_DEFAULT_mode

     
     # returns respective alpha and psi using the Python default logL function( if computable, or alpha_fail and -1 otherwise)

    _alpha = run_minka_DEFAULT(_file, _delimiter, prec, tolerance, maxiter, scipy_mode)
    K = _alpha.shape
    _psi = find_psi(_alpha, K)

    
    if(_psi== -1):
         print("Minka by LM error: invalid psi, terminate the program")
         exit()
    print("psi = ",  _psi);
    return _alpha, _psi

################ end of Minka by DEFAULT #############################################


    
###########################################
#
#   MAIN PROCEDURE
#	runs a series of experiments with several datasets described by their parameters
#	USAGE: call it with python3.11 LogL-global-v5.py &1 &2
#   USAGE: first parameter &1: precision for the mpmath computation
#   USAGE: second parameter &2: requested accuracy for the mantissa in LM
#
###########################################


print('--------------------------------------------------------------------------------')
print('     Author of the Script: Alessandro Languasco; (C) 2022')
print('     developed in August-September 2022 for a joint work with ')
print('     S. Al-Haj Baddar (University of Jordan) and M. Migliardi (Padua University)')
print('--------------------------------------------------------------------------------')
print('    Versions =')
print('python: {}'.format(sys.version))
print('numpy: {}'.format(np.__version__))
print('mpmath: {}'.format(mp.__version__))
print('scipy: {}'.format(scipy.__version__))
print ('---------------------------------------------')	
from datetime import datetime
# datetime object containing current date and time
now = datetime.now()
# dd/mm/YY H:M:S
dt_string = now.strftime("%d/%m/%Y %H:%M:%S")
print("date and time =", dt_string)
print ('---------------------------------------------')	

prec = int(prec)
if prec > 10 or prec < 2:
	sys.exit(' **** ERROR: 2< = prec <= 10')
#endif	
print('**** SETTING: internal precision =', defaultprecision, 'decimal digits')
print('**** SETTING: required accuracy for LM (mantissa) =', prec, 'decimal digits')
print('**************************************************')	
print('**** START experiments')
print('**************************************************')	

######  Experiments  ######

######################################
########## setup ends ################
######################################

#maximum number of iterations until convergence
_maxiter = 1000000

# sample of the dataset for building the model
sample_dataset_file = "test_fix_counts.csv"
# the complete dataset
all_dataset_file = "all_fix_counts.csv" 

# D is the matrix of counts from the sample dataset( number of rows is the number of instances, number of columns is the number of categories)
# N is the total number of counts across all categories (i.e. summation of all values in D)
# X is a vector of counts summations  from the sample dataset(one entry per category)
D, N = load_dataset_from_file(sample_dataset_file, delim=',')
X = find_X(D)
# K is the number of categories
K = len(X)

# D_real is the matrix of counts from the complete dataset
# N_real is the total number of counts across all categories (i.e. summation of all values in D_real)
# X_real is a vector of counts summations  from the complete dataset(one entry per category)
D_real, N_real = load_dataset_from_file(all_dataset_file, delim=',')
X_real = find_X(D_real)

#vector of the actual probability for each category, across the complete dataset
real_probs = find_probs(all_dataset_file, ",")

# tolerance is used to test if convergence has been achieved
tolerance = 10 ** (-prec-2)/(K+1) 

#increasing tolerance for slow datasets to make the experimentation time reasonable
source_file = sys.argv[3]
if( "cane" in source_file or "cavallo" in source_file):
    tolerance = tolerance*1000000
if("auto" in source_file or "mideast" in source_file or "space" in source_file):
    tolerance = tolerance*100000

#goodness-of-fit parameters
conf = 0.05
_factor = 500


scipy_mode = 0

######################################
########## setup ends ################
######################################

##minka by DEFAULT  experiments (math loggamma)
print('Minka by DEFAULT experiments (math loggamma)');
alpha, psi = minka_DEFAULT_experiment_from_file(sample_dataset_file, ",",prec,tolerance, _maxiter,scipy_mode )#minka_DEFAULT_experiment_from_file("test_fix_counts.csv", ",",prec,tolerance, _maxiter )
print('************************************************************************************');
print('************************************************************************************');


##minka by DEFAULT  experiments (scipy loggamma)
print('Minka by DEFAULT experiments (scipy loggamma) ');
scipy_mode = 1
alpha, psi = minka_DEFAULT_experiment_from_file(sample_dataset_file, ",",prec,tolerance, _maxiter, scipy_mode )#minka_DEFAULT_experiment_from_file("test_fix_counts.csv", ",",prec,tolerance, _maxiter )
print('************************************************************************************');
print('************************************************************************************');


##minka by LM  experiments
print('Minka by LM experiments (C-Wrapped version):');
wrapped = 1
alpha, psi = minka_LM_experiment_from_file(sample_dataset_file, ",",prec,tolerance, _maxiter, wrapped )#minka_LM_experiment_from_file("test_fix_counts.csv", ",",prec,tolerance, _maxiter )


#vector of predicted/expected probabilities using the estimated Dirichlet model parameters
expected_probs = alpha*psi



##distance and goodness-of-fit tests results
print("tvd = ",tvd(real_probs, expected_probs));

diff_probs =  real_probs - expected_probs
print("Euclidean =  ", LA.norm(diff_probs));

print("MSE = ", _mean_square_error(real_probs,expected_probs ));

real_counts = real_probs* _factor
expected_counts = expected_probs* _factor
real_check = check_for_5(real_counts)
expected_check = check_for_5(expected_counts)
if real_check == 0 and expected_check == 0:
    chi_square_test(real_counts, expected_counts, K-1, conf)
else:
    print("sample size less than 5 in at least one category, cannot perform chi square test");

diff_mean = np.float64(0)
diff_variance = np.float64(0)
diff_mean = abs(mean_random_variable(real_probs)- mean_random_variable(expected_probs))
diff_variance = abs(variance_random_variable(real_probs)- variance_random_variable(expected_probs))
the_Z_test(mean_random_variable(real_probs), variance_random_variable(real_probs), _factor,
           mean_random_variable(expected_probs), variance_random_variable(expected_probs), _factor )


print("KL distance is ", KL_distance(real_probs, expected_probs), " and the reverse KL-distance is ", KL_distance(expected_probs, real_probs));

print('************************************************************************************');
print('************************************************************************************');
