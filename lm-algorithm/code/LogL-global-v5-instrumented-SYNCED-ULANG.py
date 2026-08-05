###########################################
#   Copyright (C) 2022 Alessandro Languasco
#	developed in August-September 2022 for a joint work with 
# 	S. Al-Haj Baddar (University of Jordan) and M. Migliardi (Padua University)
###########################################
# 	LM: stands for the Languasco-Migliardi paper
# 	YS: stands for the Yu-Shaw paper
# 	UL: stands for Languasco's alternative description of YS_mesh
###########################################
# 	08/14/2022: latest version with the horizontal shift procedure to handle the Euler-Maclaurin formula for LM
# 	08/15/2022: more comments inserted
# 	08/19/2022: inserted more comments and lngamma, digamma, trigamma from scipy
# 	08/26/2022: inserted several #endfor, #endwhile, #endif to improve readability
# 	08/29/2022: corrected a bug in line 1228: for ell in range (1, L+2) -> for ell in range (1, L+1)
#   09/02/2022: lines 1400-1430: changed the Lmax guessing using a different estimate
#   09/14/2022: v5: created three main functions for LM: main_LM_logL; main_LM_digamma; main_LM_trigamma
#				to simplify the use of the LM-method to a final user that does not want to learn HOW it 
#				really works but just wants to see the final values.
#   09/24/2022: v5: inserted the computation of the lower bound for the mesh level L in YS; updated
#			  	references to relevant equations in UL-paper.
###########################################
#
#   MAIN PROCEDURE
#	runs a series of experiments with several datasets described by their parameters
#	USAGE: call it with python3.11 LogL-global-v5.py &1 &2
#   USAGE: first parameter &1: precision for the mpmath computation
#   USAGE: second parameter &2: requested accuracy for the mantissa in LM
#
###########################################
#
#	GOAL:
#   We compute some examples of the log-likelihood logL, digamma, trigamma functions 
#   with an internal check on the (guessed) size of logL to be sure that the logL-result will be
#	reliable using the float-type (C double precision) with the correct number of required mantissa digits
#
############################################
#
# 	Results computed using:
#
# 	mpmath: to verify the other results
#
#	math: standard approach with the float type (it fails for large overdisposed datasets)
#
#	scipy: standard approach with the float type (it fails for large overdisposed datasets)
#
#	YS_mesh: Yu-Shaw mesh approach (it has the delta-threshold problem)
#
#	my_mesh: my improved mesh approach (it is faster and simpler than YS; has the delta-threshold problem)
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

defaultprecision = sys.argv[1] # mpmath accuracy requested
prec = sys.argv[2] # number of correct decimal digits for the mantissa of the whole problem
#EXPNUM = (int)(sys.argv[3])
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

#code from Alessandro and Mauro to connect Python to C-wrapped LM-loggamma
#lib_path = os.environ.get('LM_HOME') + '/' + 'LM_time_lib.so'
#lib_path = "C:\\Users\\shalh\\OneDrive\\Desktop\\ULangPortV1\\ULangPortV1\\" +"LM_time_lib.so"
#lib_path= "C:\\Users\\shalh\OneDrive - University Of Jordan\\HP Desktop at Office\\Papers\\Digamma\\ULangPortVCurrent\\ULangPortV1_office_desktop\\ULangPortV1\\"+  "LM_time_lib.so"
#lib_path="C:\\Users\\shalh\\OneDrive - University Of Jordan\\HP Desktop at Office\\Papers\\Digamma\\ULangPortVCurrent\\ULangPortV1_office_desktop\\ULangPortV1\\"+  "LM_time_lib.so"
#"C:\\Users\\shalh\\OneDrive\\Desktop\\ULangPortV1\\ULangPortV1\\" +"LM_time_lib.so" #"C:\\Users\\Sherenaz\\Desktop\\ULangPortV1\\ULangPortV1\\"+ "LM_time_lib.so"
#os.chdir("C:\\Users\\Sherenaz\\Desktop\\ULangPortV1\\ULangPortV1\\")
lib_path = os.getcwd()+"/LM_time_lib.so"
print('lib_path=', "*"+lib_path+"*")
LM_horshift = ctypes.CDLL(lib_path)#CDLL(lib_path)





# definitions of math functions used
def lngamma(x):  # mpmath multiprecision
    y = mp.loggamma(x)
    return y

def digamma(x): # mpmath multiprecision
    y = mp.digamma(x)
    return y

def trigamma(x): # mpmath multiprecision
    y = mp.psi(1,x)
    return y

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
	# adapted from Sherenaz code
    # read the precomputed Bernoulli numbers; already weighted if needed
    # the output are four arrays of floats (64 bits; C double type)
    # vec_evenbernoulli: for computing the trigamma differences
	# vec_evenbernoullinorm: for computing the logGamma differences 
	# vec_evenbernoullinormdigamma: for computing the digamma differences 	
	# vec_errcoeff: for computing the error in the Euler-Maclaurin formula for logGamma differences 
	# vec_errcoeff_digamma: for computing the error in the Euler-Maclaurin formula for digamma differences 
	# vec_errcoeff_trigamma: for computing the error in the Euler-Maclaurin formula for trigamma differences 	
    
    lines_1 = []
    lines_2 = []
    lines_3 = []
    lines_4 = []
    lines_5 = []
    lines_6 = []
	
    vec_list = 100*[] #(double*)malloc(100*sizeof(double));
    vec_evenbernoullinorm = []*100 #(double*)malloc(100*sizeof(double));
    vec_evenbernoullinormdigamma = []*100 #(double*)malloc(100*sizeof(double));
    vec_errcoeff = []*100 #(double*)malloc(100*sizeof(double));
    vec_errcoeff_digamma = []*100 #(double*)malloc(100*sizeof(double));
    vec_errcoeff_trigamma = []*100 #(double*)malloc(100*sizeof(double));
            
    with open('bernreal-100.txt') as f:
        lines_1 = f.readlines()
    f.close()
    vec_evenbernoulli = np.arange(0, len(lines_1), dtype=float)
    
    with open('bernreal-norm-100.txt') as f:
        lines_2 = f.readlines()
    f.close()
    vec_evenbernoullinorm = np.arange(0, len(lines_2), dtype=float) #np.array(lines_2)

    with open('bernreal-norm-100-digamma.txt') as f:
         lines_3 = f.readlines()
    f.close()
    vec_evenbernoullinormdigamma = np.arange(0, len(lines_3), dtype=float) #np.array(lines_3)

    with open('err_coeff-100.txt') as f:
        lines_4 = f.readlines()
    f.close()
    vec_errcoeff = np.arange(0, len(lines_4), dtype=float) #np.array(lines_4)

    with open('err_coeff-100_digamma.txt') as f:
        lines_5 = f.readlines()
    f.close()
    vec_errcoeff_digamma = np.arange(0, len(lines_4), dtype=float) #np.array(lines_4)

    with open('err_coeff-100_trigamma.txt') as f:
        lines_6 = f.readlines()
    f.close()
    vec_errcoeff_trigamma = np.arange(0, len(lines_4), dtype=float) #np.array(lines_4)

    i = 0
    for line in lines_1:
        vec_evenbernoulli[i] = float(line)
        #print('benr(i) =', vec_evenbernoulli[i]);
        i += 1 
    #endfor
    
    i = 0
    for line in lines_2:
        vec_evenbernoullinorm[i] = float(line)
        i += 1 
    #endfor
    
    i = 0
    for line in  lines_3:
        vec_evenbernoullinormdigamma[i] = float(line) #float(vec_evenbernoullinormdigamma[i])
        i += 1 
    #endfor
    
    i = 0
    for line in lines_4:
        vec_errcoeff[i] = float(line) #float(vec_errcoeff[i])
        i += 1 
    #endfor
    
    i = 0
    for line in lines_5:
        vec_errcoeff_digamma[i] = float(line) #float(vec_errcoeff[i])
        i += 1 
    #endfor
    
    i = 0
    for line in lines_6:
        vec_errcoeff_trigamma[i] = float(line) #float(vec_errcoeff[i])
        i += 1 
    #endfor                
    
    return  vec_evenbernoulli,  vec_evenbernoullinorm,   vec_evenbernoullinormdigamma, vec_errcoeff, vec_errcoeff_digamma, vec_errcoeff_trigamma

# aux function
def wasserstein_distance_function(P, Q):
    #computes the wasserstein_distance metric
    #input: vector of real probabilities P, of size K
    #input: vector of proposed proobabilities Q, of size K
    #output: KL(P||Q)
    
    return wasserstein_distance(P, Q)

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

def the_KS_test(expected_probs,N1, real_probs, N2):
    #computes the KS-2sample test
    #input: probs per first sample
    #input: probs per second sample
    #N1: size of first sample
    #N2: size of second sample
    #returns : just prints the KS test result(if statistic value is higher than the critical value, the two distributions are different)
    #noteLalpha assumed 0.05, its coff. assumed to be 1.36
    '''You reject the null hypothesis that the two samples were drawn
        from the same distribution if the p-value is less than your significance level.
        The significance level of p value is usually set at 0.05.
        https://stats.stackexchange.com/questions/514341/how-to-interpret-ks-statistic-and-p-value-form-scipy-ks-2samp#:~:text=The%20significance%20level%20of%20p%20value%20is%20usually%20set%20at%200.05.
    '''
    statistics, pvalue = ks_2samp(real_probs, expected_probs)#,alternative='two-sided', method='asymp')
    alpha_coff = 1.36 # for alpha is 0.05
    _product = np.float64(0)
    _product = (N1*N2)
    _sum = np.float64(0)
    _sum = N1+N2
    critical_value = alpha_coff*math.sqrt(_sum/_product)
    #print("KS critical = ", critical_value);
    #print("KS statistics = ", statistics);
    print("KS pvalue = ", pvalue);
    '''if statistics > critical_value:
        print("KS-test says they belong to different distributions");
    else:
        print("KS-test says they belong to the same distribution");
    '''
    if pvalue < 0.05:
        print("KS-test says they belong to different distributions");
    else:
        print("KS-test says they belong to the same distribution");
    
    
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
    #print("Z-test value is ", z_test);
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
    IV = np.linalg.inv(V)# should be this IV = np.linalg.inv(V), but use pinv(V) if V, the cov matrix, is singular
    #results =  cdist(array_1,array_2,'mahalanobis', VI=IV) # could be a different way to do it, supports other distances like cosine
    print(distance.mahalanobis(array_1, array_2, IV))
            
def the_cosine_distance(array_1, array_2):
    #calculates the Mahalanobis distance between two vectors
    #input: array_1: first vector
    #input: array_2: second vector
    #output: prints the Mahalanobis distance between array_1 and array_2 according to distance.mahalanobis function
    
    #V = np.cov(np.array([array_1, array_2]).T)
    #IV = np.linalg.inv(V)# should be this IV = np.linalg.inv(V), but use pinv(V) if V, the cov matrix, is singular
    #results =  cdist(array_1,array_2,'mahalanobis', VI=IV) # could be a different way to do it, supports other distances like cosine
    print(distance.cdist(array_1, array_2, 'cosine'));
    
def the_manhattan_distance(array_1, array_2):
    #calculates the Mahalanobis distance between two vectors
    #input: array_1: first vector
    #input: array_2: second vector
    #output: prints the Mahalanobis distance between array_1 and array_2 according to distance.mahalanobis function
    
    #V = np.cov(np.array([array_1, array_2]).T)
    #IV = np.linalg.inv(V)# should be this IV = np.linalg.inv(V), but use pinv(V) if V, the cov matrix, is singular
    #results =  cdist(array_1,array_2,'manhattan') # could be a different way to do it, supports other distances like cosine
    #print(results);
    print(distance.cityblock(array_1, array_2))
    #print(distance.cdist(array_1, array_2, 'manhattan'));            
               
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
        #print("k is ", k, " current diff is ", _diff, " squared diff is ", );
        _sum_prob = _sum_prob + squared_diff

    #print("_sum_prob from MSE is ", _sum_prob, " and K is ", K);
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
    K = np.shape(D_vector)[0]#len(D_vector)
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
'''
            for col in range(1, K):
                if(D_values[row][col] > D_values[row][max_index]):
                    max_index = col
                if(D_values[row][col] < D_values[row][min_index]):
                    min_index = col
        #end for col
                    
            if(D_values[row][min_index] < 0):
                change = -1*D_values[row][min_index] + 1
                D_values[row][max_index] =  D_values[row][max_index] - change
                D_values[row][min_index] = 1
'''
            
    #return D_values

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
    #print(D_probs);
    return D_probs

def probs_to_counts(probs_vector):
    # generate count vector from probabilities vector, of K values
    # assume values in probs_vector to be of 2 decimal places
    # input: probabilities vector, of K values, its values should add up to 1
    # returns: corresponding count vector, with cardinality of 100
    factor = 10000
    K = len(probs_vector) #np.shape(probs_vector)
    #print("KKK = ", K);
    counts_vector = np.zeros(K, np.float64)
    _sum = int(0)
    '''
    for k in range(K-1):
        counts_vector[k] = math.floor(probs_vector[k]*factor)
        _sum = _sum + counts_vector[k]
    counts_vector[K-1] = factor - _sum
    if counts_vector[K-1] < 0 :
        counts_vector[K-1] = 0
    '''
    for k in range(K):
        counts_vector[k] = math.floor(probs_vector[k]*factor)
        #_sum = _sum + counts_vector[k]
    #counts_vector[K-1] = factor - _sum
    #if counts_vector[K-1] < 0 :
    #    counts_vector[K-1] = 0
    return counts_vector

def find_ni(D_counts, i):
    # adapted from Sherenaz code
    # input: D_counts: a matrix of integer counts, with R rows, and K cols
    # input: i: a row in D_counts 
    # output: the sum of row i in D_counts
    n_i= int(0)
    N, K = D_counts.shape
    for k in range(K):
        n_i = n_i + D_counts[i][k]
    return n_i

'''
def find_N(X): 
    # adapted from Sherenaz code
    # input: an array of integers
    # output: the sum of its components

    #types
    N = int(0)
    
    for x in X:
        N = N + x
        #endfor
    return N
'''
def find_X(D):# factor=1):
    # adapted from Sherenaz code
    # input: a matrix of integer counts, with R rows, and K columns
    # output: a vector summary of the input matrix, has K components,  each is summation of a column
    
    factor = 1
    #print("in find_X, input is ");
    #print(np.matrix(D));
    #condenses matrix D of N instances (rows) across K classes (columns) to a vector of K classes
    rows = np.size(D, 0)
    cols = np.size(D, 1)
    X = np.zeros(cols, dtype=np.float64) #[0]* cols #np.zeros(cols)#[0]* cols
    for c in range(cols):
        _sum = int(0)
        for r in range(rows):
            _sum  = _sum + D[r][c]
        X[c] = (int)(_sum * factor)
    #print("in find_X, output is ");
    #print(X);
    return X
def load_dataset_from_file(_file, delim=','):
    # input: _file: the file name
    # input: _delimiter inside file, defualt file format is csv

    # output: two-dimensional matrix of data corresponding to the entries in the file, and N: total observations (summations of all values in matrix)
    #D_values =np.genfromtxt('biden-trump-training.csv' , delim) #genfromtxt
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

    #print("after fix counts");
    #print(np.matrix(D_values));

    #print("Dataset with Rows = ", R, " cols = ", K, " uploaded successfully");
    X = find_X(D_values)
    N = find_N(X)
    return D_values, N
def find_N(X): 
    # adapted from Sherenaz code
    # input: an array of integers
    # output: the sum of its components

    #types
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
    #
    #input: real_counts_vector: 
    #input: expected_counts_vector:

    #null hypothesis: real and expected frequencies belong to the same distribution
    #alternative hyppothesis: real and expected frequencies DO NOT belong to the same distribution

    #steps: 1. get calcuated chisquare value using real_counts_vector,  and expected_counts_vector
    #       2. get critical value from table using significance level and dof
    #       3. if calculated > critical, reject null, otherwise accept it.  

    critical = chi2.ppf(conf, dof)
    #print('real_counts_vector = ',real_counts_vector, ' and expected counts vector = ', expected_counts_vector);
    _chivalue, _pvalue = chisquare(real_counts_vector, expected_counts_vector)#, dof)
    #print('Chi test: critical value from table is ', critical, " while chi value is ", _chivalue, ' and pvalue is ',_pvalue, ' while conf is ', conf);
    if _pvalue < conf: #_chivalue > critical:
        #print("Chi-square test comparing ", real_counts_vector, " and ", expected_counts_vector,
        print("Chi-test: Reject, Not same dist.");
    else:
        #print("Chi-square test comparing ", real_counts_vector, " and ", expected_counts_vector);
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
###### logL/digamma/trigamma functions withthe Euler-Maclaurin formula; see LM-paper and UL-paper
##############################################################################
# main logL-LM function  
def main_LM_logL(probs, psi, K, X, vec_evenbernoullinorm, vec_errcoeff, LM_accuracy):
	#initialization
	res_LM = np.float64(0) 
	# calls the error/hor_shift eval function
	[LMm, hor_shift] = opt_err_LM_logL(probs, psi, K, X, vec_errcoeff, LM_accuracy)
	#print('LM optimal m for the requested mantissa accuracy =', LMm)	
	#print('LM horizontal shift for the requested mantissa accuracy =', hor_shift)	
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
		#print (" y = 1 ")
		return ( -log(x) )
	#endif
	if  y == 2 :
		#print (" y = 2 ")	
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
		#retval = vec_errcoeff[m-1] * x ** (2*m+1) * (1 - d ** (2*m+1))
		expo = 2*m+1;
		retval = vec_errcoeff[m-1] * (x ** expo) * (1 - (d ** expo) )
		#print("vecerrcoeff= ", vec_errcoeff[m-1])
		#print("x = ", x)
		#print("y = ", y)
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
	#print(retval)	
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
		#print("j = ", j)
		m = 1 
		#print("m = ", m)
		toterr = err(probs, psi, K, X, N, m , vec_errcoeff, j)
		#print("minimal error logL = ", (K+1.0)* (toterr))
		m += 1
		toterr1 = err(probs, psi, K, X, N, m, vec_errcoeff, j) 

		#print(toterr)
		#print(toterr1)
	
		while toterr1 < toterr and toterr > LM_accuracy and m < 100:
			# we have 100 precomputed coefficients; so m +1 < 100
			toterr = toterr1
			#print("m = ", m)
			#print("minimal error logL = ", (K+1.0)* (toterr))
			m += 1  
			toterr1 = err(probs, psi, K, X, N, m, vec_errcoeff, j) 
			#print(toterr)
			#print(toterr1)
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
	#print('minimal error =', min_err)
	#maxprec = floor(abs(log10(min_err)))-1
	#print('The Euler-Maclaurin can be used to compute logL with at most ', maxprec, ' decimal digits')
	#print("m_opt = ", m_opt)			
	#print("hor_shift = ", hor_shift)

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
	
# LM function for digamma 			 
def LM_digamma(x, y, m, vec_evenbernoullinormdigamma, hor_shift):
	# computes the Euler-Maclaurin formula for the digamma differences
	# in the interval [1/x, 1/x + y]
	# with a repeated products strategy	
	# m: sum-index for the Euler-Maclaurin sum
	# vec_evenbernoullinormdigamma: array of precomputed and normed values for the EM-sum of digamma
	# hor_shift: value of the horizontal shift (integer)	
	# output: retval: value of the paired digamma differences in [1/x, 1/x + y]; horizontal shift contribution included
	
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
		return x
	#endif
	if  y == 2 :
		return ( x * (x+2)/(x+1) ) 
	#endif
		
	# handling the horizontal shift contribution; see the UL-paper, eq. (30)
	hor_shift_contrib = 0
	if hor_shift != 0 :
		# computing the horizontal shift contribution
		for j in range (0, hor_shift):
			hor_shift_contrib += 1/( 1 + j * x)
		#endfor
		hor_shift_contrib = hor_shift_contrib * x
		# performing the horizontal shift	
		y = y - hor_shift	
		x = x/(1 + hor_shift * x)
	#endif	
    
	
	# computing the Euler-Maclaurin formula; see LM-paper, Theorem 2
	d = 1/( 1 + x * (y-1) )
	# building the repeated product strategy
	stepx = x * x
	stepd = d * d
	startx = stepx
	startd = stepd
	fattorex = startx
	fattored = startd

	# main term
	retval = - log(d) + (0.5) * x * (1+d)	
	
	# first term in the Euler-Maclaurin formula
	retval = retval + vec_evenbernoullinormdigamma[0] * fattorex * ( 1 - fattored )
	
	for i in range (1, m):       
		fattorex = fattorex * stepx
		fattored = fattored * stepd
		# next term in the Euler-Maclaurin formula
		retval = retval + vec_evenbernoullinormdigamma[i] * fattorex * ( 1 - fattored )
	#endfor

    # adding the horizontal shift contribution
	retval = retval + hor_shift_contrib     		
	return retval

# digamma function of a dataset
def LM_digamma_dataset(probs, psi, K, X, m, vec_evenbernoullinormdigamma, hor_shift):
	# computes (logL)-first derivative on a dataset having:
	# probs: array of the probabilities for each category
	# psi: overdisposition parameter
	# K: number of categories 
	# X: counts of each category 
	# m: sum-index for the Euler-Maclaurin sum
	# vec_evenbernoullinormdigamma: array of precomputed and normed values for the EM-sum for digamma
	# hor_shift: horizontal shift value
	# output: retval: value of the paired digamma differences of the whole dataset; horizontal shift contribution included	

	# types and initializations
	N = int(0)
	retval = np.float64(0)	
	
	N = find_N(X)

	psioverprobs = np.zeros(K, dtype =float)
	for k in range(K):
		psioverprobs[k] = psi/probs[k]
	#endfor
		
	# first interval has a minus sign		
	retval = - LM_digamma(psi, N, m, vec_evenbernoullinormdigamma, hor_shift)    

	for i in range(K):
		retval = retval + LM_digamma(psioverprobs[i], X[i], m, vec_evenbernoullinormdigamma, hor_shift)
	#endfor
	
	return retval

# error functions for digamma
def LMg_digamma(x, y, m, vec_errcoeff_digamma) :
	# computes the error function for the EM formula for digamma differences
	# in the interval [1/x, 1/x + y]
	# m: sum-index for the Euler-Maclaurin sum
	# vec_errcoeff: array of precomputed and normed values for the error in the EM-sum for logL	
	# output: retval: value of the error estimate in [1/x, 1/x + y]
		
	# types and initializations
	retval = np.float64(0)		

	# error formula for digamma; see LM-paper, eq. (15)
	if y>0 and y<3: 
		return(0)
	else:	
		d = 1/(1 + x * (y-1) )
		# coeff m of the vec_errcoeff_digamma is stored in position m-1
		expo = 2*m+2; 
		retval = vec_errcoeff_digamma[m-1] * (x ** expo) * (1 - (d ** expo) )
		retval = abs(retval)
		return retval
	#endif	   

# computes the total error estimates for LM-digamma-dataset
def err_digamma(probs, psi, K, X, N, m, vec_errcoeff_digamma, j):
	# computes the total error for the EM formula for digamma differences
	# on a dataset having:
	# probs: array of the probabilities for each category
	# psi: overdisposition parameter
	# K: number of categories 
	# X: counts of each category 
	# N: sum of the counts in X
	# m: sum-index for the Euler-Maclaurin sum
	# vec_errcoeff_digamma: array of precomputed and normed values for the error in the EM-sum for digamma
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
	
	retval = LMg_digamma(aux1, aux2, m, vec_errcoeff_digamma) 
	
	for i in range(K):
		# for the horizontal shift
		aux1 =  psioverprobs[i]/( 1 + j * psioverprobs[i])		
		aux2 =  X[i]-j		
		retval = retval + LMg_digamma(aux1, aux2, m, vec_errcoeff_digamma)
	#endfor
		
	#print(retval)	
	return retval

def opt_err_LM_digamma(probs, psi, K, X, vec_errcoeff_digamma, LM_accuracy):	
	# it computes the optimal m and the horizontal shift to get results with at least LM_accuracy mantissa digits
	# probs: array of the probabilities for each category
	# psi: overdisposition parameter
	# K: number of categories 
	# X: counts of each category 
	# vec_errcoeff_digamma: array of precomputed and normed values for the error in the EM-sum for digamma
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

	while ok == 0 and j < max_hor_shift :
	
		toterr = err_digamma(probs, psi, K, X, N, 1, vec_errcoeff_digamma, j)
		toterr1 = err_digamma(probs, psi, K, X, N, 2, vec_errcoeff_digamma, j) 
		m = 2
		#print(toterr)
		#print(toterr1)
	
		while toterr1 < toterr and toterr > LM_accuracy and m < 100:
			# we have 100 precomputed coefficients; so m +1 < 100
			toterr = toterr1
			m += 1  
			toterr1 = err_digamma(probs, psi, K, X, N, m, vec_errcoeff_digamma, j) 
			#print(toterr)
			#print(toterr1)
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

	#epsilon  = math.ldexp(1.0, -53)
	#min_err = max(abs((K+1)*toterr), epsilon )
	#print('minimal error =', min_err)
	#maxprec = floor(abs(log10(min_err)))-1
	#print('The Euler-Maclaurin can be used to compute logL with at most ', maxprec, ' decimal digits')
	#print("m_opt = ", m_opt)			
	#print("hor_shift = ", hor_shift)

	return [m_opt, hor_shift]


# main trigamma-LM function  
def main_LM_trigamma(probs, psi, K, X, vec_evenbernoulli, vec_errcoeff_trigamma, LM_accuracy):
	#initialization
	restrigamma_LM = np.float64(0)  
	
	# calls the error/hor_shift eval function
	[LMm_trigamma, hor_shift_trigamma] = opt_err_LM_trigamma(probs, psi, K, X, vec_errcoeff_trigamma, LM_accuracy)
	print('LM_trigamma optimal m for the requested mantissa accuracy =', LMm_trigamma)	
	print('LM_trigamma horizontal shift for the requested mantissa accuracy =', hor_shift_trigamma)		
	if LMm_trigamma == 0 :
		restrigamma_LM = 0	# error code for the Euler-Maclaurin formula not accurate enough
	else:
		# calls the trigamma-eval function that uses the LM-technique
		restrigamma_LM = LM_trigamma_dataset(probs, psi, K, X, LMm_trigamma, vec_evenbernoulli, hor_shift_trigamma)
	#endif
	return restrigamma_LM
	 
    
# LM-function for trigamma  
def LM_trigamma (x, y, m, vec_evenbernoulli, hor_shift):
	# computes the Euler-Maclaurin formula for the trigamma differences
	# in the interval [1/x, 1/x + y]
	# with a repeated products strategy
	# m: sum-index for the Euler-Maclaurin sum
	# vec_evenbernoulli: array of precomputed values for the EM-sum of trigamma	
	# hor_shift: horizontal shift value
	# output: retval: value of the paired trigamm differences in [1/x, 1/x + y]; horizontal shift contribution included	
	
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
		return ( - (x**2) )
	#endif
	if  y == 2 :
		return ( - x**2 * ( 1/((1+x)**2) + 1) )
	#endif
	
	# handling the horizontal shift contribution; see the UL-paper, eq. (35) with ell = 1
	hor_shift_contrib = 0
	if hor_shift != 0 :
		# computing the horizontal shift contribution		
		for j in range (0, hor_shift):
			hor_shift_contrib += 1/( (1 + j * x)**2 )
		#endfor
		hor_shift_contrib = hor_shift_contrib * (x**2)
		# preforming the shift
		y = y - hor_shift	
		x = x/(1 + hor_shift * x)
	#endif	
	
	hor_shift_contrib = - hor_shift_contrib  # it has a minus sign for the trigamma function
	
	# computing the Euler-Maclaurin formula; see LM-paper, Theorem 3, ell = 1
	d = 1/( 1 + x * (y-1) )
	# building the repeated product strategy	
	stepx = x * x
	stepd = d * d
	startx = stepx * x
	startd = stepd * d
	fattorex = startx
	fattored = startd
	
	# main term (sign change at the end)
	retval = x * ( (1-d) + 0.5 * x * (1 + stepd ) ) 
    
    # first term in the Euler-Maclaurin formula
	retval += vec_evenbernoulli[0] * fattorex * ( 1 - fattored )

	for i in range (1, m):       
		fattorex = fattorex * stepx
		fattored = fattored * stepd
		# next term in the Euler-Maclaurin formula
		retval += vec_evenbernoulli[i] * fattorex * ( 1 - fattored )
	#endfor

	# The Euler-Maclaurin formula has a minus sign for the trigamma function
	retval = -retval
	
    # adding the horizontal shift contribution	
	retval = retval + hor_shift_contrib                
	return retval
	    
# trigamma function of a dataset
def LM_trigamma_dataset(probs, psi, K, X, m, vec_evenbernoulli, hor_shift):
	# computes (logL)-second derivative on a dataset having:
	# probs: array of the probabilities for each category
	# psi: overdisposition parameter
	# K: number of categories 
	# X: counts of each category 
	# m: sum-index for the Euler-Maclaurin sum
	# vec_evenbernoulli: array of precomputed values for the EM-sum for trigamma
	# hor_shift: value of the horizontal shift
	# output: retval: value of the paired digamma differences for the whole dataset; horizontal shift contribution included	

	# types and initializations
	N = int(0)
	retval = np.float64(0)	

	N = find_N(X)
	    
	psioverprobs = np.zeros(K, dtype =float)
	for k in range(K):
		psioverprobs[k] = psi/probs[k]
	#endfor

	# first interval has a minus sign
	retval = - LM_trigamma(psi, N, m, vec_evenbernoulli, hor_shift)    
	
	for i in range(K):
		retval = retval + LM_trigamma(psioverprobs[i], X[i], m, vec_evenbernoulli, hor_shift)
	#endfor        	
	
	return retval

# error functions for trigamma
def LMg_trigamma(x, y, m, vec_errcoeff_trigamma) :
	# computes the error function for the EM formula for trigamma differences
	# in the interval [1/x, 1/x + y]
	# m: sum-index for the Euler-Maclaurin sum
	# vec_errcoeff: array of precomputed and normed values for the error in the EM-sum for logL	
	# output: retval: value of the error estimate in [1/x, 1/x + y]
		
	# types and initializations
	retval = np.float64(0)		
	d = np.float64(0)			
	
	# error formula for trigamma; see LM-paper, eq. (16), ell = 1
	if y>0 and y<3: 
		return(0)
	else:	
		d = 1/( 1 + x * (y-1) )
		# coeff m of the vec_errcoeff_digamma is stored in position m-1
		expo = 2*m+3;  		
		retval = vec_errcoeff_trigamma[m-1] * (x ** expo) * (1 - (d ** expo) )
		retval = abs(retval)
		return retval
	#endif	   

# computes the total error estimates for LM-digamma-dataset
def err_trigamma(probs, psi, K, X, N, m, vec_errcoeff_trigamma, j):
	# computes the total error for the EM formula for trigamma differences
	# on a dataset having:
	# probs: array of the probabilities for each category
	# psi: overdisposition parameter
	# K: number of categories 
	# X: counts of each category 
	# N: sum of the counts in X
	# m: sum-index for the Euler-Maclaurin sum
	# vec_errcoeff_trigamma: array of precomputed and normed values for the error in the EM-sum for trigamma
	# j: to insert the horizontal shift	
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
	
	retval = LMg_trigamma(aux1, aux2, m, vec_errcoeff_trigamma) 
	
	for i in range(K):
		# for the horizontal shift
		aux1 =  psioverprobs[i]/( 1 + j * psioverprobs[i])	
		aux2 =  X[i]-j		
		retval = retval + LMg_trigamma(aux1, aux2, m, vec_errcoeff_trigamma)
	#endfor
			
	#print(retval)	
	return retval

def opt_err_LM_trigamma(probs, psi, K, X, vec_errcoeff_trigamma, LM_accuracy):	
	# it computes the optimal m and the horizontal shift to get results with at least LM_accuracy mantissa digits
	# probs: array of the probabilities for each category
	# psi: overdisposition parameter
	# K: number of categories 
	# X: counts of each category 
	# vec_errcoeff_trigamma: array of precomputed and normed values for the error in the EM-sum for trigamma
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

	while ok == 0 and j < max_hor_shift :	

		toterr = err_trigamma(probs, psi, K, X, N, 1, vec_errcoeff_trigamma, j)
		toterr1 = err_trigamma(probs, psi, K, X, N, 2, vec_errcoeff_trigamma, j) 
		m = 2
		#print(toterr)
		#print(toterr1)
	
		while toterr1 < toterr and toterr > LM_accuracy and m < 100:
			# we have 100 precomputed coefficients; so m +1 < 100
			toterr = toterr1
			m += 1  
			toterr1 = err_trigamma(probs, psi, K, X, N, m, vec_errcoeff_trigamma, j) 
			#print(toterr)
			#print(toterr1)
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
	#epsilon  = math.ldexp(1.0, -53)
	#min_err = max(abs((K+1)*toterr), epsilon )
	#print('minimal error =', min_err)
	#maxprec = floor(abs(log10(min_err)))-1
	#print('The Euler-Maclaurin can be used to compute logL with at most ', maxprec, ' decimal digits')
	#print("m_opt = ", m_opt)			
	#print("hor_shift = ", hor_shift)

	return [m_opt, hor_shift]

###############################################################
####  ----------  END LM functions implementation   
############################################################### 

###############################################################
####  ----------  BEGIN Yu-Shaw mesh implementation
###############################################################
    
# IMPLEMENTATION of BERNOULLI POLYNOMIALS using functions; 
# modified from Sherenaz code
# required only for the YuShaw-mesh and my_mesh; it seems to be the faster implementation
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
    
'''
# ALTERNATIVE IMPLEMENTATION of BERNOULLI POLYNOMIALS to use polyval; 
# it seems to be slower that the previous functional one

# coefficients of the old-Bernoulli polynomials
phi_1 = np.array([0,1])
phi_2 = np.array([0,-1,1])
phi_3 = np.array([0,1/2,-3/2,1])
phi_4 = np.array([0,0,1,-2,1])
phi_5 = np.array([0,-1/6,0,5/3,-5/2,1])
phi_6 = np.array([0,0,-1/2,0,5/2,-3,1])
phi_7 = np.array([0,1/6,0,-7/6,0,7/2,-7/2,1])
phi_8 = np.array([0,0,2/3,0,-7/3,0,14/3,-4,1])
phi_9 = np.array([0,-3/10,0,2,0,-21/5,0,6,-9/2,1])
phi_10 = np.array([0,0,-3/2,0,5,0,-7,0,15/2,-5,1])
phi_11 = np.array([0,5/6,0,-11/2,0,11,0,-11,0,55/6,-11/2,1])
phi_12 = np.array([0,0,5,0,-33/2,0,22,0,-33/2,0,11,-6,1])
phi_13 = np.array([0,-691/210,0,65/3,0,-429/10,0,286/7,0,-143/6,0,13,-13/2,1])
phi_14 = np.array([0,0,-691/30,0,455/6,0,-1001/10,0,143/2,0,-1001/30,0,91/6,-7,1])
phi_15 = np.array([0,35/2,0,-691/6,0,455/2,0,-429/2,0,715/6,0,-91/2,0,35/2,-15/2,1])
phi_16 = np.array([0,0,140,0,-1382/3,0,1820/3,0,-429,0,572/3,0,-182/3,0,20,-8,1])
phi_17 = np.array([0,-3617/30,0,2380/3,0,-23494/15,0,4420/3,0,-2431/3,0,884/3,0,-238/3,0,68/3,-17/2,1])
phi_18 = np.array([0,0,-10851/10,0,3570,0,-23494/5,0,3315,0,-7293/5,0,442,0,-102,0,51/2,-9,1])
phi_19 = np.array([0,43867/42,0,-68723/10,0,13566,0,-446386/35,0,20995/3,0,-12597/5,0,646,0,-646/5,0,57/2,-19/2,1])
phi_20 = np.array([0,0,219335/21,0,-68723/2,0,45220,0,-223193/7,0,41990/3,0,-4199,0,6460/7,0,-323/2,0,95/3,-10,1])
 
def find_phi(m):
    match m:
        case 1: return phi_1
        case 2: return phi_2
        case 3: return phi_3
        case 4: return phi_4
        case 5: return phi_5
        case 6: return phi_6
        case 7: return phi_7
        case 8: return phi_8
        case 9: return phi_9
        case 10: return phi_10
        case 11: return phi_11
        case 12: return phi_12
        case 13: return phi_13
        case 14: return phi_14
        case 15: return phi_15
        case 16: return phi_16
        case 17: return phi_17
        case 18: return phi_18
        case 19: return phi_19
        case 20: return phi_20
    return -1
'''


##  YS: D_m(x,y) function
def YS_sum(x, ypow, coeffs, YSm): 
	# computes D_m(x,y) 
	# input: x, y: to work in the [ 1/x , 1/x + y ] interval
	# input: ypow: array of precomputed powers of y from 1 to YSm
	# input: coeffs: precomputed coefficients of the YS-sum Dm 
	# input: YSm: number of summands in the YS-sum Dm
	# returns the value totale: is the D_m(x,y) value 	
	
	# types and initializations
	totale = np.float64(0)
	fattore = np.float64(0)	
	xpow = np.float64(0)	
	mplusone = int(YSm+1)	

	# y is stored in ypow[1] = y^1
	y = ypow[1]
	
	# trivial case y == 0
	if y == 0:
		return (0)
	#endif		
	# trivial case y == 1
	if y == 1:
		return (0)
	#endif		
	# trivial case y == 2
	if y == 2:
		return (log(1+x))
	#endif	
	
	# initialising for the repeated product strategy over (-x)	
	totale = 0
	fattore = (-1) * x # value step for the repeated product strategy
	xpow = fattore # we start with (-x)^1 with j = 2

	for j in range (2, mplusone):   
		totale += coeffs[j] * xpow * find_phi(ypow,j)  # Bernoulli pols as functions; seems to be the faster option
		#totale = totale + coeffs[j] * xpow * polyval(y,find_phi(j)) #slower
		#totale = totale + coeffs[j] * xpow * (bernoulli(j,y) - bernoulli(j)) #slower
		xpow *= fattore # new power of (-x)
	#endfor
	totale = - totale		# final change of sign
	return totale 



# paired difference of logGammas with YS_mesh description
def Diff_YS_mesh(Xvector, alphavector, L, ypow, coeffs, delta, YSm): 

	# input: Xvector, column of Xmatrix; contains x_k^(j), j =1,.., L, k fixed
	# input: alphavector: column of alphamatrix; contains alpha_k^(j), j =1,.., L, k fixed
	# input: L: mesh level
	# input: coeffs: precomputed coefficients of the YS-sum Dm
	# input: ypow: array to store the needed powers of y-points
	# input: delta: convergence parameter of the YS-sum Dm; for the condition xy <= delta
	# input: YSm: number of summands in the YS-sum Dm
	# returns the value of logL
	# output: retval: is the logL value over the mesh for one category
	
 	# types and initializations
	yiter = np.float64(0)	
	xiter = np.float64(0)	
	retval = np.float64(0)	
	mplusone = int(YSm+1)	

	# points iteration
	for ell in range (1, L+1):
		yiter = Xvector[ell]
		xiter = 1/alphavector[ell-1]
		ypow[1] = yiter
		for j in range (2, mplusone): 	# computing the needed powers of yiter (for computing the Bernoulli pols)
			ypow[j] =  ypow[j-1] * yiter
		#endfor
		# compute the Dm-sum in one subinterval 
		retval += YS_sum(xiter, ypow, coeffs, YSm)  
	#endfor
	
	return retval

def YS_mesh_gen(N, psi, alpha, probs, X, K, delta, Lmax):
    
    # types
	step = np.float64(0)
	totsumy = np.float64(0)		
	L = int(0)	

	start_time = perf_counter()# time.time()

	# mesh matrices initialization 	
	Xmatrix = np.zeros((Lmax, K+1), dtype=int)  # initialization
	pmatrix = np.zeros((Lmax, K+1), dtype=float)  # initialization
	alphamatrix = np.zeros((Lmax, K+1), dtype=float)  # initialization
	psivector = np.zeros(Lmax, dtype=float)  # initialization
	Nvector = np.zeros(Lmax, dtype=int)  # initialization   
		
	# to collect the mesh level for each category
	Lvector = np.zeros(K+1,  dtype=int)  # initialization      
	
	# it will contain  1/psi and alpha_k
	startvector = np.zeros(K+1, dtype=float)  # initialization       
	# it will contain 1/psi+N and alpha_k + x_k
	endvector = np.zeros(K+1, dtype=float)  # initialization       
	
	# first positions initializations
	Xmatrix[0,0] = N
	pmatrix[0,0] = 0
	alphamatrix[0,0] = 1/psi
	psivector[0] = psi
	Nvector[0]= N
	startvector[0] = 1/psivector[0]
	endvector[0] = Xmatrix[0,0] + startvector[0]
	
	# first row initializations Xmatrix, pmatrix, alphamatrix
	# initializating starting and ending points for each category
	for k in range (1, K+1):  
		startvector[k] = alpha[k-1]
		endvector[k] = X[k-1] + startvector[k]
		Xmatrix[0, k] = 0
		pmatrix[0, k] = probs[k-1]
		alphamatrix[0, k] = alpha[k-1]
	#endfor
	
	# building the Xmatrix and alphamatrix; computes totsumy (for the sharp error term estimate)
	totsumy=0 # for the sharp error terms estimate
	for k in range (0, K+1):  		    
		j=0
		while alphamatrix[j, k] < endvector[k]:	# building the mesh points until the last one is greater than the last point
			j += 1 # counter for the mesh points
			Lvector[k]+=1 # for the mesh level
			step = floor(delta * alphamatrix[j-1 , k]) # getting the next subinterval length
			Xmatrix[j, k] = step # new y_{j, K} point 
			alphamatrix[j, k] = alphamatrix[j-1 , k] + Xmatrix[j, k]  # new z_{j, K} point				
			totsumy += Xmatrix[j, k] # for the sharp error terms estimate
		#endwhile	
		
		# last point: adjusting the last point datas
		alphamatrix[j, k] = endvector[k]  # forcing the last point 
		totsumy = totsumy - Xmatrix[j, k]  # removing the last x from totsumy
		Xmatrix[j, k] = round(alphamatrix[j, k] - alphamatrix[j-1 , k])  # adjusting X for the last interval; round is needed to have an integer here
		totsumy += Xmatrix[j, k] #adjusting totsumy for the final point
	#endfor

	# mesh level
	L = Lvector[0]
	for k in range(1, K+1):
		L = max(L, Lvector[k])
	#endfor
	
	# building the Nvector and psivector
	Nvector[0] = 0
	psivector[0] = psi
	for j in range(1, L+1):
		Nvector[j] = 0
		for k in range (1, K+1):
			Nvector[j] += Xmatrix[j, k]
		#endfor
		#psivector[j] = 1/(1/psivector[j-1] + Nvector[j]) # in Yu-Shaw paper
		psivector[j] = psivector[j-1]/(1 + Nvector[j] * psivector[j-1]) # the same as before but regolarized
	#endfor

	# building the pmatrix and inserting the dummy poins for Xmatrix, alphamatrix and pmatrix
	for k in range (1, K+1):
		 Lk = Lvector[k]
		 # building pmatrix 
		 for j in range(1, Lk+1):
		 	pmatrix[j,0] = 0
		 	pmatrix[j,k] = psivector[j] * ( pmatrix[j-1,k]/psivector[j-1] + Xmatrix[j,k] )
		 #endfor
		 # inserting the dummy points	
		 for j in range (Lk+1, L+1):
		 	Xmatrix[j, k] = 0
		 	alphamatrix[j, k] = alphamatrix[Lk, k]
		 	pmatrix[j, k] = pmatrix[Lk, k] 
 	 	 #endfor		
	#endfor 	 	 
	
	
	#print("Memory size of the Xmatrix array: (MB) ", ceil(Xmatrix.nbytes/1024**2))
		
	# to free the unused memory in the matrices definitions
	Xmatrix.resize((L+1, K+1), refcheck=False)
	pmatrix.resize((L+1, K+1), refcheck=False)
	alphamatrix.resize((L+1, K+1), refcheck=False)
	psivector.resize(L+1, refcheck=False)
	Nvector.resize(L+1, refcheck=False)
	#print("Memory size of the Xmatrix array: (MB) ", ceil(Xmatrix.nbytes/1024**2))
	#print(Xmatrix)
	#print(pmatrix)
	#print(alphamatrix)
	#print(psivector)
	#print(Nvector)
	end_time = perf_counter()# time.time() 	 	 	

	#print('YS_mesh_generation exec time (seconds) =', end_time - start_time)
	return [L, totsumy, Xmatrix, pmatrix, alphamatrix, psivector, Nvector]

def estimate_Lmax(alpha, psi, delta, X, N, K):
    param = np.float64(0)
    Lmax = int(0)

    # mesh dimension guessing
    param = log(1 + delta) # aux variable
    oneoverdelta = 1/delta # aux variable

    if (psi == delta):
        Lmax = 1 + ceil( log( N ) / param ) # see eq (11) of UL-paper
    else:
        Lmax = ceil( log(1 + N / (1/psi - oneoverdelta) ) / param ) # see eq (9) of UL-paper
    #endif

    #print("alpha from estimate_Lmax = ", alpha)

    #Sherenaz#
    alpha = fix_alpha(alpha, delta)

    for k in range (0, K):
        if (alpha[k] < oneoverdelta): #delta-threshold check
            print('*********** ERROR: delta-mesh not applicable; try, e.g., to change delta; pay attention at the accuracy, though !!')
            return 0
	#endif
        if (alpha[k] == oneoverdelta):
            Lmax = max( Lmax , 1 + ceil( log( X[k] ) /param ) ) # see eq (11) of UL-paper
        else:
            Lmax = max( Lmax , ceil( log(1 + X[k] / (alpha[k] - oneoverdelta) )/param ) ) # see eq (9) of UL-paper
	#endif
    #endfor
	# upper bound (roughly with a factor of 1.5) for the mesh level L	
    Lmax = ceil(1.5*Lmax)
    return Lmax


## Calls the diffloggamma computation over categories; 
## keep track of the parameters for the error estimates
def ver_logL_YS_mesh(N, psi, psioverprobs, alpha, probs, X, K, delta, YSm):
	# input: N, psi, psioverprobs = 1/alpha, X, K: from the dataset
	# input: ypow: array to store the needed powers of y-points	
	# input: coeffs: precomputed coefficients of the YS-sum Dm
	# input: delta: convergence parameter of the YS-sum Dm; for the condition xy <= delta
	# input: YSm: number of summands in the YS-sum Dm
	# returns a vector with [ logL value, mesh level L, sum of the y steps]	
	# returns [ 0, 0, 0] if the delta threshold does not hold 	
	# output: z: is the logL value over the mesh
	# output: L: mesh level: needed for the weak error estimate
	# output: totsumy: total sum of y_j over the mesh (on all the categories): needed for the sharp error estimate
	
	# types and initializations
	retval = np.float64(0)
	z = np.float64(0)
	correction =  np.float64(0)
	totsumy = np.float64(0)	
	param = np.float64(0)	
	Lmax = int(0)
	Llowerbound = int(0)
	mplusone = int(YSm+1)
	
	#initialization of precomputed coeffs for the YS-Dm sum
	coeffs = np.empty(mplusone, dtype=float)  # initialization

	# precomputed coefficients for the YS-Dm sum	
	for j in range (2, mplusone):  
		coeffs[j] = 1/( j * (j-1) ) # denominators in YS-Dm sum
    #endfor	      
   
    # for storing the needed precomputed powers of the used y-point (for computing the Bernoulli pols)
    # such powers are computed in the Diff_YS_mesh function and passed to the YS_sum function
	ypow = np.empty(mplusone, dtype=float) # initialization

	# mesh dimension guessing 	
	param = log(1 + delta) # aux variable
	oneoverdelta = 1/delta # aux variable
	
	if (psi > delta): #delta-threshold check
		print('*********** ERROR: delta-mesh not applicable; try, e.g., to change delta; pay attention at the accuracy, though !!')
		return [0,0,0]
	#endif	
	if (psi == delta):
		Lmax = 1 + ceil( log( N ) / param ) # see eq (12) of UL-paper
	else:
		Lmax = ceil( log(1 + N / (1/psi - oneoverdelta) ) / param ) # see eq (10) of UL-paper
    #endif

	Llowerbound = ceil( log ( 1 + psi * N ) / param ) # see eq (8) of UL-paper
    
	for k in range (0, K):
		if (alpha[k] < oneoverdelta): #delta-threshold check
			print('*********** ERROR: delta-mesh not applicable; try, e.g., to change delta; pay attention at the accuracy, though !!')
			return [0,0,0]
		#endif
		if (alpha[k] == oneoverdelta): 
			Lmax = max( Lmax , 1 + ceil( log( X[k] ) /param ) ) # see eq (12) of UL-paper
		else:	
			Lmax = max( Lmax , ceil( log(1 + X[k] / (alpha[k] - oneoverdelta) )/param ) ) # see eq (10) of UL-paper
		#endif
		Llowerbound = min( Llowerbound, ceil( log ( 1 +  X[k] / alpha[k] ) / param ) ) # see eq (8) of UL-paper
	#endfor
	
	#print('upper bound for YS_mesh_level =', Lmax)
	#print('lower bound for YS_mesh_level =', Llowerbound)

	# upper bound (roughly with a factor of 1.5) for the mesh level L	
	Lmax = ceil(1.5*Lmax) 

	# calling the mesh generation procedure
	[L, totsumy, Xmatrix, pmatrix, alphamatrix, psivector, Nvector] = YS_mesh_gen(N, psi, alpha, probs, X, K, delta, Lmax)
	
 	# auxiliary vectors to pass to the logL eval function (Diff_YS_mesh)
	alphavector = np.zeros(L+1, dtype=float)  # initialization    
	Xvector = np.zeros(L+1, dtype=int)  # initialization    
	
	# start the evaluation of logL ; see eq. (23) of UL-paper

	# initialise parameters for the first category
	for j in range (0, L+1):
		alphavector[j] = 1/psivector[j]
	#endfor		
	
	# computing the Dsum
	retval = Diff_YS_mesh(Nvector, alphavector, L, ypow, coeffs, delta, YSm)
	z =  - retval # the first interval has a difflogGammas with a minus change
	
	# initialise parameters for the other categories
	for k in range (1, K+1):  
		# initialise parameters for category k
		for j in range (0, L+1):
			Xvector[j] = Xmatrix[j, k]
			alphavector[j] = alphamatrix[j, k]
		#endfor

		# computing the Dsum	
		retval = Diff_YS_mesh(Xvector, alphavector, L, ypow, coeffs, delta, YSm)
		z += retval  # update the total sum
	#endfor		
	 
	# computing the correction factor, see eq. (23) of UL-paper; it is in fact the largest term
	for j in range (1, L+1): 	
		for k in range (1, K+1):  
			correction += Xmatrix[j, k]*log(pmatrix[j-1, k])
		#endfor
	#endfor			
		
	# adjusting the final result
	z += correction

	# return logL (z), L, and totsumy
	return [z, L, totsumy]

###############################################################
####  ----------  END Yu-Shaw mesh implementation	
###############################################################

###############################################################
####  ----------  BEGIN my_mesh implementation
###############################################################


##  YS: -y*log(x)+ D_m(x,y) function
def YS_funct(x, ypow, coeffs, YSm): 
	# computes -ylog(x) + D_m(x,y) 
	# input: x, y: to work in the [ 1/x , 1/x + y ] interval
	# input: ypow: array of precomputed powers of y from 1 to YSm
	# input: coeffs: precomputed coefficients of the YS-sum Dm 
	# input: YSm: number of summands in the YS-sum Dm
	# returns the value totale: is the -ylog(x) + D_m(x,y) value 	
	
	# types and initializations
	totale = np.float64(0)
	fattore = np.float64(0)	
	xpow = np.float64(0)	
	mplusone = int(YSm+1)	
	
	# y is stored in ypow[1] = y^1
	y = ypow[1]
	# trivial case y == 1
	if y == 1:
		#print('case y=1')
		return ((-1) * log(x))
	#endif		
	# trivial case y == 2
	if y == 2:
		#print('case y=2')
		return (log(1 + x) - 2 * log(x))
	#endif
		
	# initialising for the repeated product strategy over (-x)	
	fattore = (-1) * x # value step for the repeated product strategy
	xpow = fattore # we start with (-x)^1 with j = 2

	for j in range (2, mplusone):   
		totale += coeffs[j] * xpow * find_phi(ypow,j)  # Bernoulli pols as functions; seems to be the faster option
		#totale = totale + coeffs[j] * xpow * polyval(y,find_phi(j)) #slower
		#totale = totale + coeffs[j] * xpow * (bernoulli(j,y) - bernoulli(j)) #slower
		xpow *= fattore # new power of (-x)
	#endfor
	totale = - totale		# final change of sign
	totale += - y * log(x)  # adding the main term
	return totale 

# paired difference of logGammas with my_mesh description
def Diff_my_mesh(x, y, ypow, coeffs, delta, YSm): 
	# input: N, psi, psioverprobs = 1/alpha, X, K: from the dataset
	# input: coeffs: precomputed coefficients of the YS-sum Dm
	# input: ypow: array to store the needed powers of y-points
	# input: delta: convergence parameter of the YS-sum Dm; for the condition xy <= delta
	# input: YSm: number of summands in the YS-sum Dm
	# returns a vector with [ logL value, mesh level L, sum of the y steps]
	# returns [ 0, 0, 0] if the delta threshold does not hold  
	# output: retval: is the logL value over the mesh for one category
	# output: counter: mesh level for one category: needed for the weak error estimate
	# output: sumy: sum of y_j for one category: needed for the sharp error estimate	
	
	#print('my_meshing the categories: parameter delta =', delta)
 
 	# types and initializations
	zstart = np.float64(0)
	zend = np.float64(0)
	z0 = np.float64(0)
	z1 = np.float64(0)	
	yiter = np.float64(0)	
	xiter = np.float64(0)	
	xfinal = np.float64(0)	
	retval = np.float64(0)	
	sumy = np.float64(0)	
	counter = int(1)	
	mplusone = int(YSm+1)	
		
	# stating mesh points generation
	# initialization
	zstart = 1/x
	zend = zstart + y
	
	# first pair
	z0 = zstart
	y1 = floor(delta * z0)
	# checking the delta threshold
	if y1 == 0:
		return([0,0,0]) # the delta threshold does not hold
	#endif
	z1 = z0 + y1
		
	# points iteration
	while z1 < zend:
		sumy += y1 # for the sharp error estimate
		yiter = y1
		xiter = 1/z0
		ypow[1] = yiter
		for j in range (2, mplusone): 	# computing the needed powers of yiter (for computing the Bernoulli pols)
			ypow[j] =  ypow[j-1] * yiter
		#endfor
		# compute the difflogGammas with the YS function in one subinterval 
		retval += YS_funct(xiter, ypow, coeffs, YSm)  #retval = retval + lngamma(z1)-lngamma(z0)
		# initialising points for the next iteration
		z0 = z1
		y1 = floor(delta * z0)
		z1 = z0 + y1
		counter += 1  # for the weak error estimate
	#endwhile		
	#print ('my_mesh level counter is =', counter) 
	
	# last subinterval
	y1 = round(zend - z0) # rounded to the nearest integer to get rid of some error in computation
	xfinal = 1/z0	
	ypow[1] = y1
	for j in range (2, mplusone): # computing the needed powers of y1 (for computing the Bernoulli pols)
		ypow[j] =  ypow[j-1] * y1
	#endfor

	# compute the difflogGammas with the YS function in the last subinterval
	retval += YS_funct(xfinal, ypow, coeffs, YSm) 	#retval = retval + lngamma(zend)-lngamma(z0)	
	sumy +=  y1  # for the sharp error estimate

	return [retval , counter, sumy]

## Calls the difflogGamma computation over categories; 
## keep track of the parameters for the error estimates
def ver_logL_my_mesh(N, psi, psioverprobs, X, K, delta, YSm):
	# input: N, psi, psioverprobs = 1/alpha, X, K: from the dataset
	# input: ypow: array to store the needed powers of y-points	
	# input: coeffs: precomputed coefficients of the YS-sum Dm
	# input: delta: convergence parameter of the YS-sum Dm; for the condition xy <= delta
	# input: YSm: number of summands in the YS-sum Dm
	# returns a vector with [ logL value, mesh level L, sum of the y steps]	
	# returns [ 0, 0, 0] if the delta threshold does not hold 	
	# output: z: is the logL value over the mesh
	# output: L: mesh level: needed for the weak error estimate
	# output: totsumy: total sum of y_j over the mesh (on all the categories): needed for the sharp error estimate
	
	# types and initializations
	retval = np.float64(0)
	z = np.float64(0)
	sumy = np.float64(0)
	totsumy = np.float64(0)	
	mplusone = int(YSm+1)
	L = int(0)
	L1 = int(0)	
			
	#initialization of precomputed coeffs for the YS-Dm sum
	coeffs = np.empty(mplusone, dtype=float)  # initialization

	# precomputed coefficients for the YS-Dm sum	
	for j in range (2, mplusone):  
		coeffs[j] = 1/ ( j * (j-1) ) # denominators in YS-Dm sum
    #endfor	      
    	    		
	# for storing the needed precomputed powers of the used y-point (for computing the Bernoulli pols)
	# such powers are computed in the Diff_my_mesh function and passed to the YS_funct function
	ypow = np.empty(mplusone, dtype=float) # initialization
	
	#print ('-----')
	#print ('category number ', 0)
	
	[retval , L1, sumy] = Diff_my_mesh(psi, N, ypow, coeffs, delta, YSm)
	if retval == 0:
		print('*********** ERROR: delta threshold does not hold !!!')
		return [0,0,0]	
	#endif
			
	L = max(L, L1) # initialising: mesh level: for the weak error estimate
	totsumy = sumy # initialising: sum of the y_j: for the sharp error estimate

	z =  - retval # the first interval has a difflogGammas with a minus change
	
	for k in range (0, K):  
		#print ('-----')
		#print ('category number ', k+1)
		[retval, L1, sumy] = Diff_my_mesh(psioverprobs[k], X[k], ypow, coeffs, delta, YSm)
		if retval == 0:
			print('*********** ERROR: delta threshold does not hold !!!')
			return [0,0,0]	
		#endif			
		#print(retval)			
		z += retval  # update the total sum

		L = max(L, L1) # updating: mesh level: for the weak error estimate
		totsumy += sumy # updating: sum of the y_j: for the sharp error estimate
	#endfor	
	
	return [z, L, totsumy]
	
###############################################################
####  ----------  END my_mesh implementation	
###############################################################

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
    #start_t = perf_counter()
    z =  - Diff64(psi,N)
    for k in range (0, K):
        z += Diff64(psioverprobs[k], X[k])
	#endfor
    #t= perf_counter() - start_t
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

##  FURTHER FUNCTIONS: digamma, trigamma
##  they directly use the digamma and trigamma function of mpmath
##  with no mesh of points

#  paired difference of digammas over the [ 1/x , 1/x + y ] interval mpmath (multiprecision)
def Diffdigamma(x, y): 
	z = digamma(1/x+y)-digamma(1/x)
	return z
	
# paired difference of digammas over categories mpmath (multiprecision)
def ver_digamma(N, psi, psioverprobs, X, K):
	z =  - Diffdigamma(psi,N)
	for k in range (0, K):  	 
		z += Diffdigamma(psioverprobs[k],X[k])
	#endfor	
	return z
	
#  paired difference of digammas over the [ 1/x , 1/x + y ] interval scipy (float64)
def Diffscipydigamma64(x, y): 
	z = scipydigamma64(1/x+y)-scipydigamma64(1/x)
	return z
	
# paired difference of digammas over categories scipy (float64)
def ver_scipydigamma64(N, psi, psioverprobs, X, K):
	z =  - Diffscipydigamma64(psi,N)
	for k in range (0, K):  	 
		z += Diffscipydigamma64(psioverprobs[k],X[k])
	#endfor	
	return z	

# paired difference of trigammas over the [ 1/x , 1/x + y ] interval mpmath (multiprecision)
def Difftrigamma(x, y):
	z = trigamma(1/x+y) - trigamma(1/x)
	return z	

# paired difference of trigammas over categories mpmath (multiprecision)
def ver_trigamma(N, psi, psioverprobs, X, K):
	z =  - Difftrigamma(psi,N) 
	for k in range (0, K): 	
		z += Difftrigamma(psioverprobs[k],X[k])
	#endfor	
	return z

#  paired difference of trigammas over the [ 1/x , 1/x + y ] interval scipy (float64)
def Diffscipytrigamma64(x, y): 
	z = scipytrigamma64(1/x+y)-scipytrigamma64(1/x)
	return z
	
# paired difference of trigammas over categories scipy (float64)
def ver_scipytrigamma64(N, psi, psioverprobs, X, K):
	z =  - Diffscipytrigamma64(psi,N)
	for k in range (0, K):  	 
		z += Diffscipytrigamma64(psioverprobs[k],X[k])
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
    #print("D_counts is ", D_counts);
    # this function is from Minka's python code online, it assumes D_counts is a probablity (ratios) not counts matrix
    D_probs = counts_to_probs(D_counts)
    factor = 1
    E = D_probs.mean(axis=0)
    E2 = (D_probs ** 2).mean(axis=0)
    
    result = factor*(((E[0] - E2[0]) / (E2[0] - E[0] ** 2)) * E)

    #print("actual initial alpha is ");
    #print(np.matrix(result));
    #print("oneoverdelta is ", 1.0/delta);
    first_time = 0
    for k in range (K):
        if (result[k] < 1.0/delta):
            if(first_time == 0):
                print('*********** ERROR from _init_a: delta-mesh not applicable ***********')
                first_time = 1
                YS_fails[0] = YS_fails[0]+1
            result[k] = 1/delta        
            #print('*********** ERROR from _init_a: delta-mesh not applicable; aborting *********************');
            #return alpha_fail
            
    #print("initial alpha is ");
    #print(np.matrix(result));
   
    #result = factor*np.ones(K, dtype=np.float64)
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

    #print("actual initial alpha is ");
    #print(np.matrix(result));
    #print("oneoverdelta is ", 1.0/delta);

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
        #sys.exit('*********** ERROR: LogL too large to ensure the desired precision in double; switch to multiprecision')
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
        #sys.exit('*********** ERROR: LogL too large to ensure the desired precision in double; switch to multiprecision')
        #endif
   return 1

def minka_procedure_YS_mode(prec, tolerance, D_counts, maxiter, R, K, YSm, delta, YS_fails):
        #adapted from Alessandro's run_experiment function
        #input: prec is the requested number of decimal digits of accuracy
        #tolerance: upper bound on diff between two consecutive alphas
        #input: D_counts: the matrix of counts, with K columns (i.e. categories), R rows(i.e. instances)
        #input: K : the number of categories
        #input: R : number of rows in D_counts
        #input: maxiter: maximum number of ietrations for Minka's procedure
        #input: YSm: number of terms in the D_m fomula from the YS paper, assumed to be 20, like they did
        #input: delta: error bound from YS (0 < delta < 1), if any value in vector alpha is less than 1/delta, we set it to 1/delta
        # input: YS_fails: one-value vector to count YS delta-mesh errors

        #output: alpha, estimated Dirichlet parameter vector, or a vector of K (-1) values if the procedure fails to converge, or fails to compute logL

   

    #asymp = np.float64(0)
    #asymp1 = np.float64(0)

    start_time = perf_counter()
    #print("FIXED-POINT ITERATIONS TO FIND PSI USING YS STARTED, GOING FOR ", maxiter, " ITERATIONS AT MAX.");
    print("precision is ", prec, " tolerance is ", tolerance, " maxiter is ", maxiter, " R is ", R, " K is ", K, " YSm is ", YSm, " delta is ", delta);
    res = mp.mpf(0) # mpmath variable
    res64 = np.float64(0)
    
    resdigamma = mp.mpf(0) # mpmath variable
    restrigamma = mp.mpf(0) # mpmath variable
    '''
    res_LM = np.float64(0)  
    LM_accuracy = np.float64(0)
    LMm = int(0)
    horshift = int(0)
    resdigamma_LM = np.float64(0)
    LMm_digamma = int(0)
    horshift_digamma = int(0)
    restrigamma_LM = np.float64(0)
    LMm_trigamma = int(0)
    horshift_trigamma = int(0)
    ''' 
    res_YSmesh = np.float64(0)
    YStotsumy = np.float64(0)   
    YSL = int(0)
    YSmesh_accuracy_sharp = np.float64(0)
    YSmymesh_accuracy_weak = np.float64(0)  
    '''        
    res_mymesh = np.float64(0)
    mytotsumy = np.float64(0)
    myL = int(0)
    mymesh_accuracy_sharp = np.float64(0)
    mymesh_accuracy_weak = np.float64(0)    
    '''
    common_err = np.float64(0)
    
    N = np.float64(0)   
    mplusone = int(0)
    SIGDIG  = int(15) # number of significative decimal digits in C-double (float)
    bound  = int(0)

    
 
    # find vector of counts X using D_counts
    X = find_X(D_counts)
    N = find_N(X)
    #N = np.int(0)
    #for k in range(K):
    #    N = N + X[k]
          

    #alpha vectors
    #alpha_0: a K-value vector of initially guessed alpha values, using Minka's original method
    alpha_1 = np.zeros(K, dtype=np.float64)
    alpha_0 = _init_a(D_counts, delta, YS_fails)
    #print("alpha_0 before iterations ");
    #print(np.matrix(alpha_0));
    alpha_fail = np.empty(K, dtype=float)  # initialization
    for k in range(K):
        alpha_fail[k] = -1.0
        
    if(sum(alpha_0) == -1*K):
        print("************** delta-mesh error *********************");
        return alpha_fail
    # checking if the problem can be handled with the C double precision (float)
    #SIGDIG = 15 # number of available digits in float
    # bound = SIGDIG - prec - 2 # 2 is experimentally determined on gcc on INTEL i7
    bound = SIGDIG - prec  #
    #print(bound)
    
    if (check_with_asym1(bound, X, K ) == -1) :
        print('*********** ERROR FROM MINKA by YS PROCEDURE: LogL too large to ensure the desired precision in double; switch to multiprecision')
        #print('*********** ABORTING')
        return alpha_fail
        #sys.exit('*********** ERROR: LogL too large to ensure the desired precision in double; switch to multiprecision')
        #endif

    diff = np.float64(0)
    
 
    
    # Start Minka's procedure (YS mode)
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
                #diff = polygamma(0, x+y) - polygamma(0, y) #digamma(x,y)
                #diff = digamma(x+y) - digamma(y)
                diff = scipydigamma64(x+y)- scipydigamma64(y)
                #diff = -y*np.log(1.0/x) + yu_shaw_D_m(1.0/x, y, delta, m)
                sum_i = sum_i + diff 
                
            sum_i_2 = np.float64(0)
            for i in range(R):
                n_i = find_ni(D_counts, i)
                x = n_i 
                y = sum_a0
                #diff = digamma(x+y) - digamma(y)
                diff = scipydigamma64(x+y)- scipydigamma64(y)
                #diff = polygamma(0, x+y) - polygamma(0, y) #digamma(x,y)
                #diff = -y*np.log(1.0/x) +yu_shaw_D_m(1.0/x, y, delta, m)
                sum_i_2 = sum_i_2 + diff
               
            alpha_1[k] = alpha_0[k] * (sum_i/sum_i_2)
            #if(alpha_1[k] < 1.0/delta):
            #    alpha_1[k] = 1.0/delta

          #current alpha (alpha_1) is ready
          #print("in iteration ", j, " alpha_0 = ");
          #print(np.matrix(alpha_0));#",   alpha_1 = ", alpha_1, " and alpha_0 is ", alpha_0);
          #print("alpha_1 is");
          #print(np.matrix(alpha_1));
          psi_1_YS = find_psi(alpha_1, K)
          P_est_1_YS = find_P_est(alpha_1, psi_1_YS, K)

          psi_0_YS = find_psi(alpha_0, K)
          P_est_0_YS = find_P_est(alpha_0, psi_0_YS, K)

          if (check_with_asym(bound, X, psi_1_YS,P_est_1_YS, K ) == -1) :
              #failed to compute logL, it is too large for available memory and desired precision
              return alpha_fail
            
          if (check_with_asym(bound, X, psi_0_YS,P_est_0_YS, K ) == -1) :
              #failed to compute logL, it is too large for available memory and desired precision
              return alpha_fail  

          mplusone = YSm + 1
          psioverprobs_0 = np.zeros(K, dtype=np.float64)
          psioverprobs_1 = np.zeros(K, dtype=np.float64)
          for k in range(K):
              psioverprobs_0[k] = psi_0_YS/P_est_0_YS[k]
              psioverprobs_1[k] = psi_1_YS/P_est_1_YS[k]
              

         
          # YS_mesh logL computation
          start_time_0 = perf_counter()# time.time()
          #[res_YSmesh_0, YSL_0, YStotsumy_0] = ver_logL_YSmesh(N, psi_0_YS, psioverprobs_0, alpha_0, P_est_0_YS, X, K, delta, YSm)
          Lmax_0 = estimate_Lmax(alpha_0, psi_0_YS, delta, X, N, K) 
          [L_0, totsumy_0, Xmatrix, pmatrix, alphamatrix, psivector, Nvector] = YS_mesh_gen(N, psi_0_YS, alpha_0, P_est_0_YS, X, K, delta, Lmax_0)
          end_time_0 = perf_counter()
          YSmeshexec_time_0 = end_time_0 - start_time_0 
          # working on the error term estimates
          common_err_0 = delta**YSm/(mplusone*YSm) # for the error terms
          YSmesh_accuracy_sharp_0 = common_err_0 * totsumy_0 # sharp error term estimate
          #YSmesh_accuracy_weak_0 = 2/psi_0_YS * common_err_0 * ((1+delta)**YSL_0 - 1) # weak error term estimate

          start_time_1 = perf_counter()# time.time()
          #[res_YSmesh_1, YSL_1, YStotsumy_1] = ver_logL_YSmesh(N, psi_1_YS, psioverprobs_1, alpha_1, P_est_1_YS, X, K, delta, YSm)
          Lmax_1 = estimate_Lmax(alpha_1, psi_1_YS, delta, X, N, K) 
          [L_1, totsumy_1, Xmatrix, pmatrix, alphamatrix, psivector, Nvector] = YS_mesh_gen(N, psi_1_YS, alpha_1, P_est_1_YS, X, K, delta, Lmax_1)
          end_time_1 = perf_counter()
          YSmeshexec_time_1 = end_time_1 - start_time_1 
          # working on the error term estimates
          common_err_1 = delta**YSm/(mplusone*YSm) # for the error terms
          YSmesh_accuracy_sharp_1 = common_err_1 * totsumy_1 # sharp error term estimate
          #print("YSmesh_accuracy_sharp_0 = ", YSmesh_accuracy_sharp_0);
          #print("YSmesh_accuracy_sharp_1 = ", YSmesh_accuracy_sharp_1);
          #YSmesh_accuracy_weak_1 = 2/psi_1_YS * common_err_1 * ((1+delta)**YSL_1 - 1) # weak error term estimate

          #calculate logL_0  from alpha_0 using Yu-Shaw and check if its error is OK
          #log_L_0 = res_YSmesh_0
          
          #calculate logL_1  from alpha_0 using Yu-Shaw and check if its error is OK
          #log_L_1 = res_YSmesh_1
          #print("YS logL successful for L = ",L, " wish alpha_1 of ", a1);

          if(not(YSmesh_accuracy_sharp_0 < tolerance/100 and YSmesh_accuracy_sharp_1 < tolerance/100)):
              print("Error Minka by YS fixedpoint ietration: LogL from YS are not reliable, the sharp error test failed");
              #duration = end_time - start_time
              #print("Minka using YS terminted after ", duration, " seconds");
              #print("Yu-Shaw converged after ", j, "iterations", 'returning alpha_1= ', alpha_1);
              return alpha_fail
              
          [res_YSmesh_0, YSL_0, YStotsumy_0] = ver_logL_YS_mesh(N, psi_0_YS, psioverprobs_0, alpha_0, P_est_0_YS, X, K, delta, YSm)
          [res_YSmesh_1, YSL_1, YStotsumy_1] = ver_logL_YS_mesh(N, psi_1_YS, psioverprobs_1, alpha_1, P_est_1_YS, X, K, delta, YSm)

          log_L_0 = res_YSmesh_0
          log_L_1 = res_YSmesh_1

          _abs = abs((log_L_0) - (log_L_1))
          #print("yu-Shaw fixedpoint iteration is ", j, " log_L_0 is ", log_L_0, " log_L_1 is ", log_L_1, " diff is ",  _abs," tol is ", tolerance);
          #print("yu-Shaw fixedpoint iteration is ", j,  " diff is ",  _abs," tol is ", tolerance);

          if  (_abs < tolerance):
              end_time = perf_counter()
              duration = end_time - start_time
              print("time = ", duration, " seconds, and ", duration/60 , " minutes");
              print("iterations ", j);#, 'returning alpha_1= ', alpha_1);#, "log_L_0 = ", log_L_0, " and log_L_1 = ",log_L_1);
              alpha_1 = fix_alpha(alpha_1, delta)
              return alpha_1

          #distance between consecutive alphas not close enough, prep for next iteration
          for k in range(K):
             alpha_0[k] = alpha_1[k]
            
    #raise NotConvergingError(
    print("Failed to converge after {} iterations, values are {}.".format(maxiter, alpha_1));

def fix_alpha(alpha, delta):
    K = len(alpha)
    for k in range(K):
        if(alpha[k] < 1/delta):
            alpha[k] = 1/delta
    return alpha

def run_minka_YS(_file, _delimiter, prec,tolerance,  maxiter, YSm, delta, YS_fails):

     # input: _file containing the dataset
     # input: _delimiter: delimiter inside _file
     # input: prec: precision of deciaml digits (after decimal dot)
     # input: tolerance: upper bound on diff between two consecutive alphas
     # input: maxiter: number of iterations for the fixed-point procedure
     # input: YSm: number of terms in the D_m formula from YS, set by default to 20, like they did
     # input: delta: the YS error bound, must be between 0 and 1, set to 0.2 by default like the YS paper
     # input: YS_fails: one-value vector to count YS delta-mesh errors

     #output: estimated alpha ( if computable, or alpha_fail otherwise)
    #print("Minka by YS: uploading Dataset from file ", _file);
    D_counts, N = load_dataset_from_file(_file, _delimiter)
    R,K = D_counts.shape

    _alpha = minka_procedure_YS_mode(prec, tolerance, D_counts, maxiter, R, K, YSm, delta, YS_fails)
    if(sum(_alpha)== -1.0*K): # minka failed
        print("Error from minka by YS: alpha is not computable, try a different dataset and/or a smaller precision");
        #return _alpha
    
    #_psi = find_psi(_alpha, K)
    #print("run_minka_YS_experiment completed, estimated psi is ", _psi);
    return _alpha   
      
def minka_YS_experiment_from_file(_file, _delimiter, prec, tolerance, maxiter, YSm, delta, YS_fails): #trump_biden_2020_training.csv
     # executes  run_minka_YS on using training dataset stored in _file
     # input: prec: precision of deciaml digits (after decimal dot)
     # input: tolerance: upper bound on diff between two consecutive alphas
     # input: maxiter: number of iterations for the fixed-point procedure
     # input: YSm: number of terms in the D_m formula from YS, set by default to 20, like they did
     # input: delta: the YS error bound, must be between 0 and 1, set to 0.2 by default like the YS paper
     # input: YS_fails: one-value vector to count YS delta-mesh errors
     
     # returns respective alpha

    _alpha = run_minka_YS(_file, _delimiter, prec, tolerance, maxiter, YSm, delta, YS_fails)
    K = _alpha.shape
    _psi = find_psi(_alpha, K)

    
    if(_psi== -1):
         print("error from minka by YS invalid alpha, terminate the program")
         exit()
    print("psi = ", _psi);
    return _alpha, _psi

################ end of Minka by YS #############################################

def minka_procedure_LM_mode(prec, tolerance, D_counts, maxiter,  R, K):
        #adapted from Alessandro's run_experiment function
        #input: prec is the requested number of decimal digits of accuracy
        #tolerance: upper bound on diff between two consecutive alphas (equivalent to LM_accuracy from UL notes)
        #input: D_counts: the matrix of counts, with K columns (i.e. categories), R rows(i.e. instances)
        #input: K : the number of categories
        #input: R : number of rows in D_counts
        #input: maxiter: maximum number of ietrations for Minka's procedure
        

        #output: alpha, estimated Dirichlet parameter vector, or a vector of K (-1) values if the procedure fails to converge, or fails to compute logL

   

    #asymp = np.float64(0)
    #asymp1 = np.float64(0)

    start_time = perf_counter()
    #print("FIXED-POINT ITERATIONS TO FIND PSI USING LM STARTED, GOING FOR ", maxiter, " ITERATIONS AT MAX.");
    print("precision is ", prec, " tolerance (i.e. LM_accuracy) is ", tolerance, " maxiter is ", maxiter, " R is ", R, " K is ", K);
    res = mp.mpf(0) # mpmath variable
    res64 = np.float64(0)
    
    resdigamma = mp.mpf(0) # mpmath variable
    restrigamma = mp.mpf(0) # mpmath variable
    
    res_LM = np.float64(0)  
    LM_accuracy = tolerance #np.float64(0)
    LMm = int(0)
    horshift = int(0)
    resdigamma_LM = np.float64(0)
    LMm_digamma = int(0)
    horshift_digamma = int(0)
    restrigamma_LM = np.float64(0)
    LMm_trigamma = int(0)
    horshift_trigamma = int(0)
    #initialize the LM vectors
    [vec_evenbernoulli, vec_evenbernoullinorm, vec_evenbernoullinormdigamma, vec_errcoeff, vec_errcoeffdigamma, vec_errcoefftrigamma] = LM_initBern() 
    ''' 
    res_YSmesh = np.float64(0)
    YStotsumy = np.float64(0)   
    YSL = int(0)
    YSmesh_accuracy_sharp = np.float64(0)
    YSmymesh_accuracy_weak = np.float64(0)  
            
    res_mymesh = np.float64(0)
    mytotsumy = np.float64(0)
    myL = int(0)
    mymesh_accuracy_sharp = np.float64(0)
    mymesh_accuracy_weak = np.float64(0)    
    '''
    common_err = np.float64(0)
    
    N = np.float64(0)   
    mplusone = int(0)
    SIGDIG  = int(15) # number of significative decimal digits in C-double (float)
    bound  = int(0)

    
 
    # find vector of counts X using D_counts
    X = find_X(D_counts)
    N = find_N(X)
    #N = np.int(0)
    #for k in range(K):
    #    N = N + X[k]
          

    #alpha vectors
    #alpha_0: a K-value vector of initially guessed alpha values, using Minka's original method
    alpha_1 = np.zeros(K, dtype=np.float64)
    alpha_0 = _init_a_LM(D_counts)
    #print("alpha_0 before iterations ");
    #print(np.matrix(alpha_0));
    alpha_fail = np.empty(K, dtype=float)  # initialization
    for k in range(K):
        alpha_fail[k] = -1.0

    # checking if the problem can be handled with the C double precision (float)
    #SIGDIG = 15 # number of available digits in float
    # bound = SIGDIG - prec - 2 # 2 is experimentally determined on gcc on INTEL i7
    bound = SIGDIG - prec  #
    #print(bound)
    
    if (check_with_asym1(bound, X, K ) == -1) :
        print('*********** ERROR FROM MINKA by LM PROCEDURE: LogL too large to ensure the desired precision in double; switch to multiprecision')
        #print('*********** ABORTING')
        return alpha_fail
        #sys.exit('*********** ERROR: LogL too large to ensure the desired precision in double; switch to multiprecision')
        #endif

    diff = np.float64(0)
    
 
    
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
                #diff = polygamma(0, x+y) - polygamma(0, y) #digamma(x,y)
                #diff = digamma(x+y) - digamma(y)
                diff = scipydigamma64(x+y)- scipydigamma64(y)
                #diff = -y*np.log(1.0/x) + yu_shaw_D_m(1.0/x, y, delta, m)
                sum_i = sum_i + diff 
                
            sum_i_2 = np.float64(0)
            for i in range(R):
                n_i = find_ni(D_counts, i)
                x = n_i 
                y = sum_a0
                #diff = digamma(x+y) - digamma(y)
                diff = scipydigamma64(x+y)- scipydigamma64(y)
                #diff = polygamma(0, x+y) - polygamma(0, y) #digamma(x,y)
                #diff = -y*np.log(1.0/x) +yu_shaw_D_m(1.0/x, y, delta, m)
                sum_i_2 = sum_i_2 + diff
               
            alpha_1[k] = alpha_0[k] * (sum_i/sum_i_2)
            #if(alpha_1[k] < 1.0/delta):
            #    alpha_1[k] = 1.0/delta

          #current alpha (alpha_1) is ready
          #print("in iteration ", j, " alpha_0 = ");
          #print(np.matrix(alpha_0));#",   alpha_1 = ", alpha_1, " and alpha_0 is ", alpha_0);
          #print("alpha_1 is");
          #print(np.matrix(alpha_1));
          psi_1_LM = find_psi(alpha_1, K)
          P_est_1_LM = find_P_est(alpha_1, psi_1_LM, K)

          psi_0_LM = find_psi(alpha_0, K)
          P_est_0_LM = find_P_est(alpha_0, psi_0_LM, K)

          if (check_with_asym(bound, X, psi_1_LM,P_est_1_LM, K ) == -1) :
              #failed to compute logL, it is too large for available memory and desired precision
              return alpha_fail
            
          if (check_with_asym(bound, X, psi_0_LM,P_est_0_LM, K ) == -1) :
              #failed to compute logL, it is too large for available memory and desired precision
              return alpha_fail  

        
          psioverprobs_0 = np.zeros(K, dtype=np.float64)
          psioverprobs_1 = np.zeros(K, dtype=np.float64)
          for k in range(K):
              psioverprobs_0[k] = psi_0_LM/P_est_0_LM[k]
              psioverprobs_1[k] = psi_1_LM/P_est_1_LM[k]
              

         
          # LM logL computation
          start_time_0 = perf_counter()# time.time()
          log_L_0 = main_LM_logL(P_est_0_LM, psi_0_LM, K, X, vec_evenbernoullinorm, vec_errcoeff, LM_accuracy)
          end_time_0 = perf_counter()
          LMexec_time_0 = end_time_0 - start_time_0 


          start_time_1 = perf_counter()# time.time()
          log_L_1 = main_LM_logL(P_est_1_LM, psi_1_LM, K, X, vec_evenbernoullinorm, vec_errcoeff, LM_accuracy)
          end_time_1 = perf_counter()
          LMexec_time_1 = end_time_1 - start_time_1

          #if( j < 100):
          #    print('iteration ', j, ' log_L_0 = ', log_L_0, ',', ' log_L_1 = ', log_L_1 );
          

          _abs = abs((log_L_0) - (log_L_1))

          #if( j%1000 == 0):
          #    print('iteration ', j, ' _abs = ', _abs, ',', ' tolerance = ', tolerance, ' log_L_0 = ', log_L_0);
          #    #, ' log_L_1= ', log_L_1 , ' alpha_0 = ', alpha_0, ' alpha_1 = ', alpha_1 );
          if  (_abs < tolerance):
              end_time = perf_counter()
              duration = end_time - start_time
              print("time = ", duration, " seconds, and ", duration/60, " minutes");
              print("iterations ", j);#'returning alpha_1= ', alpha_1);#, "log_L_0 = ", log_L_0, " and log_L_1 = ",log_L_1);
              return alpha_1

          #distance between consecutive alphas not close enough, prep for next iteration
          for k in range(K):
             alpha_0[k] = alpha_1[k]
            
    #raise NotConvergingError(
    print("Failed to converge after {} iterations, values are {}.".format(maxiter, alpha_1));

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

   

    #asymp = np.float64(0)
    #asymp1 = np.float64(0)

    # finish settings for wrapping
    #double mainExpAll(int PREC, int expNum, int counts [], int categories, double probabilities [], double psi_0)
    LM_horshift.loggamma_LM.restype = ctypes.c_double#ctypes.c_int
    LM_horshift.loggamma_LM.argtypes = [ ctypes.POINTER(ctypes.c_double*K), ctypes.c_double]#, ctypes.POINTER(ctypes.c_double*100), ctypes.POINTER(ctypes.c_double*100)]


    #LM_horshift.loggamma_LM.restype = ctypes.c_double
    #LM_horshift.loggamma_LM.argtypes = []

    LM_horshift.initBern_logL.restype = None
    LM_horshift.initBern_logL.argtypes = []

    LM_horshift.init_params.restype = ctypes.c_long
    LM_horshift.init_params.argtypes = [ctypes.c_int, ctypes.POINTER(ctypes.c_double*K), ctypes.c_int]
    
    print("using the LM C-wrapped version ");
    
  
    
    start_time = perf_counter()
    #print("FIXED-POINT ITERATIONS TO FIND PSI USING LM STARTED, GOING FOR ", maxiter, " ITERATIONS AT MAX.");
    print("precision is ", prec, " tolerance (i.e. LM_accuracy) is ", tolerance, " maxiter is ", maxiter, " R is ", R, " K is ", K);
    #print("Kokokokokoko");
    res = mp.mpf(0) # mpmath variable
    res64 = np.float64(0)
    
    resdigamma = mp.mpf(0) # mpmath variable
    restrigamma = mp.mpf(0) # mpmath variable
    
    res_LM = np.float64(0)  
    LM_accuracy = tolerance #np.float64(0)
    LMm = int(0)
    horshift = int(0)
    resdigamma_LM = np.float64(0)
    LMm_digamma = int(0)
    horshift_digamma = int(0)
    restrigamma_LM = np.float64(0)
    LMm_trigamma = int(0)
    horshift_trigamma = int(0)
    #initialize the LM vectors
    [vec_evenbernoulli, vec_evenbernoullinorm, vec_evenbernoullinormdigamma, vec_errcoeff, vec_errcoeffdigamma, vec_errcoefftrigamma] = LM_initBern() 
    ''' 
    res_YSmesh = np.float64(0)
    YStotsumy = np.float64(0)   
    YSL = int(0)
    YSmesh_accuracy_sharp = np.float64(0)
    YSmymesh_accuracy_weak = np.float64(0)  
            
    res_mymesh = np.float64(0)
    mytotsumy = np.float64(0)
    myL = int(0)
    mymesh_accuracy_sharp = np.float64(0)
    mymesh_accuracy_weak = np.float64(0)    
    '''
    common_err = np.float64(0)
    
    N = np.float64(0)   
    mplusone = int(0)
    SIGDIG  = int(15) # number of significative decimal digits in C-double (float)
    bound  = int(0)

    
 
    # find vector of counts X using D_counts
    X = np.zeros(K, dtype=np.float64)
    X = find_X(D_counts)
    N = find_N(X)
    #N = np.int(0)
    #for k in range(K):
    #    N = N + X[k]


    #initialize Bern. numbers for the C-wrapped loggamma-LM
    LM_horshift.initBern_logL();
    
    
    #initialize params
    X_ctypes =  X.ctypes.data_as(ctypes.POINTER(ctypes.c_double*K))
    #print("XXXX, X = ",X, " K = ",K);
    #print("XXXXX x_ctypes = ", X_ctypes.shape);
    NN = np.float64(0)  
    NN = LM_horshift.init_params(prec, X_ctypes, K)
    
    print('prec = ',prec, ' k = ', K ,' NN = ', NN, ' and N = ', N);
    #arr_somehow = X_ctypes.contents

    #for kkk in range(K):
    #    print(kkk," " ,arr_somehow[kkk]);
    #alpha vectors
    #alpha_0: a K-value vector of initially guessed alpha values, using Minka's original method
    alpha_1 = np.zeros(K, dtype=np.float64)
    alpha_0 = _init_a_LM(D_counts)
    #print("alpha_0 before iterations ");
    #print(np.matrix(alpha_0));
    alpha_fail = np.empty(K, dtype=float)  # initialization
    for k in range(K):
        alpha_fail[k] = -1.0

    # checking if the problem can be handled with the C double precision (float)
    #SIGDIG = 15 # number of available digits in float
    bound = SIGDIG - prec - 2 # 2 is experimentally determined on gcc on INTEL i7
    #bound = SIGDIG - prec  #
    #print(bound)
    
    if (check_with_asym1(bound, X, K ) == -1) :
        print('*********** ERROR FROM MINKA by LM PROCEDURE: LogL too large to ensure the desired precision in double; switch to multiprecision')
        #print('*********** ABORTING')
        return alpha_fail
        #sys.exit('*********** ERROR: LogL too large to ensure the desired precision in double; switch to multiprecision')
        #endif

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
                #diff = polygamma(0, x+y) - polygamma(0, y) #digamma(x,y)
                #diff = digamma(x+y) - digamma(y)
                diff = scipydigamma64(x+y)- scipydigamma64(y)
                #diff = -y*np.log(1.0/x) + yu_shaw_D_m(1.0/x, y, delta, m)
                sum_i = sum_i + diff 
                
            sum_i_2 = np.float64(0)
            for i in range(R):
                n_i = find_ni(D_counts, i)
                x = n_i 
                y = sum_a0
                #diff = digamma(x+y) - digamma(y)
                diff = scipydigamma64(x+y)- scipydigamma64(y)
                #diff = polygamma(0, x+y) - polygamma(0, y) #digamma(x,y)
                #diff = -y*np.log(1.0/x) +yu_shaw_D_m(1.0/x, y, delta, m)
                sum_i_2 = sum_i_2 + diff
               
            alpha_1[k] = alpha_0[k] * (sum_i/sum_i_2)
            #if(alpha_1[k] < 1.0/delta):
            #    alpha_1[k] = 1.0/delta

          #current alpha (alpha_1) is ready
          #print("in iteration ", j, " alpha_0 = ");
          #print(np.matrix(alpha_0));#",   alpha_1 = ", alpha_1, " and alpha_0 is ", alpha_0);
          #print("alpha_1 is");
          #print(np.matrix(alpha_1));
          psi_1_LM = find_psi(alpha_1, K)
          P_est_1_LM = find_P_est(alpha_1, psi_1_LM, K)

          psi_0_LM = find_psi(alpha_0, K)
          P_est_0_LM = find_P_est(alpha_0, psi_0_LM, K)

          '''
          if (check_with_asym(bound, X, psi_1_LM,P_est_1_LM, K ) == -1) :
              #failed to compute logL, it is too large for available memory and desired precision
              return alpha_fail
            
          if (check_with_asym(bound, X, psi_0_LM,P_est_0_LM, K ) == -1) :
              #failed to compute logL, it is too large for available memory and desired precision
              return alpha_fail  
         '''
        
          psioverprobs_0 = np.zeros(K, dtype=np.float64)
          psioverprobs_1 = np.zeros(K, dtype=np.float64)
          for k in range(K):
              psioverprobs_0[k] = psi_0_LM/P_est_0_LM[k]
              psioverprobs_1[k] = psi_1_LM/P_est_1_LM[k]
              


          
          
          #casting to ctypes before calling C-wrapped LM (loggamma_LM after calling pass_params )     #int_array_type = ctypes.c_int * K # defined the ctypes data type#_ctypes = ctypes.cast(X.ctypes.data, ctypes.POINTER(ctypes.c_long_Array_2)) #ctypes.c_long_Array_2

          
          P_est_0_LM_ctypes = P_est_0_LM.ctypes.data_as(ctypes.POINTER(ctypes.c_double*K))
          P_est_1_LM_ctypes = P_est_1_LM.ctypes.data_as(ctypes.POINTER(ctypes.c_double*K)) 
          #vec_evenbernoullinorm_ctypes = vec_evenbernoullinorm.ctypes.data_as(ctypes.POINTER(ctypes.c_double*100))
          #vec_errcoeff_ctypes = vec_errcoeff.ctypes.data_as(ctypes.POINTER(ctypes.c_double*100))
          
          
          # LM logL computation
          start_time_0 = perf_counter()
          log_L_0 = LM_horshift.loggamma_LM( P_est_0_LM_ctypes,psi_0_LM)#, vec_evenbernoullinorm_ctypes , vec_errcoeff_ctypes )
          if(log_L_0 == 99):
              print("loggamma in C crashed, cannot achieve precision");
              sys.exit()
          #start_time_0 = perf_counter()# time.time()
          #log_L_0 = LM_horshift.loggamma_LM()#prec, X_ctypes, K, psioverprobs_0_ctypes,psi_0_LM, vec_evenbernoullinorm_ctypes , vec_errcoeff_ctypes )
          #main_LM_logL(P_est_0_LM, psi_0_LM, K, X, vec_evenbernoullinorm, vec_errcoeff, LM_accuracy)
          end_time_0 = perf_counter()
          LMexec_time_0 = end_time_0 - start_time_0 


          start_time_1 = perf_counter()
          log_L_1 = LM_horshift.loggamma_LM(P_est_1_LM_ctypes,psi_1_LM)
          if(log_L_1 == 99):
              print("loggamma in C crashed, cannot achieve precision");
              sys.exit()
          #start_time_1 = perf_counter()# time.time()
          #log_L_1 = LM_horshift.loggamma_LM()#(prec, X_ctypes, K, psioverprobs_1_ctypes,psi_1_LM, vec_evenbernoullinorm_ctypes , vec_errcoeff_ctypes) #main_LM_logL(P_est_1_LM, psi_1_LM, K, X, vec_evenbernoullinorm,
          #vec_errcoeff, LM_accuracy)
          end_time_1 = perf_counter()
          LMexec_time_1 = end_time_1 - start_time_1

          tot_LM_time = tot_LM_time + LMexec_time_0 + LMexec_time_1

        

          _abs = abs((log_L_0) - (log_L_1))
          #if( j%1000 == 0):
          #    #print('iteration ', j, ' _abs = ', _abs, ',', ' tolerance = ', tolerance );
          #    print('iteration ', j, ' _abs = ', _abs, ',', ' tolerance = ', tolerance, ' log_L_0 = ', log_L_0);
          #    #, ' log_L_1= ', log_L_1 ,
          #    #      ' psi_0_LM = ', psi_0_LM, ' psi_1_LM = ', psi_1_LM , 'P_est_0_LM = ', P_est_0_LM, ' P_est_1_LM = ', P_est_1_LM, ' X = ', X, ' N = ', N);
     
          if  (_abs < tolerance):
              end_time = perf_counter()
              #duration = end_time - start_time
              duration = tot_LM_time
              print("time = ", duration, " seconds, and ", duration/60, " minutes");
              print("iterations ", j);#'returning alpha_1= ', alpha_1);#, "log_L_0 = ", log_L_0, " and log_L_1 = ",log_L_1);
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
    #print("Minka by LM: uploading Dataset from file ", _file);
    D_counts, N = load_dataset_from_file(_file, _delimiter)
    R,K = D_counts.shape

    if(wrapped == 1):
         _alpha =  minka_procedure_LM_c_wrapped_mode(prec, tolerance, D_counts, maxiter,  R, K)#minka_procedure_LM_mode_c_wrapped(prec, tolerance, D_counts, maxiter, R, K)
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

   

    #asymp = np.float64(0)
    #asymp1 = np.float64(0)

    start_time = perf_counter()
    #print("FIXED-POINT ITERATIONS TO FIND PSI USING LM STARTED, GOING FOR ", maxiter, " ITERATIONS AT MAX.");
    print("precision is ", prec, " tolerance (i.e. LM_accuracy) is ", tolerance, " maxiter is ", maxiter, " R is ", R, " K is ", K);
    res = mp.mpf(0) # mpmath variable
    res64 = np.float64(0)
    
    resdigamma = mp.mpf(0) # mpmath variable
    restrigamma = mp.mpf(0) # mpmath variable
    '''
    res_LM = np.float64(0)  
    LM_accuracy = tolerance #np.float64(0)
    LMm = int(0)
    horshift = int(0)
    resdigamma_LM = np.float64(0)
    LMm_digamma = int(0)
    horshift_digamma = int(0)
    restrigamma_LM = np.float64(0)
    LMm_trigamma = int(0)
    horshift_trigamma = int(0)
    #initialize the LM vectors
    [vec_evenbernoulli, vec_evenbernoullinorm, vec_evenbernoullinormdigamma, vec_errcoeff, vec_errcoeffdigamma, vec_errcoefftrigamma] = LM_initBern()
    '''
    ''' 
    res_YSmesh = np.float64(0)
    YStotsumy = np.float64(0)   
    YSL = int(0)
    YSmesh_accuracy_sharp = np.float64(0)
    YSmymesh_accuracy_weak = np.float64(0)  
            
    res_mymesh = np.float64(0)
    mytotsumy = np.float64(0)
    myL = int(0)
    mymesh_accuracy_sharp = np.float64(0)
    mymesh_accuracy_weak = np.float64(0)    
    '''
    common_err = np.float64(0)
    
    N = np.float64(0)   
    mplusone = int(0)
    SIGDIG  = int(15) # number of significative decimal digits in C-double (float)
    bound  = int(0)

    
 
    # find vector of counts X using D_counts
    X = find_X(D_counts)
    N = find_N(X)
    #N = np.int(0)
    #for k in range(K):
    #    N = N + X[k]
          

    #alpha vectors
    #alpha_0: a K-value vector of initially guessed alpha values, using Minka's original method
    alpha_1 = np.zeros(K, dtype=np.float64)
    alpha_0 = _init_a_LM(D_counts) # _init_a_LM used in DEFAUL, too
    #print("alpha_0 before iterations ");
    #print(np.matrix(alpha_0));
    alpha_fail = np.empty(K, dtype=float)  # initialization
    for k in range(K):
        alpha_fail[k] = -1.0

    # checking if the problem can be handled with the C double precision (float)
    #SIGDIG = 15 # number of available digits in float
    # bound = SIGDIG - prec - 2 # 2 is experimentally determined on gcc on INTEL i7
    bound = SIGDIG - prec  #
    #print(bound)
    
    if (check_with_asym1(bound, X, K ) == -1) :
        print('*********** ERROR FROM MINKA by DEFAULT PROCEDURE: LogL too large to ensure the desired precision in double; switch to multiprecision')
        #print('*********** ABORTING')
        return alpha_fail
        #sys.exit('*********** ERROR: LogL too large to ensure the desired precision in double; switch to multiprecision')
        #endif

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
                #diff = polygamma(0, x+y) - polygamma(0, y) #digamma(x,y)
                #diff = digamma(x+y) - digamma(y)
                diff = scipydigamma64(x+y)- scipydigamma64(y)
                #diff = -y*np.log(1.0/x) + yu_shaw_D_m(1.0/x, y, delta, m)
                sum_i = sum_i + diff 
                
            sum_i_2 = np.float64(0)
            for i in range(R):
                n_i = find_ni(D_counts, i)
                x = n_i 
                y = sum_a0
                #diff = digamma(x+y) - digamma(y)
                diff = scipydigamma64(x+y)- scipydigamma64(y)
                #diff = polygamma(0, x+y) - polygamma(0, y) #digamma(x,y)
                #diff = -y*np.log(1.0/x) +yu_shaw_D_m(1.0/x, y, delta, m)
                sum_i_2 = sum_i_2 + diff
               
            alpha_1[k] = alpha_0[k] * (sum_i/sum_i_2)
            #if(alpha_1[k] < 1.0/delta):
            #    alpha_1[k] = 1.0/delta

          #current alpha (alpha_1) is ready
          #print("in iteration ", j, " alpha_0 = ");
          #print(np.matrix(alpha_0));#",   alpha_1 = ", alpha_1, " and alpha_0 is ", alpha_0);
          #print("alpha_1 is");
          #print(np.matrix(alpha_1));
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
              start_time_0 = perf_counter()# time.time()
              log_L_0 = ver_logL_64(N, psi_0_DEFAULT, psioverprobs_0, X, K) #ver_logLscipy64 #ver_logL_64 #main_LM_logL(P_est_0_LM, psi_0_LM, K, X, vec_evenbernoullinorm, LM_accuracy)
              end_time_0 = perf_counter()
              DEFAULTexec_time_0 = end_time_0 - start_time_0
              start_time_1 = perf_counter()# time.time()
              log_L_1 = ver_logL_64(N, psi_1_DEFAULT, psioverprobs_1, X, K)#ver_logLscipy64#ver_logL_64 #main_LM_logL(P_est_1_LM, psi_1_LM, K, X, vec_evenbernoullinorm, LM_accuracy)
              end_time_1 = perf_counter()
              DEFAULTexec_time_1 = end_time_1 - start_time_1
          else:
              start_time_0 = perf_counter()# time.time()
              log_L_0 = ver_logLscipy64(N, psi_0_DEFAULT, psioverprobs_0, X, K) # #ver_logL_64 #main_LM_logL(P_est_0_LM, psi_0_LM, K, X, vec_evenbernoullinorm, LM_accuracy)
              end_time_0 = perf_counter()
              DEFAULTexec_time_0 = end_time_0 - start_time_0
              start_time_1 = perf_counter()# time.time()
              log_L_1 = ver_logLscipy64(N, psi_1_DEFAULT, psioverprobs_1, X, K)#ver_logLscipy64#ver_logL_64 #main_LM_logL(P_est_1_LM, psi_1_LM, K, X, vec_evenbernoullinorm, LM_accuracy)
              end_time_1 = perf_counter()
              DEFAULTexec_time_1 = end_time_1 - start_time_1

          # add up time    
          tot_default_time = tot_default_time + DEFAULTexec_time_0 + DEFAULTexec_time_1
          
          #if( j%1000 == 0):
          #   print('iteration ', j, ' DEFAULTexec_time_0 = ', DEFAULTexec_time_0, ',', ' DEFAULTexec_time_1 = ', DEFAULTexec_time_1 );

          _abs = abs((log_L_0) - (log_L_1))
        
          #if( j%1000 == 0):
          #    #print('iteration ', j, ' _abs = ', _abs, ',', ' tolerance = ', tolerance );
          #    print('iteration ', j, ' _abs = ', _abs, ',', ' tolerance = ', tolerance, ' log_L_0 = ', log_L_0);
          #    #, ' log_L_1= ', log_L_1 ,
          #    #      ' psi_0_DEFAULT = ', psi_0_DEFAULT, ' psi_1_DEFAULT = ', psi_1_DEFAULT, 'P_est_0_DEFAULT = ', P_est_0_DEFAULT,
          #    #      ' P_est_1_DEFAULT = ', P_est_1_DEFAULT, ' X = ', X, ' N = ', N );
     
          if  (_abs < tolerance):
              end_time = perf_counter()
              #duration = end_time - start_time
              duration = tot_default_time
              print("time = ", duration, " seconds, and ", duration/60, " minutes");
              print("iterations ", j);#'returning alpha_1= ', alpha_1);#, "log_L_0 = ", log_L_0, " and log_L_1 = ",log_L_1);
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

###############################################################     	
#################  EXPERIMENTS ###############
###############################################################

# this function computes logL, digamma, trigamma likelihood functions for a dataset 
def run_experiment(K, psi, probs, X, delta, YSm):
	# input: K, psi, probs, X: from the dataset; delta, YSm: for the D_m sum
	# input: K: number of categories in the dataset
	# input: psi: value of the overdisposition parameter psi
	# input: probs: array of K positions: probabilities of each category
	# input: X: array of K positions: counts for each category		
	# input: delta: convergence parameter of the YS-sum Dm; for the condition xy <= delta
	# input: YSm: number of summands in the YS-sum Dm
	#
	# primary outputs	
	# output: asymp: asymptotic_state of the system as psi->0+	
	# output: asymp1: asymptotic_state of the system as psi->0+ but using frequencies
	# output: res; res64: logL value computed using mpmath or math lngamma functions; no mesh
	# output: res_my_mesh; res_YS_mesh : logL value computed using my_mesh or YS_mesh of points
	# output: res_LM: logL value computed using LM-paper
	# output: ver_logL - res_my_mesh: actual distance between the two ways of computing logL; the same for res64
	# output: my_mesh_accuracy_sharp; YS_mesh_accuracy_sharp : sharp estimate of the accuracy in computing res_my_mesh
	# output: my_mesh_accuracy_weak; YS_mesh_accuracy_weak: weak estimate of the accuracy in computing res_my_mesh	
	# output: myL, YSL: mesh levels: needed for the weak error estimate
	# output: running times for res and res_my_mesh; and their ratio; the same for res and res_YS_mesh
	# 
	# secondary outputs
	# output: resdigamma: paired difference of digammas over categories; computed with mpmath to check other results; no mesh
	# output: restrigamma: paired difference of trigammas over categories; computed with mpmath to check other results; no mesh
	# output: resdigamma_LM: paired difference of digammas over categories; computed using LM-paper
	# output: restrigamma_LM: paired difference of trigammas over categories; computed using LM-paper

	# types and initializations
	asymp = np.float64(0)
	asymp1 = np.float64(0)
	
	mp.mp.dps = 53 # prints 53 digits in multiprecision
	res = mp.mpf(0) # mpmath variable
	resdigamma = mp.mpf(0) # mpmath variable
	restrigamma = mp.mpf(0) # mpmath variable

	res64 = np.float64(0)
	resscipy64 = np.float64(0)	
	resscipydigamma64 = np.float64(0)	
	resscipytrigamma64 = np.float64(0)	
	
	res_LM = np.float64(0)	
	LM_accuracy = np.float64(0)
	LMm = int(0)
	hor_shift = int(0)
	resdigamma_LM = np.float64(0)
	LMm_digamma = int(0)
	hor_shift_digamma = int(0)
	restrigamma_LM = np.float64(0)
	LMm_trigamma = int(0)
	hor_shift_trigamma = int(0)
		
	res_YS_mesh = np.float64(0)
	YStotsumy = np.float64(0)	
	YSL = int(0)
	YS_mesh_accuracy_sharp = np.float64(0)
	YSmy_mesh_accuracy_weak = np.float64(0)	
			
	res_my_mesh = np.float64(0)
	mytotsumy = np.float64(0)
	myL = int(0)
	my_mesh_accuracy_sharp = np.float64(0)
	my_mesh_accuracy_weak = np.float64(0)	
	
	common_err = np.float64(0)
	
	N = np.float64(0)	
	mplusone = int(YSm+1)
	SIGDIG  = int(15) # number of significative decimal digits in C-double (float)
	bound  = int(0)

	# computing the needed secondary parameters
	psioverprobs = np.empty(K, dtype = float) # initialization
	alpha = (np.zeros(K, dtype = float)).tolist()  # initialization; inserted the tolist to force the output to print the commas


	for k in range (0, K):  
		psioverprobs[k] = psi/probs[k] # needed for the D_m sum
		alpha[k] = probs[k]/psi	# alpha vector			
		N += X[k] # computing the sum of the cardinalities
		asymp += X[k]*log(probs[k]) # computing the asymptotic_state of the system as psi->0+	
	#endfor	   
    	  
	for k in range (0, K):  
		asymp1 += X[k]*log(X[k]/N)	# computing the asymptotic_state of the system as psi->0+ with the frequencies
	#endfor	    
    
	# checking if the problem can be handled with the C double precision (float)
	#SIGDIG = 15 # number of available digits in float
	# bound = SIGDIG - prec - 2 # 2 is experimentally determined on gcc on INTEL i7
	bound = SIGDIG - prec  #
	#print(bound)
	if log10(abs(asymp)) > bound :
		print('*********** ERROR: LogL too large to ensure the desired precision in double; switch to multiprecision')
		print('*********** ABORTING')
		sys.exit('*********** ERROR: LogL too large to ensure the desired precision in double; switch to multiprecision')
	#endif

    # delta: convergence parameter of the YS-sum Dm; for the condition xy <= delta   
	# fixed to 0.2 in YS-paper    
	if ( delta < 0 or delta >= 1): # coherence check for delta
		print('*********** ERROR: delta has to be in (0,1)')
		return()
	#endif	

    # YSm: number of summands in the YS-Dm sum;     	   
	# fixed to 20 in YS-paper 
	if ( YSm < 1 or YSm >= 51): # coherence check for YSm
		print('*********** ERROR: YSm has to be in [1,50]')
		return()
	#endif	
		
	print ('--------')	
	print('PARAMETERS') 
	print ('--------')		
	print('N =',N)
	print('K =', K) 
	print('psi =',psi)
	print('probabilities =', probs)
	print('counts vector =', X)
	print('alpha vector =', alpha)
	#print('psi/probs vector =', psioverprobs)
	print('asymptotic_state; using probabilities (psi -> 0+) =', asymp)
	print('asymptotic_state_1; using frequencies (psi -> 0+) =', asymp1)		
	print('required accuracy for LM (mantissa): at least', prec, 'decimal digits')			
	print ('--------------------------------')		
	print ('-------- REMARK --------')
	print ('Precomputed Bernoulli numbers for LM and old Bernoulli polynomials for both YS_mesh and my_mesh')
	print ('The exec times to perform both computations and read these data are not considered')
	print ('--------------------------------')		
	
	# logL computation mpmath
	start_time = perf_counter()# time.time()
	res = ver_logL(N, psi, psioverprobs, X, K)
	end_time = perf_counter() #time.time()	
	no_meshexec_time = end_time - start_time 
	
	# logL computation math (float64 format)
	start_time = perf_counter()# time.time()
	res64 = ver_logL_64(N, psi, psioverprobs, X, K)
	end_time = perf_counter() #time.time()	
	no_meshexec64_time = end_time - start_time 
	
	# logL computation scipy (float64 format)
	start_time = perf_counter()# time.time()
	resscipy64 = ver_logLscipy64(N, psi, psioverprobs, X, K)
	end_time = perf_counter() #time.time()	
	no_meshscipyexec64_time = end_time - start_time 
	
	# LMlogL computation
	# read the precomputed coefficients for the LM functions
	[vec_evenbernoulli, vec_evenbernoullinorm, vec_evenbernoullinormdigamma, vec_errcoeff, vec_errcoeff_digamma, vec_errcoeff_trigamma] = LM_initBern()    		
	
	start_time = perf_counter()# time.time()
	# we divide by (K+1) since we want that the total error has prec decimal digits correct
	LM_accuracy = (10 ** (-prec-2))/(K+1) 
	res_LM = main_LM_logL(probs, psi, K, X, vec_evenbernoullinorm, vec_errcoeff, LM_accuracy)
	end_time = perf_counter()
	LMexec_time = end_time - start_time 
	
	# YS_mesh logL computation
	start_time = perf_counter()# time.time()
	[res_YS_mesh, YSL, YStotsumy] = ver_logL_YS_mesh(N, psi, psioverprobs, alpha, probs, X, K, delta, YSm) 
	end_time = perf_counter()
	YS_meshexec_time = end_time - start_time 
	
	# working on the error term estimates
	common_err = delta**YSm/(mplusone*YSm) # for the error terms
	YS_mesh_accuracy_sharp = common_err * YStotsumy # sharp error term estimate
	YS_mesh_accuracy_weak = 2/psi * common_err * ((1+delta)**YSL - 1) # weak error term estimate

	# my_mesh logL computation
	start_time = perf_counter()# time.time()
	[res_my_mesh, myL, mytotsumy] = ver_logL_my_mesh(N, psi, psioverprobs, X, K, delta, YSm) 
	end_time = perf_counter() #time.time()	
	my_meshexec_time = end_time - start_time 
	# working on the error term estimates
	#common_err = delta**YSm/(mplusone*YSm) # for the error terms
	my_mesh_accuracy_sharp = common_err * mytotsumy # sharp error term estimate
	my_mesh_accuracy_weak = 2/psi * common_err * ((1+delta)**myL - 1) # weak error term estimate

	# other functions values: digamma, trigamma for this dataset
	# with mpmath
	resdigamma = ver_digamma (N, psi, psioverprobs, X, K)
	restrigamma = ver_trigamma (N, psi, psioverprobs, X, K)
	# with scipy
	resscipydigamma64 = ver_scipydigamma64 (N, psi, psioverprobs, X, K)
	resscipytrigamma64 = ver_scipytrigamma64 (N, psi, psioverprobs, X, K)
	# with Languasco-Migliardi
	resdigamma_LM = main_LM_digamma(probs, psi, K, X, vec_evenbernoullinormdigamma, vec_errcoeff_digamma, LM_accuracy)
	restrigamma_LM = main_LM_trigamma(probs, psi, K, X, vec_evenbernoulli, vec_errcoeff_trigamma, LM_accuracy)

	
	print ('--------')	
	print('RESULTS') 
	print ('--------')	
	print('logL (mpmath) =', res)
	print('logL (mpmath) execution time (seconds) =', no_meshexec_time)
	print('logL (float)  =', res64)
	print('logL (float) execution time (seconds) =', no_meshexec64_time)
	print('logL (scipy)  =', resscipy64)
	print('logL (scipy) execution time (seconds) =', no_meshscipyexec64_time)

	print('--------  Languasco - Migliardi  ----- ')
	if res_LM == 0 :
		print('*************** ERROR: LM not applicable; Euler-Maclaurin (logGamma) formula not accurate enough')		
	else:
		print('required accuracy for LM (mantissa) =', prec, 'decimal digits')			
		print('mantissa accuracy requested for LM =', LM_accuracy)
		print('logL_LM       =', res_LM)
		print('difference (logL(mpmath)-logL_LM) [logs-accuracy affected] =', float(res - res_LM))
		print('difference (logL_64 -logL_LM) [logs-accuracy affected] =', res64 - res_LM)					
		print('LM logL execution time (seconds) =', LMexec_time)
		print('Ratio time logL LM/no_mesh(mpmath) =', LMexec_time/no_meshexec_time)
		print('Ratio time logL LM/no_mesh_64(math) =', LMexec_time/no_meshexec64_time)
		print('Ratio time logL LM/no_mesh_64(scipy) =', LMexec_time/no_meshscipyexec64_time)		
	
	#endif
	
	#print ('----   MESHES  ----- ')	
	if res_my_mesh == 0:
		print('-------- ')
		print('--------  mesh results UNAVAILABLE ----- ')
		print('*************** ERROR: delta-mesh not applicable; try, e.g., to change delta; pay attention at the accuracy, though !!')
	else:
		print('--------  my mesh  ----- ')
		print('logL_my_mesh  =', res_my_mesh)
		print('difference (logL(mpmath)-logL_my_mesh) [logs-accuracy affected] =', float(res - res_my_mesh))
		print('difference (logL_64 -logL_my_mesh) [logs-accuracy affected] =', res64 - res_my_mesh)	
		print('delta mesh parameter =', delta)	
		print('sum level m in D_m =', YSm)	
		print('level of my_mesh (max L_k ) =', myL)
		print('sharp upper bound my_mesh accuracy [only the tail of the series (UL-paper, eq. (18))] =', my_mesh_accuracy_sharp)	
		print('weak upper bound my_mesh accuracy [only the tail of the series (UL-paper, eq. (19))] =', my_mesh_accuracy_weak)	
		print('my_mesh logL execution time (seconds) =', my_meshexec_time)
		print('Ratio time logL my_mesh/no_mesh(mpmath) =', my_meshexec_time/no_meshexec_time)
		print('Ratio time logL my_mesh/no_mesh_64(math)=', my_meshexec_time/no_meshexec64_time)
		print('Ratio time logL my_mesh/no_mesh_64(scipy) =', my_meshexec_time/no_meshscipyexec64_time)		
		print('--------  YS mesh  ----- ')
		print('logL_YS_mesh   =', res_YS_mesh)
		print('difference (logL-logL_YS_mesh)[logs-accuracy affected] =', float(res - res_YS_mesh))	
		print('delta mesh parameter =', delta)	
		print('sum level m in D_m =', YSm)	
		print('difference (logL_64 -logL_YS_mesh) [logs-accuracy affected] =', res64 - res_YS_mesh)			
		print('level of YS_mesh (max L_k) =', YSL)
		print('sharp upper bound YS_mesh accuracy [only the tail of the series (UL-paper, eq. (18))] =', YS_mesh_accuracy_sharp)	
		print('weak upper bound YS_mesh accuracy [only the tail of the series (UL-paper, eq. (19))] =', YS_mesh_accuracy_weak)	
		print('Ratio time logL YS_mesh/no_mesh(mpmath)=', YS_meshexec_time/no_meshexec_time)
		print('Ratio time logL YS_mesh/no_mesh_64(math)=', YS_meshexec_time/no_meshexec64_time)
		print('Ratio time logL YS_mesh/no_mesh_64(scipy)=', YS_meshexec_time/no_meshscipyexec64_time)		
	#endif	
	
	print ('------------')
	print('Other, maybe relevant, values: digamma; trigamma: (mpmath and LM)')	
	print ('------------')
	print('digamma (mpmath)        =', resdigamma)
	print('digamma (scipy float64) =', resscipydigamma64)
	if resdigamma_LM == 0 :
		print('*************** ERROR: LM not applicable; Euler-Maclaurin (digamma) formula not accurate enough')	
	else:	
		print('required accuracy for LM (mantissa) =', prec, 'decimal digits')			
		print('mantissa accuracy requested for LM =', LM_accuracy)
		print('digamma_LM              =', resdigamma_LM)
		print('digamma(mpmath) - digamma_LM =', float(resdigamma - resdigamma_LM))
	#endif
	print ('----')	
	print('trigamma (mpmath)        =', restrigamma)
	print('trigamma (scipy float64) =', resscipytrigamma64)	
	if restrigamma_LM == 0 :
		print('*************** ERROR: LM not applicable; Euler-Maclaurin (trigamma) formula not accurate enough')					
	else:	
		print('required accuracy for LM (mantissa) =', prec, 'decimal digits')			
		print('mantissa accuracy requested for LM =', LM_accuracy)
		print('trigamma_LM              =', restrigamma_LM)	
		print('trigamma(mpmath) - trigamma_LM =', float(restrigamma - restrigamma_LM))
	#endif
	print ('------------')	
	print ('---------------------------------------------')	
# end run_experiment()

# first experiment Sherenaz
def exp_1():
	# experiment primary parameters: change them for another experiment
	print ('---------------------------------------------')	
	print('Experiment 1: toy dataset')

	K = int(3)
	psi = np.float64(0.0006553541292731333)
	delta = np.float64(0.2)
	YSm = int(20)

	probs = np.empty(K, dtype=float) # initialization
	X = np.empty(K, dtype=float)  # initialization
	probs = [0.52503178,0.45019068,0.02477754] 
	X = [49563,42498,2339]  

	
	run_experiment(K, psi, probs, X, delta, YSm)
	print ('---------------------------------------------')	
	print('**** END experiment 1')
	print ('---------------------------------------------')	
# end exp_1()

# second experiment Sherenaz
def exp_2():
	# experiment primary parameters: change them for another experiment
	print ('---------------------------------------------')	
	print('Experiment 2: toy dataset')

	K = int(3)
	psi = np.float64(0.00035163722205396167)
	delta = np.float64(0.2)
	YSm = int(20)

	probs = np.empty(K, dtype=float) # initialization
	X = np.empty(K, dtype=float)  # initialization	
	probs = [0.52503178,0.45019068,0.02477754] 
	X = [49563,42498,2339]  
	
	run_experiment(K, psi, probs, X, delta, YSm)
	print ('---------------------------------------------')	
	print('**** END experiment 2')
	print ('---------------------------------------------')	
# end exp_2()

# third experiment Sherenaz
def exp_3():
	# experiment primary parameters: change them for another experiment
	print ('---------------------------------------------')	
	print('Experiment 3: Biden-Trump dataset')
	print('Data from fixed point approx obtained using YU-SHAW')

	K = int(3)
	psi = np.float64(0.011703752923417576)
	delta = np.float64(0.2)
	YSm = int(20)

	probs = np.empty(K, dtype=float) # initialization
	X = np.empty(K, dtype=float)  # initialization	
	probs = [0.47819390270733633742406496,0.46328733271014178850588744,0.05851876461708788] 
	X = [208904,129528,5516]


	run_experiment(K, psi, probs, X, delta, YSm)
	print ('---------------------------------------------')	
	print('**** END experiment 3')
	print ('---------------------------------------------')	
# end exp_3()

# fourth experiment Sherenaz
def exp_4():
	# experiment primary parameters: change them for another experiment
	print ('---------------------------------------------')	
	print('Experiment 4: Biden-Trump dataset')
	print('Data from fixed point approx obtained using LM')
	
	K = int(3)
	psi = np.float64(0.05311523264120303)
	delta = np.float64(0.2)
	YSm = int(20)

	probs = np.empty(K, dtype=float) # initialization
	X = np.empty(K, dtype=float)  # initialization	
	probs = [0.4792271623353988067803923,0.4640208533731721268766011,0.056751984816436824381373] 
	X = [208904,129528,5516] 
	
	run_experiment(K, psi, probs, X, delta, YSm)
	print ('---------------------------------------------')	
	print('**** END experiment 4')
	print ('---------------------------------------------')	
# end exp_4()

# fifth experiment Sherenaz
def exp_5():
	# experiment primary parameters: change them for another experiment
	print ('---------------------------------------------')	
	print('Experiment 5: Biden-Trump dataset')
	print('Data from fixed point approx obtained using the scipy_digamma function')
	
	K = int(3)
	psi = np.float64(0.055978273265730884)
	delta = np.float64(0.2)
	YSm = int(20)

	probs = np.empty(K, dtype=float) # initialization
	X = np.empty(K, dtype=float)  # initialization	
	probs = [0.4798162494870931076124994, 0.46460834705958107578601984, 0.0555754032308647454009404]
	X = [208904,129528,5516]

	run_experiment(K, psi, probs, X, delta, YSm)
	print ('---------------------------------------------')	
	print('**** END experiment 5')
	print ('---------------------------------------------')	
# end exp_5()

# fig5 experiment Yu-Shaw
def exp_6():
	# experiment primary parameters: change them for another experiment
	print ('---------------------------------------------')	
	print('Experiment 6: figure 5 Yu Shaw example')
	
	K = int(4)
	psi = np.float64(1/200)
	delta = np.float64(0.2)
	YSm = int(20)


	probs = np.empty(K, dtype=float) # initialization
	X = np.empty(K, dtype=float)  # initialization	
	probs = [0.1, 0.2, 0.3, 0.4]
	
	size = int(20)
	n = int(0)
	
	for i in range (1, size+1):
		print('i =', i)
		n=2500*i
		X=[n, n, n, n]

		run_experiment(K, psi, probs, X, delta, YSm)
		print ('---------------------------------------------')	
	#endfor
		
	print('**** END experiment 6')
	print ('---------------------------------------------')	
# end exp_6()

# fig6 experiment Yu-Shaw
def exp_7():
	# experiment primary parameters: change them for another experiment
	print ('---------------------------------------------')	
	print('Experiment 7: figure 6 Yu Shaw example')
	
	K = int(3)
	psi = np.float64(1/60)
	delta = np.float64(0.2)
	YSm = int(20)

	probs = np.empty(K, dtype=float) # initialization
	X = np.empty(K, dtype=float)  # initialization	
	probs = [1/6, 1/3, 1/2]
	
	size = int(20)
	n = int(0)
	
	for i in range (2, size+1):
		print('i =', i)
		n=10**i
		X=[n, 2*n, 3*n];  

		run_experiment(K, psi, probs, X, delta, YSm)
		print ('---------------------------------------------')	
	#endfor	
	print('**** END experiment 7')
	print ('---------------------------------------------')	
# end exp_7()

def exp_8():
	# experiment primary parameters: change them for another experiment
	print ('---------------------------------------------')	
	print('Experiment 8: for handling the special cases y=1; y=2')

	K = int(3)
	psi = np.float64(0.0006553541292731333)
	delta = np.float64(0.2)
	YSm = int(20)

	probs = np.empty(K, dtype=float) # initialization
	X = np.empty(K, dtype=float)  # initialization
	probs = [0.52503178,0.45019068,0.02477754] 
	X = [49563,2,1]  

	
	run_experiment(K, psi, probs, X, delta, YSm)
	print ('---------------------------------------------')	
	print('**** END experiment 8')
	print ('---------------------------------------------')	
# end exp_8()

    
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

######  Sherenaz experiments  ######

_YSm = 20
_delta = 0.2
_maxiter = 1000000
conf = 0.05
dataset_file = "test_fix_counts.csv"
all_dataset_file = "all_fix_counts.csv"
D, N = load_dataset_from_file(dataset_file, delim=',')
X = find_X(D)
K = len(X)#.shape
tolerance = 10 ** (-prec-2)/(K+1) #Apr. 20th, 2023: added by Sherenaz
#tolerance = tolerance*1000000# 1000000 # should not do this!!! but just to speedup convergernce to see if YS would work
_factor = 500

#overdispersion index
D_real, N_real = load_dataset_from_file(all_dataset_file, delim=',')
X_real = find_X(D_real) 
#real_mean  = mean_random_variable_counts(X_real)
#real_variance = variance_random_variable_counts(X_real)
#print('OVERDISPERSION INDEX = ', real_variance/real_mean);

##minka by LM 
#alpha, psi = minka_LM_experiment_from_file(dataset_file, ",",prec,tolerance, _maxiter )#minka_LM_experiment_from_file("test_fix_counts.csv", ",",prec,tolerance, _maxiter )
#print("alpha = ", alpha, "psi = ", psi);

scipy_mode = 0
##minka by DEFAULT  experiments
print('Minka by DEFAULT experiments (math loggamma)');
alpha, psi = minka_DEFAULT_experiment_from_file(dataset_file, ",",prec,tolerance, _maxiter,scipy_mode )#minka_DEFAULT_experiment_from_file("test_fix_counts.csv", ",",prec,tolerance, _maxiter )


#D, N = load_dataset_from_file("all_fix_counts.csv", delim=',')



real_probs = find_probs(all_dataset_file, ",")#find_probs("test_fix_counts.csv", ",") #


#print("real_probs = ", real_probs);
expected_probs = alpha*psi
#print("expected_probs = ", expected_probs);

print('************************************************************************************');
print('************************************************************************************');

print('Minka by DEFAULT experiments (scipy loggamma ');
scipy_mode = 1
alpha, psi = minka_DEFAULT_experiment_from_file(dataset_file, ",",prec,tolerance, _maxiter, scipy_mode )#minka_DEFAULT_experiment_from_file("test_fix_counts.csv", ",",prec,tolerance, _maxiter )


#D, N = load_dataset_from_file("all_fix_counts.csv", delim=',')



real_probs = find_probs(all_dataset_file, ",")#find_probs("test_fix_counts.csv", ",") #


#print("real_probs = ", real_probs);
expected_probs = alpha*psi
#print("expected_probs = ", expected_probs);


'''
print("tvd = ",tvd(real_probs, expected_probs));

fake_probs = np.array([0.505, 0.375, 0.001, 0.001, 0.118])   
#print("fake probs = ",fake_probs)

real_probs_wider = np.zeros(K, np.longdouble)
alpha_wider = np.zeros(K, np.longdouble)
for k in range(K):
    real_probs_wider[k] = real_probs[k]
    alpha_wider[k] = alpha[k]




diff_probs =  real_probs - expected_probs
print("Euclidean =  ", LA.norm(diff_probs));
print("MSE = ", _mean_square_error(real_probs,expected_probs ));


real_counts = real_probs* _factor
expected_counts = expected_probs* _factor
fake_counts = fake_probs* _factor



real_check = check_for_5(real_counts)
expected_check = check_for_5(expected_counts)
fake_check = check_for_5(fake_counts)

if real_check == 0 and expected_check == 0:
    chi_square_test(real_counts, expected_counts, K-1, conf)
else:
    print("sample size less than 5 in at least one category, cannot perform chi square test");

print("!!FAKE Chi-Square TEST!!")
if real_check == 0 and fake_check == 0:
    chi_square_test(real_counts, fake_counts, K-1, conf)
else:
    print("!!FAKE Chi-Square !!: sample size less than 5 in at least one category, cannot perform chi square test");

diff_mean = np.float64(0)
diff_variance = np.float64(0)

diff_mean = abs(mean_random_variable(real_probs)- mean_random_variable(expected_probs))
diff_variance = abs(variance_random_variable(real_probs)- variance_random_variable(expected_probs))



the_Z_test(mean_random_variable(real_probs), variance_random_variable(real_probs), _factor,
           mean_random_variable(expected_probs), variance_random_variable(expected_probs), _factor )


kl_distance = KL_distance(real_probs,expected_probs)

print("KL ditance is ", kl_distance, " and the reverse KL-distnce is ", KL_distance(expected_probs, real_probs));



'''
'''
##minka by LM  experiments
print('Minka by LM experiments (Pure Python)');
wrapped = 0
alpha, psi = minka_LM_experiment_from_file(dataset_file, ",",prec,tolerance, _maxiter, wrapped )#minka_LM_experiment_from_file("test_fix_counts.csv", ",",prec,tolerance, _maxiter )


K = len(alpha)#.shape
#D, N = load_dataset_from_file("all_fix_counts.csv", delim=',')


real_probs = find_probs(all_dataset_file, ",")#find_probs("test_fix_counts.csv", ",") #


#print("real_probs = ", real_probs);
expected_probs = alpha*psi
#print("expected_probs = ", expected_probs);
'''


'''
print("tvd = ",tvd(real_probs, expected_probs));

fake_probs = np.array([0.505, 0.375, 0.001, 0.001, 0.118])   
#print("fake probs = ",fake_probs)

real_probs_wider = np.zeros(K, np.longdouble)
alpha_wider = np.zeros(K, np.longdouble)
for k in range(K):
    real_probs_wider[k] = real_probs[k]
    alpha_wider[k] = alpha[k]




diff_probs =  real_probs - expected_probs
print("Euclidean =  ", LA.norm(diff_probs));
print("MSE = ", _mean_square_error(real_probs,expected_probs ));


real_counts = real_probs* _factor
expected_counts = expected_probs* _factor
fake_counts = fake_probs* _factor



real_check = check_for_5(real_counts)
expected_check = check_for_5(expected_counts)
fake_check = check_for_5(fake_counts)

if real_check == 0 and expected_check == 0:
    chi_square_test(real_counts, expected_counts, K-1, conf)
else:
    print("sample size less than 5 in at least one category, cannot perform chi square test");

#print("!!FAKE Chi-Square TEST!!")
#if real_check == 0 and fake_check == 0:
#    chi_square_test(real_counts, fake_counts, K-1, conf)
#else:
#    print("!!FAKE Chi-Square !!: sample size less than 5 in at least one category, cannot perform chi square test");

diff_mean = np.float64(0)
diff_variance = np.float64(0)

diff_mean = abs(mean_random_variable(real_probs)- mean_random_variable(expected_probs))
diff_variance = abs(variance_random_variable(real_probs)- variance_random_variable(expected_probs))



the_Z_test(mean_random_variable(real_probs), variance_random_variable(real_probs), _factor,
           mean_random_variable(expected_probs), variance_random_variable(expected_probs), _factor )


kl_distance = KL_distance(real_probs,expected_probs)

print("KL ditance is ", kl_distance, " and the reverse KL-distnce is ", KL_distance(expected_probs, real_probs));

#print("Fake Tests");
#print("FAKE Z Test");
#the_Z_test(mean_random_variable(real_probs), variance_random_variable(real_probs), _factor,
#          mean_random_variable(fake_probs), variance_random_variable(fake_probs), _factor )

'''
print('************************************************************************************');
print('************************************************************************************');

##minka by LM  experiments
print('Minka by LM experiments (C-Wrapped version):');
wrapped = 1
alpha, psi = minka_LM_experiment_from_file(dataset_file, ",",prec,tolerance, _maxiter, wrapped )#minka_LM_experiment_from_file("test_fix_counts.csv", ",",prec,tolerance, _maxiter )

'''
K = len(alpha)#.shape

real_probs = find_probs(all_dataset_file, ",")
expected_probs = alpha*psi


real_probs_wider = np.zeros(K, np.longdouble)
alpha_wider = np.zeros(K, np.longdouble)
for k in range(K):
    real_probs_wider[k] = real_probs[k]
    alpha_wider[k] = alpha[k]




'''
K = len(alpha)#.shape
#D, N = load_dataset_from_file("all_fix_counts.csv", delim=',')


real_probs = find_probs(all_dataset_file, ",")#find_probs("test_fix_counts.csv", ",") #


#print("real_probs = ", real_probs);
expected_probs = alpha*psi
#print("expected_probs = ", expected_probs);




print("tvd = ",tvd(real_probs, expected_probs));

fake_probs = np.array([0.505, 0.375, 0.001, 0.001, 0.118])   
#print("fake probs = ",fake_probs)

real_probs_wider = np.zeros(K, np.longdouble)
alpha_wider = np.zeros(K, np.longdouble)
for k in range(K):
    real_probs_wider[k] = real_probs[k]
    alpha_wider[k] = alpha[k]




diff_probs =  real_probs - expected_probs
print("Euclidean =  ", LA.norm(diff_probs));
print("MSE = ", _mean_square_error(real_probs,expected_probs ));


real_counts = real_probs* _factor
expected_counts = expected_probs* _factor
fake_counts = fake_probs* _factor



real_check = check_for_5(real_counts)
expected_check = check_for_5(expected_counts)
fake_check = check_for_5(fake_counts)

if real_check == 0 and expected_check == 0:
    chi_square_test(real_counts, expected_counts, K-1, conf)
else:
    print("sample size less than 5 in at least one category, cannot perform chi square test");

#print("!!FAKE Chi-Square TEST!!")
#if real_check == 0 and fake_check == 0:
#    chi_square_test(real_counts, fake_counts, K-1, conf)
#else:
#    print("!!FAKE Chi-Square !!: sample size less than 5 in at least one category, cannot perform chi square test");

diff_mean = np.float64(0)
diff_variance = np.float64(0)

diff_mean = abs(mean_random_variable(real_probs)- mean_random_variable(expected_probs))
diff_variance = abs(variance_random_variable(real_probs)- variance_random_variable(expected_probs))



the_Z_test(mean_random_variable(real_probs), variance_random_variable(real_probs), _factor,
           mean_random_variable(expected_probs), variance_random_variable(expected_probs), _factor )


kl_distance = KL_distance(real_probs,expected_probs)

print("KL ditance is ", kl_distance, " and the reverse KL-distnce is ", KL_distance(expected_probs, real_probs));

#print("Fake Tests");
#print("FAKE Z Test");
#the_Z_test(mean_random_variable(real_probs), variance_random_variable(real_probs), _factor,
#          mean_random_variable(fake_probs), variance_random_variable(fake_probs), _factor )


print('************************************************************************************');
print('************************************************************************************');

##minka by LM  experiments
print('Minka by LM experiments (Pure-Python version):');
wrapped = 0
alpha, psi = minka_LM_experiment_from_file(dataset_file, ",",prec,tolerance, _maxiter, wrapped )#minka_LM_experiment_from_file("test_fix_counts.csv", ",",prec,tolerance, _maxiter )

'''
K = len(alpha)#.shape

real_probs = find_probs(all_dataset_file, ",")
expected_probs = alpha*psi


real_probs_wider = np.zeros(K, np.longdouble)
alpha_wider = np.zeros(K, np.longdouble)
for k in range(K):
    real_probs_wider[k] = real_probs[k]
    alpha_wider[k] = alpha[k]




'''
K = len(alpha)#.shape
#D, N = load_dataset_from_file("all_fix_counts.csv", delim=',')


real_probs = find_probs(all_dataset_file, ",")#find_probs("test_fix_counts.csv", ",") #


#print("real_probs = ", real_probs);
expected_probs = alpha*psi
#print("expected_probs = ", expected_probs);




print("tvd = ",tvd(real_probs, expected_probs));

fake_probs = np.array([0.505, 0.375, 0.001, 0.001, 0.118])   
#print("fake probs = ",fake_probs)

real_probs_wider = np.zeros(K, np.longdouble)
alpha_wider = np.zeros(K, np.longdouble)
for k in range(K):
    real_probs_wider[k] = real_probs[k]
    alpha_wider[k] = alpha[k]




diff_probs =  real_probs - expected_probs
print("Euclidean =  ", LA.norm(diff_probs));
print("MSE = ", _mean_square_error(real_probs,expected_probs ));


real_counts = real_probs* _factor
expected_counts = expected_probs* _factor
fake_counts = fake_probs* _factor



real_check = check_for_5(real_counts)
expected_check = check_for_5(expected_counts)
fake_check = check_for_5(fake_counts)

if real_check == 0 and expected_check == 0:
    chi_square_test(real_counts, expected_counts, K-1, conf)
else:
    print("sample size less than 5 in at least one category, cannot perform chi square test");

#print("!!FAKE Chi-Square TEST!!")
#if real_check == 0 and fake_check == 0:
#    chi_square_test(real_counts, fake_counts, K-1, conf)
#else:
#    print("!!FAKE Chi-Square !!: sample size less than 5 in at least one category, cannot perform chi square test");

diff_mean = np.float64(0)
diff_variance = np.float64(0)

diff_mean = abs(mean_random_variable(real_probs)- mean_random_variable(expected_probs))
diff_variance = abs(variance_random_variable(real_probs)- variance_random_variable(expected_probs))



the_Z_test(mean_random_variable(real_probs), variance_random_variable(real_probs), _factor,
           mean_random_variable(expected_probs), variance_random_variable(expected_probs), _factor )


kl_distance = KL_distance(real_probs,expected_probs)

print("KL ditance is ", kl_distance, " and the reverse KL-distnce is ", KL_distance(expected_probs, real_probs));

#print("Fake Tests");
#print("FAKE Z Test");
#the_Z_test(mean_random_variable(real_probs), variance_random_variable(real_probs), _factor,
#          mean_random_variable(fake_probs), variance_random_variable(fake_probs), _factor )


print('************************************************************************************');
print('************************************************************************************');
'''
##minka by YS experiments

print('Minka by YS experiments');

YS_fails = np.zeros(2, dtype=np.int32)

YS_fails[0] = 0
alpha, psi = minka_YS_experiment_from_file(dataset_file, ",",prec,tolerance, _maxiter, _YSm,_delta, YS_fails )

real_probs = find_probs(all_dataset_file, ",")
expected_probs = alpha*psi

print("YS_fails = ", YS_fails[0])

print('**************************************************')	
print('**** END experiments')
print('**************************************************')	

# end verification


'''
'''
****************************
RESULTS PRINTOUT:
****************************

see the files:

1) python3 LogL-global-v5.py 38 3 >exec_experiments-3digits.txt

exec_experiments-3digits.txt

2) python3 LogL-global-v5.py 38 6 >exec_experiments-6digits.txt
exec_experiments-6digits.txt

3) python3 LogL-global-v5.py 38 9 >exec_experiments-9digits.txt
exec_experiments-9digits.txt
'''
