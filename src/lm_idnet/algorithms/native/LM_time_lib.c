#define _USE_MATH_DEFINES
#include <stdio.h>
#include <sys/time.h>
#include <stdlib.h>
#include <math.h>
#include <unistd.h>
#include <errno.h>
#include <string.h>
#include "constants_hor_shift.h"
#include "LM_hor_shift.h"
#include "init_hor_shift.h"

//aux functions by Sherenaz to make Minka's fixed-point iteration easier to code
//uses freshpsioverprobs to update the contents of psioverprobs vector
//both vectors are of length K, obviously
void update_psioverprobs(double freshpsioverprobs[])
{
	for(int i = 0; i < K ; i++)
		psioverprobs[i] = freshpsioverprobs[i];
}

//mock input to fool optimizer
void pippero(double *a, long num)
{
	FILE *f = fopen("gaucho.txt", "r");
	long i;
	for(i=0;i<K;i++)
	{
		fscanf(f, "%lf", &(a[i]));
	}
	fclose(f);
}

/**********	 OUR WORK	 **********/

//an init function by Sherenaz to make Minka's fixed-point iteration easier to code
//receives a vector of counts and length sent from the Python program to initialize X and K
//both vectors are of length K, obviously
void initExp100Dataset(int counts[], int categories, double probabilities[], double psi_0)
{
	SIGDIG = 15; // meaningful digits for the double type
	//SIGDIG = 19; // meaningful digits for the longdouble type

	//printf("Experiment 100: some Minka dataset\n");

	K = categories;
	psi = psi_0;

	long i;

	probs = (double*)malloc(K*sizeof(double));
 	psioverprobs = (double*)malloc(K*sizeof(double));
 	X = (double*)malloc(K*sizeof(double));
 	logprobs = (double*)malloc(K*sizeof(double));

	for(i = 0; i < K ; i++){
		probs[i] = probabilities[i];
		psioverprobs[i] = psi/probs[i];
		logprobs[i] = log(probs[i]);
		X[i] = counts[i];

	}

}

void initDatasetFromFile(const char *filename)
{
	FILE *in = NULL;
	if( (in = fopen(filename, "r")) == NULL)
	{
		perror("cannot open file, ABORT");
		exit(1);
	}

	SIGDIG = 15; // meaningful digits for the double type
	//SIGDIG = 19; // meaningful digits for the longdouble type

	//read K
	if(fscanf(in, "%ld", &K) < 1)
	{
		perror("Error reading K: ABORT");
		exit(1);
	}

	//read psi
	if(fscanf(in, "%lf", &psi) < 1)
	{
		perror("Error reading psi: ABORT");
		exit(1);
	}

	probs = (double*)malloc(K*sizeof(double));
 	psioverprobs = (double*)malloc(K*sizeof(double));
 	X = (double*)malloc(K*sizeof(double));
 	logprobs = (double*)malloc(K*sizeof(double));

	//read probs
	long i = 0;
	for(i=0;i<K;i++)
	{
		if(fscanf(in, "%lf", &(probs[i])) < 1)
		{
			perror("Error reading probs: ABORT");
			exit(1);
		}
	}

	//read X
	for(i=0;i<K;i++)
	{
		if(fscanf(in, "%lf", &(X[i])) < 1)
		{
			perror("Error reading probs: ABORT");
			exit(1);
		}
	}

	for(i=0;i<K;i++)
	{
		psioverprobs[i] = psi/probs[i];
		logprobs[i] = log(probs[i]);
	}

	fclose(in);
}



void initBern()
{
	FILE* f1 = fopen("bernreal-100.txt", "r");
	FILE* f2 = fopen("bernreal-norm-100.txt", "r");
	FILE* f3 = fopen("bernreal-norm-100-digamma.txt", "r");
	FILE* f4 = fopen("err_coeff-100.txt", "r");
	FILE* f5 = fopen("err_coeff-100_digamma.txt", "r");
	FILE* f6 = fopen("err_coeff-100_trigamma.txt", "r");
	int i = 0;
	vec_evenbernoulli = (double*)malloc(100*sizeof(double));
	vec_evenbernoullinorm = (double*)malloc(100*sizeof(double));
	vec_evenbernoullinormdigamma = (double*)malloc(100*sizeof(double));
	vec_errcoeff = (double*)malloc(100*sizeof(double));
	vec_errcoeff_digamma = (double*)malloc(100*sizeof(double));
	vec_errcoeff_trigamma = (double*)malloc(100*sizeof(double));

	for(i=0;i<100;i++)
	{
		fscanf(f1, "%lf", &(vec_evenbernoulli[i]));
		fscanf(f2, "%lf", &(vec_evenbernoullinorm[i]));
		fscanf(f3, "%lf", &(vec_evenbernoullinormdigamma[i]));
		fscanf(f4, "%lf", &(vec_errcoeff[i]));
		fscanf(f5, "%lf", &(vec_errcoeff_digamma[i]));
		fscanf(f6, "%lf", &(vec_errcoeff_trigamma[i]));
	}

	fclose(f1);
	fclose(f2);
	fclose(f3);
	fclose(f4);
	fclose(f5);
	fclose(f6);

	for(i=0;i<2;i++)
	{
		vect_logL[i] = 0;
		vect_digamma[i] = 0;
		vect_trigamma[i] = 0;
	}
}

static void freeBern_logL()
{
	free(vec_evenbernoullinorm);
	free(vec_errcoeff);
	vec_evenbernoullinorm = NULL;
	vec_errcoeff = NULL;
}

int initBern_logL_paths(const char *bernoulli_path, const char *error_path)
    {
    if (bernoulli_path == NULL || error_path == NULL)
        return 1;

    FILE* f2 = fopen(bernoulli_path, "r");
	FILE* f4 = fopen(error_path, "r");
	if (f2 == NULL || f4 == NULL)
		{
		if (f2 != NULL) fclose(f2);
		if (f4 != NULL) fclose(f4);
		return 1;
		}
	double* bernoulli = (double*)malloc(100*sizeof(double));
	double* errors = (double*)malloc(100*sizeof(double));
	if (bernoulli == NULL || errors == NULL)
		{
		free(bernoulli);
		free(errors);
		fclose(f2);
		fclose(f4);
		return 2;
		}
	for(int i=0;i<100;i++)
		{
		if (fscanf(f2, "%lf", &(bernoulli[i])) != 1 ||
			fscanf(f4, "%lf", &(errors[i])) != 1)
			{
			free(bernoulli);
			free(errors);
			fclose(f2);
			fclose(f4);
			return 3;
			}
		}

	fclose(f2);
	fclose(f4);
	freeBern_logL();
	vec_evenbernoullinorm = bernoulli;
	vec_errcoeff = errors;

	for(int i=0;i<2;i++)
	{
		vect_logL[i] = 0;
	}
	return 0;
}

void initBern_logL()
{
	if (initBern_logL_paths("bernreal-norm-100.txt", "err_coeff-100.txt") != 0)
		fprintf(stderr, "Cannot initialize LM coefficient tables\n");
}





// computation of logL
double LM_logL(double x, double y, long m, int hor_shift)
{
	double retval = 0;
	long i = 0;
	long j = 0;

	/* inderted by AL 23/04/2023 */
	/* special values */
	if (y == 0)
		{
		return 0.0;
		}
	if (y == 1)
		{
		retval = -log(x);
		return retval;
		}
	if (y == 2)
		{
		/*retval = - 2.0 * log(x) + log(1.0 + x);*/
		retval = log((1.0 + x)/(x*x));
		return retval;
		}

	/* handling the horizontal shift contribution; see LM-paper, eq. (12)	*/

	double hor_shift_contrib = 0;
	if (hor_shift != 0)
		{
		double oneoverx = 1.0 / x;
		hor_shift_contrib = log(oneoverx);
		/* computing the horizontal shift contribution */
		for(j = 1 ; j < hor_shift; j++)
			{
			hor_shift_contrib += log(oneoverx + j);
			}
		/* performing the horizontal shift */
		y = y - hor_shift;
		/*x = x / ( 1.0 + hor_shift * x);*/
		x = 1.0 / ( oneoverx + hor_shift );
		}
	/* now x and y are NEW: contain the horizontal shift corrections */
	double yminusone = y - 1.0;
	double d = 1.0 / (1.0 + x * yminusone );

	double	stepx = x * x;
	double	stepd = d * d;
	//double	startx = x;
	//double	startd = d;

	double	fattorex = x;//startx;
	double	fattored = d;//startd;

	retval = -y * log(x) - yminusone - (1.0/x + y - 0.5) * log(d);
	retval += vec_evenbernoullinorm[0] * fattorex * ( - 1.0 + fattored );

	for(i = 1; i < m; i++)
		{
			fattorex = fattorex * stepx;
			fattored = fattored * stepd;
			retval += vec_evenbernoullinorm[i] * fattorex * ( - 1.0 + fattored );
		}

		/* adding the horizontal shift contribution */
	retval = retval + hor_shift_contrib;
	return retval;
}

double logL_diretta(long m, int hor_shift)
{
	//printf("N = %f\n", N);
	double retval = -LM_logL(psi, N, m, hor_shift);
	long i = 0;
	for ( i = 0; i < K; i++)
	{
		retval += LM_logL(psioverprobs[i], X[i], m, hor_shift);
	}
	return retval;
}
//by Sherenaz, utilizes m_gloabl and hor_shift_gloabl
double logL_diretta_global()
{
	//printf("N = %f\n", N);
	double retval = -LM_logL(psi, N, m_global, hor_shift_global);
	long i = 0;
	for ( i = 0; i < K; i++)
	{
		retval += LM_logL(psioverprobs[i], X[i], m_global, hor_shift_global);
	}
	return retval;
}


long* opterrNoPrint_logL(long *maxprec, double *toterr, double n)
{
	long m = 0;
	long i = 0;
	double toterr1;
	int hor_shift = 0;
	int max_hor_shift = 1000;
	int ok = 0;
	int j = 0;
	long m_opt = 0;
	long expo = 0;
	double coeff = 0;
	double aux_psi = 0;
	double aux_psioverprobs = 0;
	double psi_pow = 0;
	double step_psi = 0;
	double* workspace = (double*)malloc(2*K*sizeof(double));
	if (workspace == NULL)
		return NULL;
	double* psioverprobs_pow = workspace;
	double* step_psioverprobs = workspace + K;
	long counter = 0;

	while (ok == 0 && j < max_hor_shift)
	{
		// *************	begin comput. error for m=1
		counter += 1;
		//printf("j = %d\n", j);
		m = 1;
		//expo = 2*m + 1; // logL starts with 2*m+1
		expo = 3;
		// coeff m of the vec_errcoeff is stored in position m-1
		coeff = vec_errcoeff[m-1];
		// for the horizontal shift; first category
		aux_psi = psi / (1.0 + j * psi);
		// step for the repeated product strategy for the next values
		step_psi = aux_psi * aux_psi;
		//psi_pow = pow(aux_psi, expo);
		// start for the repeated product strategy for the next values
		psi_pow = step_psi * aux_psi;
		// contribution of the first term
		*toterr = coeff * psi_pow;
		for(i = 0; i < K; i++)
		{
			/* for the horizontal shift */
			//aux= psi / probs[i];
 			//aux1 =	aux / ( 1.0 + j * aux);
			//aux2 =	X[i] - j;
			//retval += g_logL(aux1 , aux2 , m);
			// for the horizontal shift; other categories
			aux_psioverprobs = psioverprobs[i] / ( 1.0 + j * psioverprobs[i]);
			// step for the repeated product strategy
			step_psioverprobs[i] = aux_psioverprobs * aux_psioverprobs;
			//psioverprobs_pow[i] = pow(aux_psioverprobs, expo);
			// start for the repeated product strategy
			psioverprobs_pow[i] = step_psioverprobs[i] * aux_psioverprobs;
			// contribution of the subsequent terms
			*toterr += coeff * psioverprobs_pow[i];
		}

		// ************* end comput. error for m=1

		//*************	begin comput. error for m=2
		//printf("m = %ld\n", m);
		//*toterr = err_logL( m , j );
		//printf("minimal error logL = %32.30f\n", (K+1.0)* (toterr));
		m += 1;
		//expo = 2*m + 1;
		//expo += 2;
		//coeff m of the vec_errcoeff is stored in position m-1
		coeff = vec_errcoeff[m-1];
		psi_pow *= step_psi;
		toterr1 = coeff * psi_pow;
		for(i = 0; i < K; i++)
		{
			/* for the horizontal shift */
			psioverprobs_pow[i] *= step_psioverprobs[i];
			toterr1 += coeff * psioverprobs_pow[i];
		}
		//toterr1 = err_logL( m, j );

		// ************* end comput. error for m=2


		while ( toterr1 < *toterr && *toterr > LM_accuracy && m < 100)
		{
			//printf("m = %ld\n", m);
			*toterr = toterr1;
			//printf("minimal error logL = %32.30f\n", (K+1.0)* (toterr));
			// *************	begin comput. error for the next m
			m += 1;
			//expo = 2*m + 1;
			//expo += 2;
			// coeff m of the vec_errcoeff is stored in position m-1
			coeff = vec_errcoeff[m-1];
			psi_pow *= step_psi;
			toterr1 = coeff * psi_pow;
			for(i = 0; i < K; i++)
			{
				/* for the horizontal shift */
				psioverprobs_pow[i] *= step_psioverprobs[i];
				toterr1 += coeff * psioverprobs_pow[i];
			}
			counter += 1;
			//toterr1 = err_logL( m, j );
			// ************** end comput. error for the next m
		}

		if (*toterr < LM_accuracy)
		{
			ok = 1;
			hor_shift = j; /* horizontal shift for this accuracy */
			if (m <= 100)
			{
				m_opt = m-1; /* optimal m for this accuracy*/
			}
			else
			{
				m_opt = 100; /* optimal m for this accuracy*/
			}
		}
		j += 1;
	}
	if (ok == 0)
	{
		printf("Horizontal shift [logL] too large (>1000) for the required accuracy; ask for a smaller accuracy");
		free(workspace);
		return NULL;
	}

	//printf("minimal error = %32.30f\n", (K+1.0)* (*toterr));
	*maxprec = floor(fabs(log10((K+1.0) * (*toterr)))) -1;
	//printf("maxprec logL = %ld\n", *maxprec);
	//printf("m_opt = %ld\n", m_opt);
	//printf("hor_shift = %d\n", hor_shift);

	//printf("total number of iterations (err_logL) = %ld\n", counter);
	vect_logL[0] = m_opt;
	vect_logL[1] = hor_shift;
	free(workspace);
	return vect_logL;
}


static void free_params()
{
	free(X);
	free(probs);
	free(psioverprobs);
	free(logprobs);
	X = NULL;
	probs = NULL;
	psioverprobs = NULL;
	logprobs = NULL;
}

/*BY Sherenaz*/
long init_params(int PREC, double counts[], int categories ){
	//initializes: K(number of categories) , X (vector of counts), and (N) total counts , PRECISION(equal to prec, by Sherenaz), LM_accuracy, SIGDIG
	//allocates memory for X, probs, psioverprobs, and logprobs

    if (counts == NULL || categories <= 0 || PREC <= 0)
        return -1;

    long i = 0;
	K = categories;
	N = 0;
	long NN=0;
	PRECISION = PREC;
	SIGDIG = 15; // meaningful digits for the double type
	//SIGDIG = 19; // mea ningful digits for the longdouble type

	// we divide by (K+1) since we want that the total error has prec decimal digits correct
	LM_accuracy = pow(10, -(PREC+2));
	LM_accuracy = LM_accuracy / (K + 1.0) ;

	//allocate vector X, and other vectors
	free_params();
	X = (double*)malloc(K*sizeof(double));
 	probs = (double*)malloc(K*sizeof(double));
 	psioverprobs = (double*)malloc(K*sizeof(double));
 	logprobs = (double*)malloc(K*sizeof(double));
	if (X == NULL || probs == NULL || psioverprobs == NULL || logprobs == NULL)
	{
		free_params();
		return -1;
	}

	//fillup only vector X, and integral N
	for(i = 0; i < K ; i++){
		X[i] = counts[i];
		N += X[i];
		//NN+=X[i];
	}
	return NN;
}
/*BY Sherenaz*/
double loggamma_LM( double probabilities[],  double psi_0)
	//to refresh probs and psi before each call to loggamma_LM
{
	double pown = 0;
	long i = 0;
 	//refresh psi, probs, and psioverprobs
	psi = psi_0;
	for(i = 0; i < K ; i++){
		//probs[i] = probabilities[i];
		psioverprobs[i] = psi/probabilities[i];
		logprobs[i] = log(probabilities[i]);

	}


	//reset asymp
	asymp = 0;

	for(i=0;i<K;i++)
	{
		//X[i] = pown;
		//	printf("X[%ld] = %f\n", i, X[i]);

		asymp += logprobs[i] * X[i];
	}

	//must be done before every call to loggamma_LM
	double bound = SIGDIG - PRECISION -2 ;
	if ( log10(fabs(asymp)) > bound)
	{
		printf("N = %f\nK = %ld\n",	N, K);
 		printf("logL of this case is asymptotic to (psi -> 0+) = %32.30f\n", asymp);
	 	fprintf(stderr, "ERROR: LogL too large to assure the desired precision in double; switch to multiprecision\n");
		printf("***** END PROGRAM *****\n");
		free_params();
		return 99;
		//exit(1);
	}

	double toterr;
	long maxprec;
	int hor_shift_logL = 0;
 	long acc_logL = 0;
 	double toterr_logL = 0;
 	long opt_m_logL = 0;
	double res_logL = 0;

	if (opterrNoPrint_logL(&maxprec, &toterr, pown) == NULL)
	{
		free_params();
		return NAN;
	}

	//opt_m_logL = vect_logL[0];
	//hor_shift_logL = vect_logL[1];

	acc_logL = maxprec;
	toterr_logL = toterr;

    //by Sherenaz
	m_global = vect_logL[0];
	hor_shift_global = vect_logL[1];

	res_logL = logL_diretta_global();//opt_m_logL, hor_shift_logL);
	return res_logL;
}
