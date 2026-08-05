#include <stdio.h>
#define _USE_MATH_DEFINES
#include <math.h>
#include <stdlib.h>
#include "constants_hor_shift.h"
#include "LM_hor_shift.h"
#include "init_hor_shift.h"

/**********  aux function: pow_longer of a double with an int exponent; iterative   **********/

double pow_long(double b, long a)
{
  double retval = 1.0;
  
  if(a == 0)
    return retval;
	
  int flag = 0;
  if ( a < 0 )
    {	a = -a;
    	flag = 1;
    }
    	
    while ( a > 0 )
    {
        if ( a & 1 ) // execute when the least significant bit is 1
            retval *= b;
 
        b *= b;
        a >>= 1 ; // divide by 2 and take the remainder (shift 1 bit on the right)
    }    

  if ( flag )
    retval = 1.0/retval;
    
  return retval;
}



/**********   OUR WORK   **********/

// main logL-LM function  
double main_LM_logL(int PREC)
{
	//initialization
	double pown = 0;
	double toterr;	
	long maxprec;
	
	init_dataset(pown);
	printf("init dataset data: done\n");
		
	// we divide by (K+1) since we want that the total error has prec decimal digits correct
	double LM_accuracy = 0;
	LM_accuracy = pow(10, -(PREC+2));
	LM_accuracy = LM_accuracy / (K + 1.0) ;

	double res_logL = 0;
	int hor_shift_logL = 0;
	long acc_logL = 0;
	double toterr_logL = 0; 			
	long opt_m_logL = 0;
	// reads the needed bernoulli numbers
	initBern_logL();
	// calls the error/hor_shift eval function
	opterrNoPrint_logL(&maxprec, &toterr, pown);
	
	opt_m_logL = vect_logL[0];
	hor_shift_logL = vect_logL[1];
	acc_logL = maxprec;
	toterr_logL = toterr;
	
	printf("opt_m_logL = %ld, horizontal shift logL = %d,  toterr_logL (EM formula) = %E, maxprec_logL (EM formula) = %ld\n", opt_m_logL, hor_shift_logL, (K+1)*toterr_logL, acc_logL);
	if (opt_m_logL == 0 )
		{res_logL = 0;}	// error code for the Euler-Maclaurin formula not accurate enough
	else
		{// calls the logL-eval function that uses the LM-technique
		res_logL = logL_diretta(opt_m_logL, hor_shift_logL);
		}
	return res_logL;
}	

// computation of logL 
double LM_logL(double x, double y, long m, int hor_shift)
{
	double retval = 0;
	long i = 0;
	long j = 0;
	
	/* inderted by AL 23/04/2023 */
	/* special values */
	if (y == 1)
		{ 
		retval = -log(x);
		return retval;
		}
	if (y == 2)
		{ 
		retval = - 2.0 * log(x) + log(1.0 + x);
		return retval;
		}
		
	/* handling the horizontal shift contribution; see LM-paper, eq. (12)	*/
	double hor_shift_contrib = 0;
	if (hor_shift != 0) 
		{
		double oneoverx = 1.0 / x;
		/* computing the horizontal shift contribution */
		for(j = 0 ; j < hor_shift; j++)
			{
			hor_shift_contrib += log(oneoverx + j); 	
			}
		/* performing the horizontal shift */
		y = y - hor_shift;
		x = x / ( 1.0 + hor_shift * x);
		}
	
	double d = (1.0) / (1.0 + x * (y - 1.0) );

	double	stepx = x * x;
	double	stepd = d * d;
	double	startx = x;
	double	startd = d;
		
	double	fattorex = startx;
	double	fattored = startd;
	
	retval = -y * log(x) - (y - 1.0) - (1.0/x + y - 0.5) * log(d);
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

//double logL_diretta(double m, double N, double psi, double logprobs, psioverprobs, X, K)
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

double logL(long m, int hor_shift)
{
	return 	logL_diretta(m, hor_shift);
}

// funzioni per gli errori 
double g_logL(double x, double y, long m)
{
	double retval = 0.0;
	if(y>0 && y<3)
	{
		return 0.0;
	}
	// coeff m of the vec_errcoeff is stored in position m-1	

	long expo = 2*m + 1;
	//printf ("expo = %ld\n", expo); 
	double d = (1.0) / (1.0 + x * (y - 1.0) );	
	retval = vec_errcoeff[m-1] * pow_long(x, expo) * (1.0 - pow_long( d, expo) );
	//retval = vec_errcoeff[m-1] * pow_long(x, (2*m+1)) * (1.0-1.0/pow_long( (1.0+x*(y-1.0)), (2*m+1)) );
	//printf ("vec_errcoeff[m-1] = %32.30f\n", vec_errcoeff[m-1]); 
	//printf ("x = %lf\n", x); 
	//printf ("y = %lf\n", y); 
	//printf ("d = %32.30ff\n", d); 	
	//printf ("x^(2m+1) = %32.30ff\n", pow_long(x, expo)); 
	//printf ("d-factor = %32.30ff\n", (1.0 - pow_long( d, expo) )); 			
	retval = fabs(retval);

	return retval;
}

double err_logL(long m, int j)
{

	/* for the horizontal shift */
	double aux1 = psi / (1.0 + j * psi);
	double aux2 = N - j;
	double aux = 0;
	
	/*double retval = g(psi, N, m);*/
	double retval = g_logL(aux1, aux2, m); 
	
	long i = 0;
	for(i = 0; i < K; i++) 
	{
		/* for the horizontal shift */
		aux = psi / probs[i];
		aux1 =  aux / ( 1.0 + j * aux);
		aux2 =  X[i] - j;	
		/*retval += g(psi/probs[i], X[i], m);*/
		retval += g_logL(aux1 , aux2 , m);
	}
	//printf("err: retval = %f\n", retval);
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
	
	while (ok == 0 && j < max_hor_shift) {
		//printf("j = %d\n", j);				
		m = 1;
		//printf("m = %ld\n", m);		
		*toterr = err_logL( m , j );
		//printf("minimal error logL = %32.30f\n", (K+1.0)* (toterr));		
		m += 1;  	
		toterr1 = err_logL( m, j );

		while ( toterr1 < *toterr && *toterr > LM_accuracy && m < 100)
			{
			//printf("m = %ld\n", m);					
			*toterr = toterr1;
			//printf("minimal error logL = %32.30f\n", (K+1.0)* (toterr));		
			m += 1;  	
			toterr1 = err_logL( m, j );
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
		printf("Horizontal shift too large (>1000) for the required accuracy; ask for a smaller accuracy");		
		vect_logL[0] = 0;
		vect_logL[1] = 0;
		return vect_logL;		
		}
			
	//printf("minimal error = %32.30f\n", (K+1.0)* (*toterr));		
	*maxprec = floor(fabs(log10((K+1.0) * (*toterr)))) -1;
	//printf("maxprec logL = %ld\n", *maxprec);		
	//printf("m_opt = %ld\n", m_opt);					
	//printf("hor_shift = %d\n", hor_shift);	
	
	vect_logL[0] = m_opt;
	vect_logL[1] = hor_shift;
	return vect_logL;
}

 
