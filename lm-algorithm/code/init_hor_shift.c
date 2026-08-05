#include <math.h>
#include <stdlib.h>
#include <stdio.h>
#include "constants_hor_shift.h"


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
		{	vect_logL[i] = 0;
			vect_digamma[i] = 0;
			vect_trigamma[i] = 0;
		}
	            
}



void initBern_logL()
{

        FILE* f2 = fopen("bernreal-norm-100.txt", "r");
        FILE* f4 = fopen("err_coeff-100.txt", "r");
        int i = 0;
        vec_evenbernoullinorm = (double*)malloc(100*sizeof(double));
        vec_errcoeff = (double*)malloc(100*sizeof(double));
        
        for(i=0;i<100;i++)
        {
                fscanf(f2, "%lf", &(vec_evenbernoullinorm[i]));
                fscanf(f4, "%lf", &(vec_errcoeff[i]));
        }

        fclose(f2);
        fclose(f4);
        
        for(i=0;i<2;i++)
		{	vect_logL[i] = 0;
		}
	            
}


