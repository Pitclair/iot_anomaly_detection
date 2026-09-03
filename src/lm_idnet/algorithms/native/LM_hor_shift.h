double logL_diretta(long m, int hor_shift);

double LM_logL(double x, double y, long m, int hor_shift);

long* opterrNoPrint_logL(long *maxprec, double* toterr, double n);

double loggamma_LM( double probabilities[],  double psi_0);

double loggamma_LM_matrix(
	int precision_digits,
	const double *counts,
	long rows,
	int categories,
	const double *probabilities,
	double psi);

int initBern_logL_paths(const char *bernoulli_path, const char *error_path);

long init_params(int PREC, double counts[], int categories );

double logL_diretta_global();
