#include <math.h>
#include <stdlib.h>
#include <stdio.h>
#include <omp.h>

#define Max(a,b) ((a)>(b)?(a):(b))
#define NMAX 5000

#ifndef N
#define N 66
#endif

double maxeps = 1e-7;
int itmax = 10000;
double eps;
int i,j;

double A[NMAX][NMAX];

//int arrayN[] = {66, 130, 258, 514, 1026, 2050};
// int nmax = 6;

void relax();
void init();
void verify();

int main(int argc, char **argv)
{
    double t_start = omp_get_wtime();

    init();

    int it;
    for (it = 1; it <= itmax; it++)
    {
        eps = 0.0;
        relax();
        //printf("it=%4i   eps=%f\n", it, eps);
        if (eps < maxeps) break;
    }

    verify();

    double t_end = omp_get_wtime();
    printf("Iteartion = %d, N = %d, Time = %f seconds\n", it, N, t_end - t_start);
}

void init()
{ 

	for(j=0; j<=N-1; j++)
	for(i=0; i<=N-1; i++)
	{
		if(i==0 || i==N-1 || j==0 || j==N-1)
		A[i][j]= 0.;
		else A[i][j]= ( 2 + i + j ) ;
	}
} 


void relax()
{

	for(j=1; j<=N-2; j++)
	for(i=1; i<=N-2; i++)
	{ 
		double e;
		e=A[i][j];
		A[i][j]=(2*A[i-1][j]+A[i+1][j]+2*A[i][j-1]+A[i][j+1])/6.;
		eps=Max(eps, fabs(e-A[i][j]));
	}    
}


void verify()
{ 
	double s;

	s=0.;
	for(j=0; j<=N-1; j++)
	for(i=0; i<=N-1; i++)
	{
		s=s+A[i][j]*(i+1)*(j+1)/(N*N);
	}
	printf("S = %f, ", s);

}