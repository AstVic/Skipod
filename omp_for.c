#include <math.h>
#include <stdlib.h>
#include <stdio.h>
#include <omp.h>

#define Max(a,b) ((a)>(b)?(a):(b))
#define NMAX 5000

double maxeps = 1e-7;
int itmax = 10000;
double eps;
int i,j;
int N;

double A[NMAX][NMAX];

int arrayN[] = {66, 130, 258, 514, 1026, 2050};
int nmax = 6;

void relax();
void init();
void verify();
void myfunc();

int main(int argc, char **argv) {
    for (int index = 0; index < nmax; index++) {
        N = arrayN[index];
        myfunc();
    }
    return 0;
}

void myfunc()
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
    #pragma omp parallel for
    for (int i = 0; i < N; i++)
    {
        A[i][0] = 0.0;
        A[i][N-1] = 0.0;
    }
    
    #pragma omp parallel for
    for (int j = 0; j < N; j++)
    {
        A[0][j] = 0.0;
        A[N-1][j] = 0.0;
    }

    #pragma omp parallel for collapse(2)
    for (int i = 1; i < N-1; i++)
        for (int j = 1; j < N-1; j++)
            A[i][j] = 2.0 + i + j;
}

void relax()
{   
    #pragma omp parallel  private(i, j)  reduction(max:eps)
    {
        #pragma omp for
        for(i=1; i<=N-2; i++)
        for(j=1 + i%2; j<=N-2; j += 2)
        {
            double old = A[i][j];
            double newv = (2.0*A[i-1][j] + A[i+1][j] + 2.0*A[i][j-1] + A[i][j+1]) / 6.;
            A[i][j] = newv;
            double diff = fabs(newv - old);
            if (diff > eps) eps = diff;
        }
    }
    #pragma omp parallel private(i, j) reduction(max:eps)
    {
        #pragma omp for
        for(i=1; i<=N-2; i++)
        for(j=1+(i+1)%2; j<=N-2; j += 2)
        {
            double old = A[i][j];
            double newv = (2.0*A[i-1][j] + A[i+1][j] + 2.0*A[i][j-1] + A[i][j+1]) / 6.;
            A[i][j] = newv;
            double diff = fabs(newv - old);
            if (diff > eps) eps = diff;
        }
    }
}

void verify()
{
    double s = 0.0;
    double nn = 1.0 / (N * N); 

    #pragma omp parallel for reduction(+:s)
    for (int i = 0; i < N; i++)
    {
        double fi = (i + 1) * nn; 
        double row_sum = 0.0;
        for (int j = 0; j < N; j++)
        {
            row_sum += A[i][j] * fi * (j + 1);
        }
        s += row_sum;
    }
    
    printf("S = %f, ", s);
}