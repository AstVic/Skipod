#include <math.h>
#include <stdlib.h>
#include <stdio.h>
#include <omp.h>

#define Max(a,b) ((a)>(b)?(a):(b))
#define N (2*2*2*2*2*2*2*2*2*2+2)

double maxeps = 1e-7;
int itmax = 100;
double eps;

double A[N][N];

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
        printf("it=%4i   eps=%f\n", it, eps);
        if (eps < maxeps) break;
    }

    verify();

    double t_end = omp_get_wtime();
    printf("Time = %f seconds\n", t_end - t_start);

    return 0;
}

void init()
{
    // Инициализация границ
    for (int i = 0; i < N; i++)
    {
        A[i][0] = 0.0;
        A[i][N-1] = 0.0;
    }
    for (int j = 0; j < N; j++)
    {
        A[0][j] = 0.0;
        A[N-1][j] = 0.0;
    }

    // Инициализация внутренних ячеек
    for (int i = 1; i < N-1; i++)
        for (int j = 1; j < N-1; j++)
            A[i][j] = 2.0 + i + j;
}

void relax()
{
    const double inv6 = 1.0 / 6.0;

    for (int i = 1; i < N-1; i++)
        for (int j = 1; j < N-1; j++)
        {
            double old = A[i][j];
            double newv = (2.0*A[i-1][j] + A[i+1][j] + 2.0*A[i][j-1] + A[i][j+1]) * inv6;
            A[i][j] = newv;
            double diff = fabs(newv - old);
            if (diff > eps) eps = diff;
        }
}

void verify()
{
    double s = 0.0;

    for (int i = 0; i < N; i++)
    {
        double fi = (double)(i+1);
        for (int j = 0; j < N; j++)
            s += A[i][j] * fi * (double)(j+1);
    }

    s /= (double)(N * N);
    printf("  S = %f\n", s);
}
