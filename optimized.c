#include <math.h>
#include <stdlib.h>
#include <stdio.h>
#include <omp.h>

#define Max(a,b) ((a)>(b)?(a):(b))
#define N (2*2*2*2*2*2+2)

double maxeps = 0.1e-7;
int itmax = 100;
double eps;

double A[N][N];

void relax();
void init();
void verify(); 

int main(int an, char **as)
{
    double t_start = omp_get_wtime();

    int it;

    init();

    for (it = 1; it <= itmax; it++)
    {
        eps = 0.;
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
    // cache-friendly порядок: i внешний, j внутренний
    for (int i = 0; i < N; i++)
    for (int j = 0; j < N; j++)
    {
        if (i == 0 || i == N-1 || j == 0 || j == N-1)
            A[i][j] = 0.;
        else
            A[i][j] = 2 + i + j;
    }
}


void relax()
{
    const double inv6 = 1.0 / 6.0;  // избегаем деления в цикле

    // оптимальный порядок обхода
    for (int i = 1; i < N-1; i++)
    for (int j = 1; j < N-1; j++)
    {
        double old = A[i][j];

        double newv =
            ( 2.0 * A[i-1][j]
            +       A[i+1][j]
            + 2.0 * A[i][j-1]
            +       A[i][j+1]
            ) * inv6;

        A[i][j] = newv;

        double diff = fabs(newv - old);
        if (diff > eps) eps = diff;
    }
}


// void verify()
// {
//     double s = 0.0;

//     for (int i = 0; i < N; i++)
//     for (int j = 0; j < N; j++)
//         s += A[i][j] * (i+1) * (j+1) / (double)(N*N);

//     printf("  S = %f\n", s);
// }

void verify()
{
    double s = 0.0;
    const double invNN = 1.0 / (double)(N * N);

    for (int i = 0; i < N; i++)
    {
        double fi = (double)(i + 1);
        double si = 0.0;  // локальная сумма строки

        for (int j = 0; j < N; j++)
        {
            si += A[i][j] * fi * (double)(j + 1);
        }

        s += si;
    }

    s *= invNN;

    printf("  S = %f\n", s);
}
