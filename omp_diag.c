#include <math.h>
#include <stdlib.h>
#include <stdio.h>
#include <omp.h>

#define Max(a,b) ((a)>(b)?(a):(b))
#define N (2*2*2*2*2*2*2*2*2*2*2*2*2+2)

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
    // Распараллеливаем инициализацию границ
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

    // Распараллеливаем инициализацию внутренних ячеек
    #pragma omp parallel for collapse(2)
    for (int i = 1; i < N-1; i++)
        for (int j = 1; j < N-1; j++)
            A[i][j] = 2.0 + i + j;
}

void relax()
{
    const double inv6 = 1.0 / 6.0;
    double local_eps = 0.0;

    // Обход по диагоналям: k = i + j
    // Диагонали от 2 (1+1) до 2*(N-2) ((N-2)+(N-2))
    for (int k = 2; k <= 2*(N-2); k++)
    {
        double diag_eps = 0.0;
        
        // Определяем границы для текущей диагонали
        int i_min = (k <= N-1) ? 1 : k - (N-2);
        int i_max = (k <= N-1) ? k-1 : N-2;
        
        // Параллельно обрабатываем все точки на диагонали
        #pragma omp parallel for reduction(max:diag_eps)
        for (int i = i_min; i <= i_max; i++)
        {
            int j = k - i;
            if (j >= 1 && j <= N-2)
            {
                double old = A[i][j];
                // Используем новые значения для A[i-1][j] и A[i][j-1] 
                // (они на предыдущих диагоналях, уже обработаны)
                // и старые значения для A[i+1][j] и A[i][j+1]
                // (они на следующих диагоналях, еще не обработаны)
                double newv = (2.0*A[i-1][j] + A[i+1][j] + 2.0*A[i][j-1] + A[i][j+1]) * inv6;
                A[i][j] = newv;
                double diff = fabs(newv - old);
                if (diff > diag_eps) diag_eps = diff;
            }
        }
        
        if (diag_eps > local_eps) local_eps = diag_eps;
    }
    
    eps = local_eps;
}

void verify()
{
    double s = 0.0;
    double nn = 1.0 / (N * N); 

    // Распараллеливаем вычисление суммы с редукцией
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
    
    printf("  S = %f\n", s);
}