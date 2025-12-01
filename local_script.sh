#!/bin/bash

set -e
cd "$(dirname "$0")" 

OUT=logs
rm -rf "$OUT"
mkdir -p "$OUT"

: > "$OUT/build.log"
: > "$OUT/meta.txt"
: > "$OUT/original.log"
: > "$OUT/optimized.log"
: > "$OUT/omp_diag.log"
: > "$OUT/omp_for.log"

echo "DATE=$(date -Iseconds) HOST=$(hostname) PWD=$(pwd)" >> "$OUT/meta.txt"

CORES=$(getconf _NPROCESSORS_ONLN 2>/dev/null || nproc 2>/dev/null || echo 1)
echo "CORES=$CORES" >> "$OUT/meta.txt"

N_LIST="66 130 258 514 1026 2050"
P_LIST="1 2 3 4 5 6 7 8"  

for N in $N_LIST; do
  echo "N=$N" >> "$OUT/meta.txt"    

  gcc -std=gnu99 -O3 -fopenmp -DN=$N -o original original.c >> "$OUT/build.log" 2>&1
  gcc -std=gnu99 -O3 -fopenmp -DN=$N -o optimized optimized.c >> "$OUT/build.log" 2>&1
  gcc -std=gnu99 -O3 -fopenmp -DN=$N -o omp_diag omp_diag.c >> "$OUT/build.log" 2>&1
  gcc -std=gnu99 -O3 -fopenmp -DN=$N -o omp_for omp_for.c >> "$OUT/build.log" 2>&1

  export OMP_NUM_THREADS=1

  echo "N=$N" >> "$OUT/original.log"
  ./original >> "$OUT/original.log" 2>&1

  echo "N=$N" >> "$OUT/optimized.log"
  ./optimized >> "$OUT/optimized.log" 2>&1

  echo "N=$N" >> "$OUT/omp_diag.log"
  echo "N=$N" >> "$OUT/omp_for.log"

  for p in $P_LIST; do
    export OMP_NUM_THREADS="$p"

    echo "N=$N p=$p" >> "$OUT/omp_diag.log"
    ./omp_diag >> "$OUT/omp_diag.log" 2>&1

    echo "N=$N p=$p" >> "$OUT/omp_for.log"
    ./omp_for >> "$OUT/omp_for.log" 2>&1
  done
done

echo "Тестирование завершено. Результаты в папке $OUT/"