export OMP_NUM_THREADS=$(python3 -c 'import os; print(os.cpu_count())')

f2py --backend meson -c -m reflectivity reflectivity.f90 --f90flags="-O3 -march=native -ffast-math -funroll-loops -fopenmp -fopenmp-simd" --opt="-O3 -march=native -ffast-math -funroll-loops -fopenmp" -lgomp

f2py --backend meson -c -m reflectivity_adj reflectivity_adj.f90 --f90flags="-O3 -march=native -ffast-math -funroll-loops -fopenmp -fopenmp-simd" --opt="-O3 -march=native -ffast-math -funroll-loops -fopenmp" -lgomp

#python3 reflectivity_benchmark.py