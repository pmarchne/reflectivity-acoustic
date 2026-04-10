export OMP_NUM_THREADS=24

f2py -c -m reflectivity ./src/fortran/reflectivity.f90 --f90flags="-O3 -march=native -ffast-math -funroll-loops -fopenmp -fopenmp-simd" --opt="-O3 -march=native -ffast-math -funroll-loops -fopenmp" -lgomp

f2py -c -m reflectivity_adj ./src/fortran/reflectivity_adj.f90 --f90flags="-O3 -march=native -ffast-math -funroll-loops -fopenmp -fopenmp-simd" --opt="-O3 -march=native -ffast-math -funroll-loops -fopenmp" -lgomp

#python3 reflectivity_benchmark.py