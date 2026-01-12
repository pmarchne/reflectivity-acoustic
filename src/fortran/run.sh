export OMP_NUM_THREADS=512

f2py -c -m reflectivity reflectivity.f90 --f90flags="-O3 -march=native -ffast-math -funroll-loops -fopenmp -fopenmp-simd" --opt="-O3 -march=native -ffast-math -funroll-loops -fopenmp" -lgomp

python3 reflectivity_benchmark.py