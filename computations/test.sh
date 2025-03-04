mpirun -n 1 python3 computations/poisson.py data/parallel/direct_1
mpirun -n 2 python3 computations/poisson.py data/parallel/direct_2
mpirun -n 3 python3 computations/poisson.py data/parallel/direct_3
mpirun -n 4 python3 computations/poisson.py data/parallel/direct_4
mpirun -n 5 python3 computations/poisson.py data/parallel/direct_5
mpirun -n 6 python3 computations/poisson.py data/parallel/direct_6

mpirun -n 1 --bind-to core --map-by socket python3 computations/poisson.py data/parallel/direct_1_with_options
mpirun -n 2 --bind-to core --map-by socket python3 computations/poisson.py data/parallel/direct_2_with_options
mpirun -n 3 --bind-to core --map-by socket python3 computations/poisson.py data/parallel/direct_3_with_options
mpirun -n 4 --bind-to core --map-by socket python3 computations/poisson.py data/parallel/direct_4_with_options
mpirun -n 5 --bind-to core --map-by socket python3 computations/poisson.py data/parallel/direct_5_with_options
mpirun -n 6 --bind-to core --map-by socket python3 computations/poisson.py data/parallel/direct_6_with_options


mpirun -n 1 python3 computations/poisson2.py data/parallel/krylov_1
mpirun -n 2 python3 computations/poisson2.py data/parallel/krylov_2
mpirun -n 3 python3 computations/poisson2.py data/parallel/krylov_3
mpirun -n 4 python3 computations/poisson2.py data/parallel/krylov_4
mpirun -n 5 python3 computations/poisson2.py data/parallel/krylov_5
mpirun -n 6 python3 computations/poisson2.py data/parallel/krylov_6

mpirun -n 1 --bind-to core --map-by socket python3 computations/poisson2.py data/parallel/krylov_1_with_options
mpirun -n 2 --bind-to core --map-by socket python3 computations/poisson2.py data/parallel/krylov_2_with_options
mpirun -n 3 --bind-to core --map-by socket python3 computations/poisson2.py data/parallel/krylov_3_with_options
mpirun -n 4 --bind-to core --map-by socket python3 computations/poisson2.py data/parallel/krylov_4_with_options
mpirun -n 5 --bind-to core --map-by socket python3 computations/poisson2.py data/parallel/krylov_5_with_options
mpirun -n 6 --bind-to core --map-by socket python3 computations/poisson2.py data/parallel/krylov_6_with_options