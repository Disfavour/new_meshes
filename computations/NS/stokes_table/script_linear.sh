export OMP_PROC_BIND=spread
export OMP_PLACES=threads
mpirun -n 6 --bind-to core /bin/python3 /home/disfavour/projects/new_meshes/computations/NS_biharmonic/NS_biharmonic_linear_experiments.py 128 1 >> results.txt
mpirun -n 6 --bind-to core /bin/python3 /home/disfavour/projects/new_meshes/computations/NS_biharmonic/NS_biharmonic_linear_experiments.py 128 2 >> results.txt
mpirun -n 6 --bind-to core /bin/python3 /home/disfavour/projects/new_meshes/computations/NS_biharmonic/NS_biharmonic_linear_experiments.py 128 3 >> results.txt
mpirun -n 6 --bind-to core /bin/python3 /home/disfavour/projects/new_meshes/computations/NS_biharmonic/NS_biharmonic_linear_experiments.py 256 1 >> results.txt
mpirun -n 6 --bind-to core /bin/python3 /home/disfavour/projects/new_meshes/computations/NS_biharmonic/NS_biharmonic_linear_experiments.py 256 2 >> results.txt
mpirun -n 6 --bind-to core /bin/python3 /home/disfavour/projects/new_meshes/computations/NS_biharmonic/NS_biharmonic_linear_experiments.py 256 3 >> results.txt
mpirun -n 6 --bind-to core /bin/python3 /home/disfavour/projects/new_meshes/computations/NS_biharmonic/NS_biharmonic_linear_experiments.py 512 1 >> results.txt
mpirun -n 6 --bind-to core /bin/python3 /home/disfavour/projects/new_meshes/computations/NS_biharmonic/NS_biharmonic_linear_experiments.py 512 2 >> results.txt
mpirun -n 6 --bind-to core /bin/python3 /home/disfavour/projects/new_meshes/computations/NS_biharmonic/NS_biharmonic_linear_experiments.py 512 3 >> results.txt
