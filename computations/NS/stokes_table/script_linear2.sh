export OMP_PROC_BIND=spread
export OMP_PLACES=threads
mpirun -n 6 --bind-to core /bin/python3 /home/disfavour/projects/new_meshes/computations/NS_biharmonic/NS_biharmonic_linear_experiments.py meshes/unit_square/unit_square_128.msh 1 >> results2.txt
mpirun -n 6 --bind-to core /bin/python3 /home/disfavour/projects/new_meshes/computations/NS_biharmonic/NS_biharmonic_linear_experiments.py meshes/unit_square/unit_square_128.msh 2 >> results2.txt
mpirun -n 6 --bind-to core /bin/python3 /home/disfavour/projects/new_meshes/computations/NS_biharmonic/NS_biharmonic_linear_experiments.py meshes/unit_square/unit_square_128.msh 3 >> results2.txt
mpirun -n 6 --bind-to core /bin/python3 /home/disfavour/projects/new_meshes/computations/NS_biharmonic/NS_biharmonic_linear_experiments.py meshes/unit_square/unit_square_256.msh 1 >> results2.txt
mpirun -n 6 --bind-to core /bin/python3 /home/disfavour/projects/new_meshes/computations/NS_biharmonic/NS_biharmonic_linear_experiments.py meshes/unit_square/unit_square_256.msh 2 >> results2.txt
mpirun -n 6 --bind-to core /bin/python3 /home/disfavour/projects/new_meshes/computations/NS_biharmonic/NS_biharmonic_linear_experiments.py meshes/unit_square/unit_square_256.msh 3 >> results2.txt
mpirun -n 6 --bind-to core /bin/python3 /home/disfavour/projects/new_meshes/computations/NS_biharmonic/NS_biharmonic_linear_experiments.py meshes/unit_square/unit_square_512.msh 1 >> results2.txt
mpirun -n 6 --bind-to core /bin/python3 /home/disfavour/projects/new_meshes/computations/NS_biharmonic/NS_biharmonic_linear_experiments.py meshes/unit_square/unit_square_512.msh 2 >> results2.txt
mpirun -n 6 --bind-to core /bin/python3 /home/disfavour/projects/new_meshes/computations/NS_biharmonic/NS_biharmonic_linear_experiments.py meshes/unit_square/unit_square_512.msh 3 >> results2.txt
