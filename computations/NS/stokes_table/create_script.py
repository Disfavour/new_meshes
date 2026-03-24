import os.path


def run1():
    mesh_sizes = (128, 256, 512)
    p_degree = (1, 2, 3)

    with open(os.path.join(os.path.dirname(__file__), 'script_linear.sh'), "w") as f:
        f.write("export OMP_PROC_BIND=spread\n")
        f.write("export OMP_PLACES=threads\n")

        # /usr/bin/mpirun

        for n in mesh_sizes:
            for p in p_degree:
                f.write(f"mpirun -n 6 --bind-to core /bin/python3 /home/disfavour/projects/new_meshes/computations/NS_biharmonic/NS_biharmonic_linear_experiments.py {n} {p} >> results.txt\n")


def run2():
    mesh_sizes = (128, 256, 512)
    mesh_names = (f'meshes/unit_square/unit_square_{ms}.msh' for ms in mesh_sizes)
    p_degree = (1, 2, 3)

    with open(os.path.join(os.path.dirname(__file__), 'script_linear2.sh'), "w") as f:
        f.write("export OMP_PROC_BIND=spread\n")
        f.write("export OMP_PLACES=threads\n")

        # /usr/bin/mpirun

        for n in mesh_names:
            for p in p_degree:
                f.write(f"mpirun -n 6 --bind-to core /bin/python3 /home/disfavour/projects/new_meshes/computations/NS_biharmonic/NS_biharmonic_linear_experiments.py {n} {p} >> results2.txt\n")


if __name__ == '__main__':
    run2()
