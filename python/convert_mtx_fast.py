import fast_matrix_market as fmm
from petsc4py import PETSc

def convert(file_in, file_out):
    # Read MTX into SciPy CSR (C-optimized)
    A = fmm.mmread(file_in).tocsr()

    # You want: columns = sources, rows = destinations
    # So swap row/col ordering once in C with .transpose()
    A = A.transpose().tocsr()  # << Critical optimization

    nrows, ncols = A.shape

    # Create PETSc matrix
    mat = PETSc.Mat().create()
    mat.setSizes([nrows, ncols])
    mat.setType(PETSc.Mat.Type.AIJ)
    mat.setUp()

    indptr = A.indptr.astype('int32', copy=False)
    indices = A.indices.astype('int32', copy=False)
    data = A.data

    # Bulk insert using PETSc’s optimized CSR loader
    mat.setValuesCSR(indptr, indices, data)

    mat.assemblyBegin()
    mat.assemblyEnd()

    # Write to PETSc binary file
    viewer = PETSc.Viewer().createBinary(file_out, 'w')
    mat.view(viewer)
    viewer.destroy()
    mat.destroy()


convert("../data/small.mtx", "../data/small.pm")
