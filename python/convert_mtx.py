from scipy.io import mmread
from petsc4py import PETSc

def convert(file_in, file_out):
    A = mmread(file_in).tocsr()
    nrows, ncols = A.shape
    mat = PETSc.Mat().create()
    mat.setSizes([nrows, ncols])
    mat.setType('aij')  # PETSc sparse format
    mat.setUp()

    for i in range(nrows):
        row_start = A.indptr[i]
        row_end = A.indptr[i + 1]
        cols = A.indices[row_start:row_end]
        vals = A.data[row_start:row_end]
        mat.setValues(i, cols, vals)

    mat.assemblyBegin()
    mat.assemblyEnd()

    viewer = PETSc.Viewer().createBinary(file_out, 'w')
    mat.view(viewer)
    viewer.destroy()
    mat.destroy()
