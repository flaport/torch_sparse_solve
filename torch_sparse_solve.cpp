#include <torch/extension.h>
#include <ATen/ATen.h>
#include <vector>
#include <klu.h>

at::Tensor solve_forward(at::Tensor A, at::Tensor b);
std::vector<at::Tensor> solve_backward(at::Tensor grad, at::Tensor A, at::Tensor b, at::Tensor x);
void _klu_solve(at::Tensor Ap, at::Tensor Ai, at::Tensor Ax, at::Tensor b);
std::vector<at::Tensor> _coo_to_csc(int ncol, at::Tensor Ai, at::Tensor Aj, at::Tensor Ax);

at::Tensor solve_forward(at::Tensor A, at::Tensor b) {
    int p = at::size(b, 0);
    int m = at::size(b, 1);
    int n = at::size(b, 2);
    at::Tensor bflat = at::clone(at::reshape(at::transpose(b, 1, 2), {p, m*n}));
    at::Tensor Ax = at::reshape(A._values(), {p, -1});
    at::Tensor Ai = at::reshape(at::_cast_Int(A._indices()[1]), {p, -1});
    at::Tensor Aj = at::reshape(at::_cast_Int(A._indices()[2]), {p, -1});
    for (int i = 0; i < p; i++) {
        std::vector<at::Tensor> Ap_Ai_Ax = _coo_to_csc(m, Ai[i], Aj[i], Ax[i]);
        _klu_solve(Ap_Ai_Ax[0], Ap_Ai_Ax[1], Ap_Ai_Ax[2], bflat[i]); // result will be in bflat
    }
    return at::transpose(bflat.view({p,n,m}), 1, 2);
}

std::vector<at::Tensor> solve_backward(at::Tensor grad, at::Tensor A, at::Tensor b, at::Tensor x) {
    at::Tensor gradb = solve_forward(A.transpose(1,2).coalesce(), grad);
    int batch_size = A.size(0);
    std::vector<at::Tensor>  gradA_list;
    for (int i=0; i<batch_size; i++) {
        at::Tensor indices = A[i].coalesce()._indices();
        at::Tensor gradA_tmp = at::sparse_coo_tensor(indices, -at::sum((gradb.index({i, indices.index({0})}) * x.index({i, indices.index({1})})), -1)).unsqueeze(0);
        gradA_list.push_back(gradA_tmp);
    }
    at::Tensor gradA = at::cat(gradA_list, 0);
    return {gradA, gradb};
}

void _klu_solve(at::Tensor Ap, at::Tensor Ai, at::Tensor Ax, at::Tensor b) {
    int ncol = at::size(Ap, 0) - 1;
    int nb = at::size(b, 0);
    int* ap = Ap.data_ptr<int>();
    int* ai = Ai.data_ptr<int>();
    double* ax = Ax.data_ptr<double>();
    double* bb = b.data_ptr<double>();
    klu_symbolic* Symbolic;
    klu_numeric* Numeric;
    klu_common Common;
    klu_defaults(&Common);
    Symbolic = klu_analyze(ncol, ap, ai, &Common);
    Numeric = klu_factor(ap, ai, ax, Symbolic, &Common);
    klu_solve(Symbolic, Numeric, ncol, nb/ncol, bb, &Common);
    klu_free_symbolic(&Symbolic, &Common);
    klu_free_numeric(&Numeric, &Common);
}

std::vector<at::Tensor> _coo_to_csc(int ncol, at::Tensor Ai, at::Tensor Aj, at::Tensor Ax) {
    int nnz = at::size(Ax, 0);
    at::TensorOptions options = at::TensorOptions().dtype(torch::kInt32).device(at::device_of(Ai));
    at::Tensor Bp = at::zeros(ncol+1, options);
    at::Tensor Bi = at::zeros_like(Ai);
    at::Tensor Bx = at::zeros_like(Ax);

    int* ai = Ai.data_ptr<int>();
    int* aj = Aj.data_ptr<int>();
    double* ax = Ax.data_ptr<double>();

    int* bp = Bp.data_ptr<int>();
    int* bi = Bi.data_ptr<int>();
    double* bx = Bx.data_ptr<double>();

    //compute number of non-zero entries per row of A
    for (int n = 0; n < nnz; n++) {
        bp[aj[n]] += 1;
    }

    //cumsum the nnz per row to get Bp
    int cumsum = 0;
    int temp = 0;
    for(int j = 0; j < ncol; j++) {
        temp = bp[j];
        bp[j] = cumsum;
        cumsum += temp;
    }
    bp[ncol] = nnz;

    //write Ai, Ax into Bi, Bx
    int col = 0;
    int dest = 0;
    for(int n = 0; n < nnz; n++) {
        col = aj[n];
        dest = bp[col];
        bi[dest] = ai[n];
        bx[dest] = ax[n];
        bp[col] += 1;
    }

    int last = 0;
    for(int i = 0; i <= ncol; i++) {
        temp = bp[i];
        bp[i] = last;
        last = temp;
    }

    return {Bp, Bi, Bx};
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("solve_forward", &solve_forward, "solve forward");
    m.def("solve_backward", &solve_backward, "solve backward");
    m.def("_klu_solve", &_klu_solve, "sparse solve");
    m.def("_coo_to_csc", &_coo_to_csc, "COO to CSC");
}

