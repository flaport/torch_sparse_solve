#include <torch/extension.h>
#include <ATen/ATen.h>
#include <vector>
#include <klu.h>

at::Tensor solve_forward(at::Tensor A, at::Tensor b);
std::vector<at::Tensor> solve_backward(at::Tensor grad, at::Tensor A, at::Tensor b, at::Tensor x);
void _klu_solve(at::Tensor Ap, at::Tensor Ai, at::Tensor Ax, at::Tensor b);
std::vector<at::Tensor> _coo_to_csc(at::Tensor A);

at::Tensor solve_forward(at::Tensor A, at::Tensor b) {
    int p = at::size(b, 0);
    int m = at::size(b, 1);
    int n = at::size(b, 2);
    at::Tensor bflat = at::clone(at::reshape(at::transpose(b, 1, 2), {p, m*n}));
    for (int i = 0; i < p; i++) {
        std::vector<at::Tensor> Ap_Ai_Ax = _coo_to_csc(A[i].coalesce());
        _klu_solve(Ap_Ai_Ax[0], Ap_Ai_Ax[1], Ap_Ai_Ax[2], bflat[i]); // result will be in bflat
    }
    return at::transpose(bflat.view({p,n,m}), 1, 2);
}

std::vector<at::Tensor> solve_backward(at::Tensor grad, at::Tensor A, at::Tensor b, at::Tensor x) {
    at::Tensor gradb = solve_forward(at::transpose(A, -1, -2), grad);
    at::Tensor gradA = -at::bmm(gradb, at::transpose(x, -1, -2));
    return {gradA, gradb};
}

void _klu_solve(at::Tensor Ap, at::Tensor Ai, at::Tensor Ax, at::Tensor b) { // from github.com/kagami-c/PyKLU
    int n_col = at::size(Ap, 0) - 1;
    int n_b = at::size(b, 0);
    int* ap = Ap.data_ptr<int>();
    int* ai = Ai.data_ptr<int>();
    double* ax = Ax.data_ptr<double>();
    double* bb = b.data_ptr<double>();
    for (int i = 0; i < n_b; i+= n_col) {
        klu_symbolic* Symbolic;
        klu_numeric* Numeric;
        klu_common Common;
        klu_defaults(&Common);
        Symbolic = klu_analyze(n_col, ap, ai, &Common);
        Numeric = klu_factor(ap, ai, ax, Symbolic, &Common);
        klu_solve(Symbolic, Numeric, n_col, 1, &bb[i], &Common);
        klu_free_symbolic(&Symbolic, &Common);
        klu_free_numeric(&Numeric, &Common);
    }
}

std::vector<at::Tensor> _coo_to_csc(at::Tensor A) { // based on https://github.com/scipy/scipy/blob/3b36a57/scipy/sparse/sparsetools/coo.h#L34
    at::Tensor Ax = A.values();
    int nnz = Ax.size(0);
    int n_col = A.size(1);

    at::Tensor indices = A.indices();
    at::Tensor Ai = at::_cast_Int(indices[0]);
    at::Tensor Aj = at::_cast_Int(indices[1]);

    at::TensorOptions options = at::TensorOptions().dtype(torch::kInt32).device(at::device_of(Ai));
    at::Tensor Bp = at::zeros(n_col+1, options);
    at::Tensor Bi = at::zeros_like(Ai);
    at::Tensor Bx = at::zeros_like(Ax);

    //compute number of non-zero entries per row of A
    for (int n = 0; n < nnz; n++) {
        Bp[Aj[n]] += 1;
    }

    //cumsum the nnz per row to get Bp
    at::Tensor cumsum = at::zeros_like(Bp[0]);
    at::Tensor temp = at::zeros_like(Bp[0]);
    for(int j = 0; j < n_col; j++) {
        at::fill_(temp, Bp[j]);
        Bp[j] = cumsum;
        at::fill_(cumsum, cumsum + temp);
    }
    Bp[n_col] = nnz;

    //write Ai, Ax into Bi, Bx
    at::Tensor col = at::zeros_like(Aj[0]);
    at::Tensor dest = at::zeros_like(Bp[0]);
    for(int n = 0; n < nnz; n++) {
        at::fill_(col, Aj[n]);
        at::fill_(dest, Bp[col]);
        Bi[dest] = Ai[n];
        Bx[dest] = Ax[n];
        Bp[col] = Bp[col] + 1;
    }

    at::Tensor last = at::zeros_like(temp);
    for(int i = 0; i <= n_col; i++) {
        at::fill_(temp, Bp[i]);
        Bp[i] = last;
        at::fill_(last, temp);
    }

    return {Bp, Bi, Bx};
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward",     &solve_forward,  "solve forward");
    m.def("backward",    &solve_backward, "solve backward");
    m.def("_klu_solve",  &_klu_solve,     "sparse solve");
    m.def("_coo_to_csc", &_coo_to_csc,    "COO to CSC");
}
