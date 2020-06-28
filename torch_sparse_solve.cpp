#include <torch/extension.h>
#include <ATen/ATen.h>
#include <vector>
#include <klu.h>


at::Tensor solve_forward(at::Tensor A, at::Tensor b){
    auto result = at::zeros_like(b);
    for (int i = 0; i < at::size(b, 0); i++){
        result[i] = at::mm(at::inverse(A[i]), b[i]); // we'll use an actual solver later.
    }
    return result;
}

std::vector<at::Tensor> solve_backward(at::Tensor grad, at::Tensor A, at::Tensor b, at::Tensor x){
    auto gradb = solve_forward(at::transpose(A, -1, -2), grad);
    auto gradA = -at::bmm(gradb, at::transpose(x, -1, -2));
    return {gradA, gradb};
}

std::vector<at::Tensor> _coo_to_csc(at::Tensor A){ // based on https://github.com/scipy/scipy/blob/3b36a57/scipy/sparse/sparsetools/coo.h#L34
    A = A.coalesce();
    at::Tensor Ax = A.values();
    int nnz = Ax.size(0);
    int n_col = A.size(1);

    at::Tensor indices = A.indices();
    at::Tensor Ai = indices[0];
    at::Tensor Aj = indices[1];

    auto options = torch::TensorOptions().dtype(torch::kInt64).device(at::device_of(Ai));
    at::Tensor Bp = at::zeros(nnz+1, options);
    at::Tensor Bi = at::zeros_like(Ai);
    at::Tensor Bx = at::zeros_like(Ax);

    //compute number of non-zero entries per row of A
    for (int n = 0; n < nnz; n++){
        Bp[Aj[n]] += 1;
    }

    //cumsum the nnz per row to get Bp
    at::Tensor cumsum = at::zeros_like(Bp[0]);
    at::Tensor temp = at::zeros_like(Bp[0]);
    for(int j = 0; j < n_col; j++){
        at::fill_(temp, Bp[j]);
        Bp[j] = cumsum;
        at::fill_(cumsum, cumsum + temp);
    }
    Bp[n_col] = nnz;

    //write Ai, Ax into Bi, Bx
    at::Tensor col = at::zeros_like(Aj[0]);
    at::Tensor dest = at::zeros_like(Bp[0]);
    for(int n = 0; n < nnz; n++){
        at::fill_(col, Aj[n]);
        at::fill_(dest, Bp[col]);
        Bi[dest] = Ai[n];
        Bx[dest] = Ax[n];
        Bp[col] = Bp[col] + 1;
    }

    at::Tensor last = at::zeros_like(temp);
    for(int i = 0; i <= n_col; i++){
        at::fill_(temp, Bp[i]);
        Bp[i] = last;
        at::fill_(last, temp);
    }


    return {Bp, Bi, Bx};
}

int _sparse_solve(int n, int* Ap, int* Ai, double* Ax, double* b) { // from github.com/kagami-c/PyKLU
    klu_symbolic* Symbolic;
    klu_numeric* Numeric;
    klu_common Common;
    klu_defaults(&Common);
    Symbolic = klu_analyze(n, Ap, Ai, &Common);
    Numeric = klu_factor(Ap, Ai, Ax, Symbolic, &Common);
    klu_solve(Symbolic, Numeric, 5, 1, b, &Common);
    klu_free_symbolic(&Symbolic, &Common);
    klu_free_numeric(&Numeric, &Common);
    return 0;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &solve_forward, "solve forward");
    m.def("backward", &solve_backward, "solve backward");
    m.def("_sparse_solve", &_sparse_solve, "sparse solve");
    m.def("_coo_to_csc", &_coo_to_csc, "COO to CSC");
}
