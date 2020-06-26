#include <torch/extension.h>
#include <ATen/ATen.h>
#include <vector>
#include <klu.h>

int sparse_solve(int n, int* Ap, int* Ai, double* Ax, double* b) { // from github.com/kagami-c/PyKLU
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

torch::Tensor solve_forward(torch::Tensor A, torch::Tensor b){
  auto result = torch::zeros_like(b);
  for (int i = 0; i < at::size(b, 0); i++){
      result[i] = torch::mm(torch::inverse(A[i]), b[i]); // we'll use an actual solver later.
  }
  return result;
}

std::vector<torch::Tensor> solve_backward(torch::Tensor grad, torch::Tensor A, torch::Tensor b, torch::Tensor x){
    auto gradb = solve_forward(at::transpose(A, -1, -2), grad);
    auto gradA = -torch::bmm(gradb, at::transpose(x, -1, -2));
    return {gradA, gradb};
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("forward", &solve_forward, "solve forward");
  m.def("backward", &solve_backward, "solve backward");
  m.def("_sparse_solve", &sparse_solve, "sparse solve");
}
