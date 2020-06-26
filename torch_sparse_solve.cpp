#include <torch/extension.h>
#include <ATen/ATen.h>
#include <vector>

torch::Tensor solve_forward(torch::Tensor A, torch::Tensor b){
  auto result = torch::zeros_like(b);
  for (int i = 0; i < at::size(b, 0); i++){
      result[i] = torch::mm(torch::inverse(A[i]), b[i]); // we'll use an actual solver later.
  }
  return result;
}

std::vector<torch::Tensor> solve_backward(torch::Tensor grad, torch::Tensor A, torch::Tensor b, torch::Tensor x){
    auto gradb = at::transpose(solve_forward(at::transpose(A, -1, -2), grad), -1, -2);
    auto gradA = -torch::bmm(at::transpose(gradb, -1, -2), at::transpose(x, -1, -2));
    return {gradA, gradb};
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("forward", &solve_forward, "solve forward");
  m.def("backward", &solve_backward, "solve backward");
}
