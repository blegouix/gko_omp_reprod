// Copyright (C) The DDC development team, see COPYRIGHT.md file
//
// SPDX-License-Identifier: MIT

#include <algorithm>
#include <memory>
#include <optional>
#include <type_traits>
#include <utility>

#include <ginkgo/ginkgo.hpp>

int main(int argc, char *argv[]) {
  std::shared_ptr const gko_exec = gko::OmpExecutor::create();

  std::ifstream fout_csr("../csr2.mtx");
  std::ifstream fout_rhs("../rhs2.mtx");
  std::shared_ptr csr = gko::share(gko::read<gko::matrix::Csr<double, int>>(fout_csr, gko_exec));
  std::shared_ptr rhs = gko::share(gko::read<gko::matrix::Dense<double>>(fout_rhs, gko_exec));
  fout_csr.close();
  fout_rhs.close();

  std::shared_ptr const residual_criterion =
      gko::stop::ResidualNorm<double>::build().with_reduction_factor(1e-16).on(
          gko_exec);

  std::shared_ptr const iterations_criterion =
      gko::stop::Iteration::build().with_max_iters(1000u).on(gko_exec);

  std::unique_ptr const solver_factory =
      gko::solver::Bicgstab<double>::build()
          .with_criteria(residual_criterion, iterations_criterion)
          .on(gko_exec);

  std::shared_ptr<gko::solver::Bicgstab<double>> m_solver = solver_factory->generate(csr);
  gko_exec->synchronize();

  std::shared_ptr result = gko::share(gko::initialize<gko::matrix::Dense<double>>({0.,0.,0.,0.,0.,0.,0.,0.,0.,0.}, gko_exec));
  m_solver->apply(rhs, result);
  gko_exec->synchronize();

  std::shared_ptr one = gko::share(gko::initialize<gko::matrix::Dense<double>>({1.}, gko_exec));
  std::shared_ptr neg_one = gko::share(gko::initialize<gko::matrix::Dense<double>>({-1.}, gko_exec));
  std::shared_ptr residual = gko::share(gko::initialize<gko::matrix::Dense<double>>({0.,0.,0.,0.,0.,0.,0.,0.,0.,0.}, gko_exec));
  csr->apply(one, result, neg_one, rhs);
  gko::write(std::cout, rhs);
  return 1;
}
