#include <algorithm>
#include <memory>
#include <optional>
#include <type_traits>
#include <utility>

#include <ginkgo/ginkgo.hpp>

int main(int argc, char *argv[]) {
  std::shared_ptr const gko_exec = gko::OmpExecutor::create();

  // load matrices stored as .mtx files
  std::ifstream fout_csr("../csr.mtx");
  std::ifstream fout_rhs("../rhs.mtx");
  std::shared_ptr csr = gko::share(gko::read<gko::matrix::Csr<double, int>>(fout_csr, gko_exec));
  std::shared_ptr rhs = gko::share(gko::read<gko::matrix::Dense<double>>(fout_rhs, gko_exec));
  fout_csr.close();
  fout_rhs.close();

  // refill rhs with the same values it already stores but with better precision (commenting this would make it work)
  for (int i=0; i<10; i++) {
	rhs->at(i) = -std::cos(((double)i/5-1)*3.141592653589793238462643383279502884197);
  }

  std::cout << "----- rhs -----\n";
  gko::write(std::cout, rhs);

  // setup solver 
  std::shared_ptr const residual_criterion =
      gko::stop::ResidualNorm<double>::build().with_reduction_factor(1e-16).on( // 1e-19 would work
          gko_exec);

  std::shared_ptr const iterations_criterion =
      gko::stop::Iteration::build().with_max_iters(1000u).on(gko_exec);

  std::unique_ptr const solver_factory =
      gko::solver::Bicgstab<double>::build()
          .with_criteria(residual_criterion, iterations_criterion)
          .on(gko_exec);
          // .on(gko::ReferenceExecutor::create()); // uncomment this to see using serial executor works

  std::shared_ptr<gko::solver::Bicgstab<double>> m_solver = solver_factory->generate(csr);

  // solve
  std::shared_ptr result = gko::share(gko::initialize<gko::matrix::Dense<double>>({0.,0.,0.,0.,0.,0.,0.,0.,0.,0.}, gko_exec));
  m_solver->apply(rhs, result);

  // compute residual 
  std::shared_ptr one = gko::share(gko::initialize<gko::matrix::Dense<double>>({1.}, gko_exec));
  std::shared_ptr neg_one = gko::share(gko::initialize<gko::matrix::Dense<double>>({-1.}, gko_exec));
  std::shared_ptr residual = gko::share(gko::initialize<gko::matrix::Dense<double>>({0.,0.,0.,0.,0.,0.,0.,0.,0.,0.}, gko_exec));
  csr->apply(one, result, neg_one, rhs);
  std::cout << "----- residual -----\n";
  gko::write(std::cout, rhs);

  std::shared_ptr residual_norm = gko::share(gko::initialize<gko::matrix::Dense<double>>({0.}, gko_exec));
  rhs->compute_norm2(residual_norm);
  std::cout << "----- residual_norm -----\n";
  gko::write(std::cout, residual_norm);
  return 1;
}
