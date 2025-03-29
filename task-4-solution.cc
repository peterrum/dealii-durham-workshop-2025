// ------------------------------------------------------------------------
//
// SPDX-License-Identifier: LGPL-2.1-or-later
// Copyright (C) 2024 - by the deal.II authors
//
// Author: Peter Munch, 2024, Uppsala University
//
// Solve Poisson problem
//   -Δu = f
// on different meshes.
//
//
// The first set are 2D Kershew meshes with different anisotropy.
// The setup is adopted from Phillips, M. and Fischer, P., 2022.
// Optimal chebyshev smoothers and one-sided v-cycles.
//
// The manufactured solution is:
//   u(x) := sin(π * x) * sin(π * y),
// which also gives the Dirichlet boundary condition.
//
// The needed right-hand side function is:
//   f(x) := 2 * π^2 * u(x).
//
//
// The second geometry is an L-shaped domain. No right-hand-side function
// is used (f:=0) and the manufactored solution polar coordinates is:
//   u(r, φ) := r^(2/3) * sin(2/3 * φ).
//
// ------------------------------------------------------------------------

#include <deal.II/base/conditional_ostream.h>
#include <deal.II/base/convergence_table.h>
#include <deal.II/base/geometric_utilities.h>
#include <deal.II/base/mpi.h>
#include <deal.II/base/parameter_handler.h>
#include <deal.II/base/quadrature_lib.h>

#include <deal.II/distributed/tria.h>

#include <deal.II/dofs/dof_handler.h>
#include <deal.II/dofs/dof_tools.h>

#include <deal.II/fe/fe_q.h>
#include <deal.II/fe/fe_system.h>
#include <deal.II/fe/fe_tools.h>
#include <deal.II/fe/mapping_q1.h>
#include <deal.II/fe/mapping_q_cache.h>

#include <deal.II/grid/grid_generator.h>

#include <deal.II/lac/trilinos_solver.h>
#include <deal.II/lac/trilinos_sparse_matrix.h>
#include <deal.II/lac/trilinos_sparsity_pattern.h>

#include <deal.II/numerics/data_out.h>
#include <deal.II/numerics/vector_tools.h>

using namespace dealii;


struct Parameters
{
  // general
  unsigned int dim         = 2;
  unsigned int fe_degree_p = 1;
  unsigned int fe_degree_v = 2;

  // mesh
  unsigned int mapping_degree = 3;
  unsigned int n_refinements  = 4;
  double       eps            = 0.3;

  // experiments
  bool compute_error_solution = true;
  bool output_paraview        = true;

  void
  parse(const std::string file_name)
  {
    if (file_name == "")
      return;

    dealii::ParameterHandler prm;
    add_parameters(prm);

    prm.parse_input(file_name, "", true);
  }

private:
  void
  add_parameters(ParameterHandler &prm)
  {
    // system
    prm.enter_subsection("system");
    prm.add_parameter("dim", dim);
    prm.add_parameter("fe degree v", fe_degree_v);
    prm.add_parameter("fe degree p", fe_degree_p);
    prm.leave_subsection();

    // geometry parameters
    prm.enter_subsection("geometry");
    prm.add_parameter("n refinements", n_refinements);
    prm.leave_subsection();

    // settings of experiments
    prm.enter_subsection("experiments");
    prm.add_parameter("output paraview", output_paraview);
    prm.add_parameter("compute error solution", compute_error_solution);
    prm.leave_subsection();
  }
};



template <int dim, typename Number>
void
test()
{
  Parameters params;

  ConvergenceTable table;

  // general
  using VectorType = LinearAlgebra::distributed::Vector<Number>;

  const MPI_Comm comm = MPI_COMM_WORLD;

  ConditionalOStream pcout(std::cout,
                           Utilities::MPI::this_mpi_process(comm) == 0);

  MappingQ1<dim> mapping;
  FESystem<dim>  fe(FE_Q<dim>(params.fe_degree_v),
                   dim,
                   FE_Q<dim>(params.fe_degree_p),
                   1);
  QGauss<dim>    quadrature(params.fe_degree_v + 1);

  parallel::distributed::Triangulation<dim> triangulation(comm);

  std::shared_ptr<Function<dim, Number>> exact_solution;
  std::shared_ptr<Function<dim, Number>> rhs_func;

  exact_solution = std::make_shared<FunctionFromFunctionObjects<dim>>(
    [&](const auto &p, const auto c) {
      const double a = numbers::PI;
      const double x = p[0];
      const double y = p[1];

      if (c == 0)
        return std::sin(a * x) * std::sin(a * x) * std::cos(a * y) *
               std::sin(a * y);
      else if (c == 1)
        return -std::cos(a * x) * std::sin(a * x) * std::sin(a * y) *
               std::sin(a * y);
      else if (c == 2)
        return std::sin(a * x) * std::sin(a * y);

      AssertThrow(false, ExcNotImplemented());

      return 0.0;
    },
    dim + 1);

  rhs_func = std::make_shared<FunctionFromFunctionObjects<dim>>(
    [&](const auto &p, const auto c) {
      const double a = numbers::PI;
      const double x = p[0];
      const double y = p[1];

      if (c == 0)
        return 2 * a * a *
                 (std::sin(a * x) * std::sin(a * x) -
                  std::cos(a * x) * std::cos(a * x)) *
                 std::sin(a * y) * std::cos(a * y) +
               4 * a * a * std::sin(a * x) * std::sin(a * x) * std::sin(a * y) *
                 std::cos(a * y) +
               a * std::sin(a * y) * std::cos(a * x);
      else if (c == 1)
        return -2 * a * a *
                 (std::sin(a * y) * std::sin(a * y) -
                  std::cos(a * y) * std::cos(a * y)) *
                 std::sin(a * x) * std::cos(a * x) -
               4 * a * a * std::sin(a * x) * std::sin(a * y) * std::sin(a * y) *
                 std::cos(a * x) +
               a * std::sin(a * x) * std::cos(a * y);
      else if (c == 2)
        return 0.0;

      AssertThrow(false, ExcNotImplemented());

      return 0.0;
    },
    dim + 1);

  // create mesh

  GridGenerator::hyper_cube(triangulation, -1.0, 1.0);
  triangulation.refine_global(params.n_refinements);


  // create DoFHandler and constraints
  DoFHandler<dim> dof_handler(triangulation);
  dof_handler.distribute_dofs(fe);

  table.add_value("dim", dim);
  table.add_value("k_u", params.fe_degree_v);
  table.add_value("k_p", params.fe_degree_p);
  table.add_value("l", params.n_refinements);
  table.add_value("#c", triangulation.n_global_active_cells());
  table.add_value("#d", dof_handler.n_dofs());

  table.add_value("dx", 2.0 / Utilities::pow(2, params.n_refinements));

  AffineConstraints<double> constraints;
  constraints.reinit(DoFTools::extract_locally_relevant_dofs(dof_handler));
  VectorTools::interpolate_boundary_values(
    mapping, dof_handler, 0, *exact_solution, constraints);
  constraints.close();

  // create vectors and set up right-hand-side vector
  const auto partitioner = std::make_shared<const Utilities::MPI::Partitioner>(
    dof_handler.locally_owned_dofs(),
    DoFTools::extract_locally_active_dofs(dof_handler),
    comm);

  LinearAlgebra::distributed::Vector<double> rhs(partitioner);
  LinearAlgebra::distributed::Vector<double> solution(partitioner);

  VectorTools::create_right_hand_side(
    mapping, dof_handler, quadrature, *rhs_func, rhs, constraints);

  // element-stiffness matrix definition
  FEValues<dim>              fe_values(mapping,
                          fe,
                          quadrature,
                          update_values | update_gradients | update_JxW_values |
                            update_quadrature_points);
  FEValuesViews::Vector<dim> velocities(fe_values, 0);
  FEValuesViews::Scalar<dim> pressure(fe_values, dim);

  if (params.fe_degree_v == params.fe_degree_p)
    for (const auto &cell : dof_handler.active_cell_iterators())
      if (cell->is_locally_owned())
        {
          fe_values.reinit(cell);

          const double delta_1 = cell->minimum_vertex_distance();

          Vector<double>                       rhs_local(fe.n_dofs_per_cell());
          std::vector<types::global_dof_index> indices(fe.n_dofs_per_cell());

          cell->get_dof_indices(indices);

          for (const unsigned int q : fe_values.quadrature_point_indices())
            {
              const auto JxW   = fe_values.JxW(q);
              const auto point = fe_values.quadrature_point(q);

              Tensor<1, dim> source;
              for (unsigned int d = 0; d < dim; ++d)
                source[d] = rhs_func->value(point, d);

              for (const unsigned int i : fe_values.dof_indices())
                rhs_local(i) +=
                  delta_1 * source * pressure.gradient(i, q) * JxW;
            }

          constraints.distribute_local_to_global(rhs_local, indices, rhs);
        }

  TrilinosWrappers::SparsityPattern sparsity_pattern;
  sparsity_pattern.reinit(dof_handler.locally_owned_dofs(),
                          dof_handler.get_communicator());
  DoFTools::make_sparsity_pattern(dof_handler, sparsity_pattern, constraints);
  sparsity_pattern.compress();

  TrilinosWrappers::SparseMatrix system_matrix;
  system_matrix.reinit(sparsity_pattern);

  for (const auto &cell : dof_handler.active_cell_iterators())
    if (cell->is_locally_owned())
      {
        fe_values.reinit(cell);

        FullMatrix<double> A(fe.n_dofs_per_cell(), fe.n_dofs_per_cell());
        std::vector<types::global_dof_index> indices(fe.n_dofs_per_cell());

        cell->get_dof_indices(indices);

        const double delta_1 = (params.fe_degree_v == params.fe_degree_p) ?
                                 cell->minimum_vertex_distance() :
                                 0.0;

        for (const unsigned int q : fe_values.quadrature_point_indices())
          {
            const auto JxW = fe_values.JxW(q);

            for (const unsigned int i : fe_values.dof_indices())
              {
                const auto div_v_i  = velocities.divergence(i, q);
                const auto eps_v_i  = velocities.symmetric_gradient(i, q);
                const auto q_i      = pressure.value(i, q);
                const auto grad_q_i = pressure.gradient(i, q);

                for (const unsigned int j : fe_values.dof_indices())
                  {
                    const auto eps_u_j  = velocities.symmetric_gradient(j, q);
                    const auto div_u_j  = velocities.divergence(j, q);
                    const auto p_j      = pressure.value(j, q);
                    const auto grad_p_j = pressure.gradient(j, q);

                    A(i, j) +=
                      (2.0 * scalar_product(eps_u_j, eps_v_i) - p_j * div_v_i +
                       div_u_j * q_i + delta_1 * grad_p_j * grad_q_i) *
                      JxW;
                  }
              }
          }

        constraints.distribute_local_to_global(A, indices, system_matrix);
      }

  TrilinosWrappers::SolverDirect solver;
  solver.solve(system_matrix, solution, rhs);

  // compute error
  if (params.compute_error_solution)
    {
      const ComponentSelectFunction<dim> u_mask(std::make_pair(0, dim),
                                                dim + 1);
      const ComponentSelectFunction<dim> p_mask(dim, dim + 1);

      Vector<Number> cell_wise_error;

      solution.update_ghost_values();
      VectorTools::integrate_difference(mapping,
                                        dof_handler,
                                        solution,
                                        *exact_solution,
                                        cell_wise_error,
                                        quadrature,
                                        VectorTools::NormType::L2_norm,
                                        &u_mask);

      const auto error_u =
        VectorTools::compute_global_error(triangulation,
                                          cell_wise_error,
                                          VectorTools::NormType::L2_norm);

      VectorTools::integrate_difference(mapping,
                                        dof_handler,
                                        solution,
                                        *exact_solution,
                                        cell_wise_error,
                                        quadrature,
                                        VectorTools::NormType::L2_norm,
                                        &p_mask);

      const auto error_p =
        VectorTools::compute_global_error(triangulation,
                                          cell_wise_error,
                                          VectorTools::NormType::L2_norm);

      solution.zero_out_ghost_values();

      pcout << "error (solution): " << error_u << " " << error_p << std::endl;

      table.add_value("error_u", error_u);
      table.set_scientific("error_u", true);
      table.add_value("error_p", error_p);
      table.set_scientific("error_p", true);
    }


  // paraview
  if (params.output_paraview)
    {
      DataOutBase::VtkFlags flags;
      if (dim > 1)
        flags.write_higher_order_cells = true;

      std::vector<std::string> labels(dim + 1, "u");
      labels[dim] = "p";

      std::vector<std::string> labels_ana(dim + 1, "ana_u");
      labels_ana[dim] = "ana_p";

      std::vector<DataComponentInterpretation::DataComponentInterpretation>
        data_component_interpretation(
          dim + 1, DataComponentInterpretation::component_is_part_of_vector);
      data_component_interpretation[dim] =
        DataComponentInterpretation::component_is_scalar;

      DataOut<dim> data_out;
      data_out.set_flags(flags);
      data_out.attach_triangulation(triangulation);
      data_out.add_data_vector(dof_handler,
                               solution,
                               labels,
                               data_component_interpretation);

      // add expected solution
      DoFHandler<dim> dof_handler_solution(triangulation);
      dof_handler_solution.distribute_dofs(FESystem<dim>(
        FE_Q<dim>(params.fe_degree_v), dim, FE_Q<dim>(params.fe_degree_p), 1));

      LinearAlgebra::distributed::Vector<double> analytical_solution(
        std::make_shared<const Utilities::MPI::Partitioner>(
          dof_handler_solution.locally_owned_dofs(),
          DoFTools::extract_locally_active_dofs(dof_handler_solution),
          comm));
      VectorTools::interpolate(mapping,
                               dof_handler_solution,
                               *exact_solution,
                               analytical_solution);
      data_out.add_data_vector(dof_handler_solution,
                               analytical_solution,
                               labels_ana,
                               data_component_interpretation);

      // add MPI ranks
      Vector<double> ranks(triangulation.n_active_cells());
      ranks = Utilities::MPI::this_mpi_process(comm);
      data_out.add_data_vector(ranks, "ranks");

      // write data
      data_out.build_patches(
        mapping,
        params.fe_degree_v,
        DataOut<dim>::CurvedCellRegion::curved_inner_cells);
      data_out.write_vtu_in_parallel("stokes.vtu", comm);
    }

  if (Utilities::MPI::this_mpi_process(MPI_COMM_WORLD) == 0)
    table.write_text(std::cout);
}



int
main(int argc, char **argv)
{
  Utilities::MPI::MPI_InitFinalize mpi_initialization(argc, argv, 1);

  test<2, double>();

}
