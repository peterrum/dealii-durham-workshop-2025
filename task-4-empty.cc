// ------------------------------------------------------------------------
//
// SPDX-License-Identifier: LGPL-2.1-or-later
// Copyright (C) 2025 by the deal.II authors
//
// This file is part of the deal.II library.
//
// Part of the source code is dual licensed under Apache-2.0 WITH
// LLVM-exception OR LGPL-2.1-or-later. Detailed license information
// governing the source code and code contributions can be found in
// LICENSE.md and CONTRIBUTING.md at the top level directory of deal.II.
//
// Author: Peter Munch, 2025, Technical University Berlin
//
// Solve Stokes problem.
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



int
main(int argc, char **argv)
{
  Utilities::MPI::MPI_InitFinalize mpi_initialization(argc, argv, 1);

  using Number                              = double;
  const unsigned int dim                    = 2;
  const unsigned int fe_degree_p            = 1;
  const unsigned int fe_degree_v            = 2;
  const unsigned int n_refinements          = 4;
  const bool         compute_error_solution = true;
  const bool         output_paraview        = true;

  ConvergenceTable table;

  // general
  using VectorType = LinearAlgebra::distributed::Vector<Number>;

  const MPI_Comm comm = MPI_COMM_WORLD;

  ConditionalOStream pcout(std::cout,
                           Utilities::MPI::this_mpi_process(comm) == 0);

  MappingQ1<dim> mapping;
  FESystem<dim>  fe(FE_Q<dim>(fe_degree_v), dim, FE_Q<dim>(fe_degree_p), 1);
  QGauss<dim>    quadrature(fe_degree_v + 1);

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
  triangulation.refine_global(n_refinements);


  // create DoFHandler and constraints
  DoFHandler<dim> dof_handler(triangulation);
  dof_handler.distribute_dofs(fe);

  table.add_value("dim", dim);
  table.add_value("k_u", fe_degree_v);
  table.add_value("k_p", fe_degree_p);
  table.add_value("l", n_refinements);
  table.add_value("#c", triangulation.n_global_active_cells());
  table.add_value("#d", dof_handler.n_dofs());

  table.add_value("dx", 2.0 / Utilities::pow(2, n_refinements));

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

  // element-stiffness matrix definition
  FEValues<dim>              fe_values(mapping,
                          fe,
                          quadrature,
                          update_values | update_gradients | update_JxW_values |
                            update_quadrature_points);
  FEValuesViews::Vector<dim> velocities(fe_values, 0);
  FEValuesViews::Scalar<dim> pressure(fe_values, dim);

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
              {
                rhs_local(i) += 0.0; // TODO

                if (fe_degree_v == fe_degree_p)
                  rhs_local(i) += 0.0; // TODO
              }
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

        const double delta_1 =
          (fe_degree_v == fe_degree_p) ? cell->minimum_vertex_distance() : 0.0;

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

                    A(i, j) += 0.0; // TODO
                  }
              }
          }

        constraints.distribute_local_to_global(A, indices, system_matrix);
      }

  TrilinosWrappers::SolverDirect solver;
  solver.solve(system_matrix, solution, rhs);

  // compute error
  if (compute_error_solution)
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

      table.add_value("error_u", error_u);
      table.set_scientific("error_u", true);
      table.add_value("error_p", error_p);
      table.set_scientific("error_p", true);
    }


  // paraview
  if (output_paraview)
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
      dof_handler_solution.distribute_dofs(
        FESystem<dim>(FE_Q<dim>(fe_degree_v), dim, FE_Q<dim>(fe_degree_p), 1));

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
        fe_degree_v,
        DataOut<dim>::CurvedCellRegion::curved_inner_cells);
      data_out.write_vtu_in_parallel("stokes.vtu", comm);
    }

  if (Utilities::MPI::this_mpi_process(MPI_COMM_WORLD) == 0)
    table.write_text(std::cout);
}
