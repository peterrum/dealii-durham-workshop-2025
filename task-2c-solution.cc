// ------------------------------------------------------------------------
//
// SPDX-License-Identifier: LGPL-2.1-or-later
// Copyright (C) 2021 - 2025 by the deal.II authors
//
// This file is part of the deal.II library.
//
// Part of the source code is dual licensed under Apache-2.0 WITH
// LLVM-exception OR LGPL-2.1-or-later. Detailed license information
// governing the source code and code contributions can be found in
// LICENSE.md and CONTRIBUTING.md at the top level directory of deal.II.
//
// Author: Peter Munch, 2021, Helmholtz-Zentrum Hereon
//         Peter Munch, 2025, Technical University Berlin
//
// Solve Poisson problem with source term.
//
// ------------------------------------------------------------------------

#include <deal.II/dofs/dof_handler.h>
#include <deal.II/dofs/dof_tools.h>

#include <deal.II/fe/fe_q.h>

#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/tria.h>

#include <deal.II/lac/dynamic_sparsity_pattern.h>
#include <deal.II/lac/precondition.h>
#include <deal.II/lac/solver_cg.h>
#include <deal.II/lac/solver_control.h>
#include <deal.II/lac/sparse_matrix.h>
#include <deal.II/lac/sparsity_pattern.h>

#include <deal.II/numerics/data_out.h>
#include <deal.II/numerics/vector_tools.h>

#include <fstream>

using namespace dealii;

int
main()
{
  const unsigned int dim = 2, degree = 3, n_global_refinements = 3;

  Triangulation<dim> tria;
  GridGenerator::hyper_cube(tria, 0, 1, true);
  tria.refine_global(n_global_refinements);

  FE_Q<dim>       fe(degree);
  QGauss<dim>     quad(degree + 1);
  QGauss<dim - 1> quad_face(degree + 1);
  MappingQ1<dim>  mapping;

  DoFHandler<dim> dof_handler(tria);
  dof_handler.distribute_dofs(fe);

  // deal with boundary conditions
  AffineConstraints<double> constraints;
  VectorTools::interpolate_boundary_values(
    mapping, dof_handler, 0, Functions::ZeroFunction<dim>(), constraints);
  constraints.close();

  // initialize vectors and system matrix
  Vector<double>       x(dof_handler.n_dofs()), b(dof_handler.n_dofs());
  SparseMatrix<double> A;
  SparsityPattern      sparsity_pattern;

  DynamicSparsityPattern dsp(dof_handler.n_dofs());
  DoFTools::make_sparsity_pattern(dof_handler, dsp);
  sparsity_pattern.copy_from(dsp);
  A.reinit(sparsity_pattern);

  // assemble right-hand side and system matrix
  FullMatrix<double>                   cell_matrix;
  Vector<double>                       cell_rhs;
  std::vector<types::global_dof_index> local_dof_indices;

  FEValues<dim>     fe_values(mapping,
                          fe,
                          quad,
                          update_values | update_gradients | update_JxW_values);
  FEFaceValues<dim> fe_face_values(mapping,
                                   fe,
                                   quad_face,
                                   update_values | update_JxW_values);

  // loop over all cells
  for (const auto &cell : dof_handler.active_cell_iterators())
    {
      fe_values.reinit(cell);

      const unsigned int dofs_per_cell = cell->get_fe().dofs_per_cell;
      cell_matrix.reinit(dofs_per_cell, dofs_per_cell);
      cell_rhs.reinit(dofs_per_cell);

      // loop over cell dofs
      for (const auto q : fe_values.quadrature_point_indices())
        {
          for (const auto i : fe_values.dof_indices())
            for (const auto j : fe_values.dof_indices())
              cell_matrix(i, j) +=
                (fe_values.shape_grad(i, q) * // grad phi_i(x_q)
                 fe_values.shape_grad(j, q) * // grad phi_j(x_q)
                 fe_values.JxW(q));           // dx

          for (const unsigned int i : fe_values.dof_indices())
            cell_rhs(i) += (fe_values.shape_value(i, q) * // phi_i(x_q)
                            1. *                          // f(x_q)
                            fe_values.JxW(q));            // dx
        }

      for (const auto &face : cell->face_iterators())
        {
          if (face->at_boundary() == false || face->boundary_id() != 1)
            continue;

          fe_face_values.reinit(cell, face);

          for (const auto q : fe_face_values.quadrature_point_indices())
            for (const unsigned int i : fe_face_values.dof_indices())
              cell_rhs(i) += (fe_face_values.shape_value(i, q) * // phi_i(x_q)
                              1. *                               // f(x_q)
                              fe_face_values.JxW(q));            // dx
        }

      local_dof_indices.resize(cell->get_fe().dofs_per_cell);
      cell->get_dof_indices(local_dof_indices);

      constraints.distribute_local_to_global(
        cell_matrix, cell_rhs, local_dof_indices, A, b);
    }

  // solve linear equation system
  ReductionControl         reduction_control(100, 1e-10, 1e-4);
  SolverCG<Vector<double>> solver(reduction_control);
  solver.solve(A, x, b, PreconditionIdentity());

  printf("Solved in %d iterations.\n", reduction_control.last_step());

  constraints.distribute(x);

  // output results
  DataOut<dim> data_out;
#if false
  DataOutBase::VtkFlags flags;
  flags.write_higher_order_cells = true;
  data_out.set_flags(flags);
#endif
  data_out.attach_dof_handler(dof_handler);
  data_out.add_data_vector(dof_handler, x, "solution");
  data_out.build_patches(mapping, degree + 1);

  std::ofstream output("solution.vtu");
  data_out.write_vtu(output);
}