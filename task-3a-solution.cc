// ---------------------------------------------------------------------
//
// Copyright (C) 2021 by the deal.II authors
//
// This file is part of the deal.II library.
//
// The deal.II library is free software; you can use it, redistribute
// it, and/or modify it under the terms of the GNU Lesser General
// Public License as published by the Free Software Foundation; either
// version 2.1 of the License, or (at your option) any later version.
// The full text of the license can be found in the file LICENSE.md at
// the top level directory of deal.II.
//
// ---------------------------------------------------------------------

#include <deal.II/distributed/tria.h>

#include <deal.II/dofs/dof_handler.h>
#include <deal.II/dofs/dof_tools.h>

#include <deal.II/fe/fe_q.h>
#include <deal.II/fe/fe_system.h>

#include <deal.II/grid/grid_generator.h>

#include <deal.II/lac/dynamic_sparsity_pattern.h>
#include <deal.II/lac/precondition.h>
#include <deal.II/lac/solver_cg.h>
#include <deal.II/lac/solver_control.h>
#include <deal.II/lac/sparse_matrix.h>
#include <deal.II/lac/sparsity_pattern.h>

#include <deal.II/numerics/data_out.h>
#include <deal.II/numerics/vector_tools.h>

using namespace dealii;

/**
 * Return stress-strain tensor.
 */
template <int dim>
SymmetricTensor<4, dim>
get_stress_strain_tensor(const double lambda, const double mu)
{
  SymmetricTensor<4, dim> tmp;
  for (unsigned int i = 0; i < dim; ++i)
    for (unsigned int j = 0; j < dim; ++j)
      for (unsigned int k = 0; k < dim; ++k)
        for (unsigned int l = 0; l < dim; ++l)
          tmp[i][j][k][l] = (((i == k) && (j == l) ? mu : 0.0) +
                             ((i == l) && (j == k) ? mu : 0.0) +
                             ((i == j) && (k == l) ? lambda : 0.0));
  return tmp;
}


/**
 * Compute strain.
 */
template <int dim>
inline SymmetricTensor<2, dim>
get_strain(const FEValues<dim> &fe_values,
           const unsigned int   shape_func,
           const unsigned int   q_point)
{
  SymmetricTensor<2, dim> tmp;

  for (unsigned int i = 0; i < dim; ++i)
    tmp[i][i] = fe_values.shape_grad_component(shape_func, q_point, i)[i];

  for (unsigned int i = 0; i < dim; ++i)
    for (unsigned int j = i + 1; j < dim; ++j)
      tmp[i][j] = (fe_values.shape_grad_component(shape_func, q_point, i)[j] +
                   fe_values.shape_grad_component(shape_func, q_point, j)[i]) /
                  2;

  return tmp;
}


int
main()
{
  const unsigned int dim = 2, degree = 1, n_refinements = 0;

  // create mesh, select relevant FEM ingredients, and set up DoFHandler
  Triangulation<dim> tria;
  GridGenerator::subdivided_hyper_rectangle(
    tria, {10, 2}, Point<dim>(0, 0), Point<dim>(1, 0.2), true);
  tria.refine_global(n_refinements);

  FESystem<dim>        fe(FE_Q<dim>(degree), dim);
  QGauss<dim>          quad(degree + 1);
  QGauss<dim - 1>      face_quad(degree + 1);
  MappingQGeneric<dim> mapping(1);

  DoFHandler<dim> dof_handler(tria);
  dof_handler.distribute_dofs(fe);

  // Create constraint matrix
  AffineConstraints<double> constraints;

  // ... fill constraint matrix
  VectorTools::interpolate_boundary_values(
    dof_handler,
    0, // left face
    Functions::ConstantFunction<dim>(std::vector<double>{0.0, 0.0}),
    constraints,
    ComponentMask(std::vector<bool>{true, false}));

  VectorTools::interpolate_boundary_values(
    dof_handler,
    2, // bottom face
    Functions::ConstantFunction<dim>(std::vector<double>{0.0, 0.0}),
    constraints,
    ComponentMask(std::vector<bool>{false, true}));

  // compute traction
  Tensor<1, dim> traction;
  traction[0] = +1e9; // x-direction
  traction[1] = +0e9; // y-direction

  // compute stress strain tensor
  const auto stress_strain_tensor_1 =
    get_stress_strain_tensor<dim>(9.695e10, 7.617e10);

  const auto stress_strain_tensor_2 =
    get_stress_strain_tensor<dim>(10 * 9.695e10, 7.617e10);

  for (const auto &cell : tria.active_cell_iterators())
    cell->set_material_id(cell->center()[0] > 0.5);

  // ... finalize constraint matrix
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
  FEValues<dim> fe_values(mapping,
                          fe,
                          quad,
                          update_gradients | update_JxW_values);

  FEFaceValues<dim> fe_face_values(mapping,
                                   fe,
                                   face_quad,
                                   update_values | update_JxW_values);


  FullMatrix<double>                   cell_matrix;
  Vector<double>                       cell_rhs;
  std::vector<types::global_dof_index> local_dof_indices;

  // loop over all cells
  for (const auto &cell : dof_handler.active_cell_iterators())
    {
      if (cell->is_locally_owned() == false)
        continue;

      fe_values.reinit(cell);

      const unsigned int dofs_per_cell = cell->get_fe().dofs_per_cell;
      cell_matrix.reinit(dofs_per_cell, dofs_per_cell);
      cell_rhs.reinit(dofs_per_cell);

      // loop over cell dofs
      for (unsigned int i = 0; i < dofs_per_cell; ++i)
        for (unsigned int j = 0; j < dofs_per_cell; ++j)
          for (unsigned int q = 0; q < fe_values.n_quadrature_points; ++q)
            {
              const auto eps_phi_i = get_strain(fe_values, i, q);
              const auto eps_phi_j = get_strain(fe_values, j, q);

              cell_matrix(i, j) +=
                (eps_phi_i *
                 (cell->material_id() == 1 ? stress_strain_tensor_1 :
                                             stress_strain_tensor_2) *
                 eps_phi_j) *
                fe_values.JxW(q);
            }

      // loop over all cell faces and their dofs
      for (const auto &face : cell->face_iterators())
        {
          // we only want to apply NBC on the right face
          if (!face->at_boundary() || face->boundary_id() != 1)
            continue;

          fe_face_values.reinit(cell, face);

          for (unsigned int q = 0; q < fe_face_values.n_quadrature_points; ++q)
            for (unsigned int i = 0; i < dofs_per_cell; ++i)
              cell_rhs(i) += fe_face_values.shape_value(i, q) *
                             traction[fe.system_to_component_index(i).first] *
                             fe_face_values.JxW(q);
        }


      local_dof_indices.resize(cell->get_fe().dofs_per_cell);
      cell->get_dof_indices(local_dof_indices);

      constraints.distribute_local_to_global(
        cell_matrix, cell_rhs, local_dof_indices, A, b);
    }

  // solve linear equation system
  ReductionControl         reduction_control;
  SolverCG<Vector<double>> solver(reduction_control);
  solver.solve(A, x, b, PreconditionIdentity());

  printf("Solved in %d iterations.\n", reduction_control.last_step());

  constraints.distribute(x);

  // output results
  DataOut<dim> data_out;
  data_out.attach_dof_handler(dof_handler);
  x.update_ghost_values();
  data_out.add_data_vector(
    dof_handler,
    x,
    "solution",
    std::vector<DataComponentInterpretation::DataComponentInterpretation>(
      dim, DataComponentInterpretation::component_is_part_of_vector));
  data_out.build_patches(mapping, degree + 1);

  std::ofstream output("solution.vtu");
  data_out.write_vtu(output);
}