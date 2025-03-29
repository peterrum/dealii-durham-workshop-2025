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
// Working with parallel meshes.
//
// ------------------------------------------------------------------------

#include <deal.II/distributed/shared_tria.h>

#include <deal.II/grid/grid_in.h>
#include <deal.II/grid/grid_out.h>

#include <deal.II/numerics/data_out.h>

using namespace dealii;

const int dim = 2;

int
main(int argc, char **argv)
{
  Utilities::MPI::MPI_InitFinalize mpi(argc, argv, 1);

  parallel::shared::Triangulation<dim> tria(
    MPI_COMM_WORLD, Triangulation<dim>::MeshSmoothing::none, true);

  {
    GridIn<dim> grid_in(tria);
    grid_in.read("beam.msh");
  }

  {
    GridOut grid_out;
    grid_out.write_mesh_per_processor_as_vtu(tria, "task-1c");
  }
}
