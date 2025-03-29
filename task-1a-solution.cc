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
// Inport and export meshes.
//
// ------------------------------------------------------------------------

#include <deal.II/grid/tria.h>

// necessary includes
#include <deal.II/grid/grid_in.h>
#include <deal.II/grid/grid_out.h>

#include <deal.II/numerics/data_out.h>

#include <fstream>

using namespace dealii;

const int dim = 2;

int
main()
{
  Triangulation<dim> tria;

  {
    // read mesh with GridIn
    GridIn<dim> grid_in(tria);
    grid_in.read("beam.msh");
  }

  {
    std::ofstream output_1("task-1a-grid.vtk");

    // write mesh with GridOut in VTK format
    GridOut grid_out;
    grid_out.write_vtk(tria, output_1);
  }

  {
    std::ofstream output_2("task-1a-data.vtk");

    // write mesh with DataOut in VTK format
    DataOut<dim> data_out;
    data_out.attach_triangulation(tria);
    data_out.build_patches();
    data_out.write_vtk(output_2);
  }
}
