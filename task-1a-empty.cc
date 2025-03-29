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

// TODO: include the right files

#include <fstream>

using namespace dealii;

const int dim = 2;

int
main()
{
  Triangulation<dim> tria;

  {
    // TODO: read mesh with GridIn
    (void)tria;
  }

  {
    std::ofstream output_1("task-1a-grid.vtk");

    // TODO: write mesh with GridOut in VTK format
    (void)tria;
    (void)output_1;
  }

  {
    std::ofstream output_2("task-1a-data.vtk");

    // TODO: write mesh with DataOut in VTK format
    (void)tria;
    (void)output_2;
  }
}
