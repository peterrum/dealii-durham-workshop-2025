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
// Loop over cells and faces and modify properties.
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
    // loop over all cells
    for (const auto &cell : tria.active_cell_iterators())
      {
        // TODO: print cell center
        (void)cell;

        // TODO: set material id
        (void)cell;

        // loop over all faces of the calls that are at the boundary
        for (const auto &face : cell->face_iterators())
          if (face->at_boundary())
            {
              // TODO: set boundary ids
            }
      }
  }

  {
    std::ofstream output_1("task-1b-volume-mesh.vtk");

    // write volume mesh with GridOut in VTK format -> validate material ids
    GridOut grid_out;
    grid_out.set_flags(GridOutFlags::Vtk(true, false, false, true));

    grid_out.write_vtk(tria, output_1);
  }
  {
    std::ofstream output_1("task-1b-surface-mesh.vtk");

    // write surface mesh with GridOut in VTK format -> validate boundary ids
    GridOut grid_out;

    GridOutFlags::Vtk flags;
    grid_out.set_flags(GridOutFlags::Vtk(false, true, false, true));

    grid_out.write_vtk(tria, output_1);
  }
}
