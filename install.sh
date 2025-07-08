#!/bin/bash
# Installation script

echo "Installing fortran-analyzer..."

# Install the package
pip install .

echo ""
echo "Installation complete!"
echo ""
echo "Usage:"
echo "  fortran-analyzer input.f90"
echo "  fortran-analyzer -m mod_parameters.f90 *.f90"
echo ""
echo "For help:"
echo "  fortran-analyzer --help"
