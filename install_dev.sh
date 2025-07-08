#!/bin/bash
# Development installation script

echo "Installing fortran-analyzer in development mode..."

# Install in development mode
pip install -e .

echo "Installing development dependencies..."
pip install -e ".[dev]"

echo ""
echo "Installation complete!"
echo ""
echo "Usage:"
echo "  fortran-analyzer input.f90"
echo "  fortran-analyzer -m mod_parameters.f90 *.f90"
echo ""
echo "Or use the wrapper script:"
echo "  ./analyze_fortran_vars.sh input.f90"
