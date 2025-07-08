#!/bin/bash
# analyze_fortran_vars.sh - Wrapper script for Fortran variable analysis
# This script can be used when the package is not installed via pip

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Try to use pip-installed version first, fall back to local
if command -v fortran-analyzer &> /dev/null; then
    PYTHON_CMD="fortran-analyzer"
elif [ -f "$SCRIPT_DIR/fortran_analyzer/analyzer.py" ]; then
    PYTHON_CMD="python3 -m fortran_analyzer"
    export PYTHONPATH="$SCRIPT_DIR:$PYTHONPATH"
else
    echo "Error: fortran-analyzer not found. Please install with 'pip install .' or run from source directory"
    exit 1
fi

# Help function
show_help() {
    cat << 'HELP_EOF'
Usage: $0 [OPTIONS] fortran_files...

Analyze Fortran source files for variable sources and dependencies

This is a wrapper script. For direct usage after pip installation, use:
    fortran-analyzer [OPTIONS] fortran_files...

OPTIONS:
    -h, --help              Show this help message
    -m, --mod-parameters    Path to mod_parameters.f90 (default: mod_parameters.f90)
    -o, --output           Output file for report (default: stdout)
    -v, --verbose          Verbose output
    -a, --show-all-globals Show all global variables (not just first 10)
    -e, --enhanced         Generate enhanced report with statistics
    -t, --truncate         Truncate long lists (default: show complete lists)
    --debug                Enable debug mode

EXAMPLES:
    # Analyze a single file
    $0 main.f90
    
    # Analyze with complete output (default)
    $0 input.f90
    
    # Analyze with truncated output for large files
    $0 --truncate large_files/*.f90
    
    # Analyze multiple files with custom mod_parameters
    $0 -m path/to/mod_parameters.f90 src/*.f90
    
    # Save report to file with all globals shown
    $0 -a -o analysis_report.txt src/*.f90

INSTALLATION:
    # Install via pip (recommended)
    pip install .
    
    # Then use directly:
    fortran-analyzer [OPTIONS] files...

HELP_EOF
}

# Check for help flag first
for arg in "$@"; do
    if [[ "$arg" == "-h" || "$arg" == "--help" ]]; then
        show_help
        exit 0
    fi
done

# Check if we have input files (look for non-option arguments)
has_files=false
for arg in "$@"; do
    if [[ ! "$arg" =~ ^- ]]; then
        has_files=true
        break
    fi
done

if ! $has_files; then
    echo "Error: No input files specified"
    show_help
    exit 1
fi

# Run the analysis
echo "Running Fortran variable analysis..."
$PYTHON_CMD "$@"
