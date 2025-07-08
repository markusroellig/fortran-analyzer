#!/bin/bash
# analyze_fortran_vars.sh - Wrapper script for Fortran variable analysis

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON_SCRIPT="$SCRIPT_DIR/fortran_variable_analyzer.py"

# Default values
MOD_PARAMETERS="mod_parameters.f90"
OUTPUT_FILE=""
VERBOSE=""
SHOW_ALL_GLOBALS=""
ENHANCED=""
TRUNCATE=""
DEBUG=""

# Help function
show_help() {
    cat << 'HELP_EOF'
Usage: $0 [OPTIONS] fortran_files...

Analyze Fortran source files for variable sources and dependencies

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
    
    # Use with find to analyze all Fortran files
    find . -name "*.f90" -o -name "*.f" | xargs $0

PATTERNS DETECTED:
    - Parameter declarations (constants)
    - Variable assignments from reads, calculations, literals
    - Array and derived type assignments  
    - Command line argument reads
    - USE statement imports
    - Cross-references with global variables
    - Procedure-level variable analysis

HELP_EOF
}

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        -h|--help)
            show_help
            exit 0
            ;;
        -m|--mod-parameters)
            MOD_PARAMETERS="$2"
            shift 2
            ;;
        -o|--output)
            OUTPUT_FILE="$2"
            shift 2
            ;;
        -v|--verbose)
            VERBOSE="-v"
            shift
            ;;
        -a|--show-all-globals)
            SHOW_ALL_GLOBALS="--show-all-globals"
            shift
            ;;
        -e|--enhanced)
            ENHANCED="--enhanced"
            shift
            ;;
        -t|--truncate)
            TRUNCATE="--truncate"
            shift
            ;;
        --debug)
            DEBUG="--debug"
            shift
            ;;
        -*)
            echo "Unknown option: $1"
            show_help
            exit 1
            ;;
        *)
            break
            ;;
    esac
done

# Check if we have input files
if [[ $# -eq 0 ]]; then
    echo "Error: No input files specified"
    show_help
    exit 1
fi

# Build command
CMD="python3 $PYTHON_SCRIPT"

if [[ -n "$MOD_PARAMETERS" ]]; then
    CMD="$CMD --mod-parameters $MOD_PARAMETERS"
fi

if [[ -n "$OUTPUT_FILE" ]]; then
    CMD="$CMD --output $OUTPUT_FILE"
fi

if [[ -n "$VERBOSE" ]]; then
    CMD="$CMD $VERBOSE"
fi

if [[ -n "$SHOW_ALL_GLOBALS" ]]; then
    CMD="$CMD $SHOW_ALL_GLOBALS"
fi

if [[ -n "$ENHANCED" ]]; then
    CMD="$CMD $ENHANCED"
fi

if [[ -n "$TRUNCATE" ]]; then
    CMD="$CMD $TRUNCATE"
fi

if [[ -n "$DEBUG" ]]; then
    CMD="$CMD $DEBUG"
fi

# Add input files
CMD="$CMD $@"

# Run the analysis
echo "Running Fortran variable analysis..."
eval $CMD
