# Fortran Analyzer

A comprehensive tool for analyzing Fortran source code to identify variable sources, assignments, and cross-references with global variables.

## Features

- **Procedure-level analysis**: Separate analysis for each subroutine and function
- **Variable scope classification**: Identifies global, local, and imported variables
- **Assignment pattern detection**: Categorizes different types of variable assignments
- **Cross-reference analysis**: Shows relationships between variables across files
- **Flexible output**: Complete lists by default, with optional truncation for large codebases

## Installation

```bash
git clone <repository-url>
cd fortran-analyzer
chmod +x analyze_fortran_vars.sh
```

## Usage

### Basic Analysis
```bash
./analyze_fortran_vars.sh input_file.f90
```

### Advanced Options
```bash
# Analyze with custom mod_parameters file
./analyze_fortran_vars.sh -m path/to/mod_parameters.f90 *.f90

# Save output to file with all global variables shown
./analyze_fortran_vars.sh -a -o analysis_report.txt *.f90

# Truncate output for large files
./analyze_fortran_vars.sh --truncate large_file.f90

# Enable debug mode
./analyze_fortran_vars.sh --debug problematic_file.f90
```

## Command Line Options

- `-h, --help`: Show help message
- `-m, --mod-parameters`: Path to mod_parameters.f90 file
- `-o, --output`: Output file for report
- `-v, --verbose`: Verbose output
- `-a, --show-all-globals`: Show all global variables
- `-e, --enhanced`: Generate enhanced report with statistics
- `-t, --truncate`: Truncate long lists
- `--debug`: Enable debug mode

## Output Analysis

The tool provides:

1. **Global Variables Summary**: Shows all variables from mod_parameters.f90
2. **File-level Analysis**: USE statements and module-level variables
3. **Procedure Analysis**: Individual analysis for each subroutine/function
4. **Cross-reference Analysis**: Global variable usage patterns

## Development

### Requirements
- Python 3.6+
- Bash shell

### Running Tests
```bash
# Run tests (when implemented)
python -m pytest tests/
```

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## License

MIT License - see LICENSE file for details
