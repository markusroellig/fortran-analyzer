# Fortran Analyzer

A comprehensive tool for analyzing Fortran source code to identify variable sources, assignments, and cross-references with global variables.

## Features

- **Procedure-level analysis**: Separate analysis for each subroutine and function
- **Variable scope classification**: Identifies global, local, and imported variables
- **Assignment pattern detection**: Categorizes different types of variable assignments
- **Cross-reference analysis**: Shows relationships between variables across files
- **Flexible output**: Complete lists by default, with optional truncation for large codebases

## Installation

### Via pip (recommended)

```bash
# Install from source
git clone <repository-url>
cd fortran-analyzer
pip install .

# Or install in development mode
pip install -e .
```

### Manual installation

```bash
git clone <repository-url>
cd fortran-analyzer
chmod +x analyze_fortran_vars.sh
```

## Usage

### Command line (after pip installation)

```bash
# Basic analysis
fortran-analyzer input_file.f90

# With custom mod_parameters file
fortran-analyzer -m path/to/mod_parameters.f90 *.f90

# Save output to file with all global variables shown
fortran-analyzer -a -o analysis_report.txt *.f90

# Truncate output for large files
fortran-analyzer --truncate large_file.f90

# Enable debug mode
fortran-analyzer --debug problematic_file.f90
```

### Via wrapper script

```bash
# Use the wrapper script (works with or without pip installation)
./analyze_fortran_vars.sh input_file.f90
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

## Development

### Requirements
- Python 3.6+

### Development Installation
```bash
# Install in development mode with dev dependencies
./install_dev.sh

# Or manually:
pip install -e ".[dev]"
```

### Running Tests
```bash
python -m pytest tests/
```

### Code Formatting
```bash
black fortran_analyzer/
flake8 fortran_analyzer/
mypy fortran_analyzer/
```

## Examples

See the `examples/` directory for sample Fortran files and usage examples.

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Run the test suite
6. Submit a pull request

## License

MIT License - see LICENSE file for details
