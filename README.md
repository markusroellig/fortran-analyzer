# Fortran Analyzer

A comprehensive tool for analyzing Fortran source code to identify variable sources, assignments, and cross-references with global variables.

## Features

- **Whole-codebase scanning**: Scans an entire directory to build a complete map of all modules and global variables before analysis.
- **Procedure-level analysis**: Separate analysis for each subroutine and function.
- **Variable scope classification**: Identifies global, local, and imported variables with codebase-wide context.
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

The analyzer now operates in two passes:
1.  **Discovery Pass**: It first scans a specified codebase directory to find all modules and global variables.
2.  **Analysis Pass**: It then performs a detailed analysis on target files, using the global context from the first pass.

### Command line (after pip installation)

```bash
# Analyze all .f90 files in a directory
# Scans the directory for globals, then analyzes every file.
fortran-analyzer path/to/your/codebase

# Analyze a single file within the context of the whole codebase
# Scans the 'codebase' dir for globals, but only generates a report for 'file_to_analyze.f90'.
fortran-analyzer path/to/your/codebase path/to/your/codebase/file_to_analyze.f90

# Analyze multiple specific files
fortran-analyzer path/to/your/codebase file1.f90 file2.f90

# Save the full report for the entire codebase to a file
fortran-analyzer path/to/your/codebase -o analysis_report.txt

# Show all global variables in the report (not just a summary)
fortran-analyzer path/to/your/codebase --show-all-globals

# Generate a detailed lifecycle "diary" for a specific global variable
fortran-analyzer path/to/your/codebase --trace-var my_global_variable

# Filter the report to only show a specific procedure
fortran-analyzer path/to/your/codebase --show-only-proc my_subroutine
```

### Via wrapper script

```bash
# Use the wrapper script (works with or without pip installation)
./analyze_fortran_vars.sh path/to/your/codebase
```

## Command Line Options

- `codebase_dir`: (Positional) The directory containing the full Fortran codebase to scan for globals.
- `files_to_analyze`: (Optional Positional) Specific Fortran files to analyze in detail. If omitted, all Fortran files in `codebase_dir` are analyzed.
- `-h, --help`: Show help message
- `-o, --output`: Output file for report
- `-v, --verbose`: Verbose output
- `--show-all-globals`: Show all global variables instead of limiting the list.
- `--enhanced`: Generate enhanced report with statistics.
- `--truncate`: Truncate long lists in the report.
- `--debug`: Enable debug mode for more detailed error output.
- `--color`: Control colorized output (`auto`, `always`, `never`).
- `--show-only-file FILENAME [FILENAME ...]`: Only show analysis for specific file(s).
- `--show-only-proc PROCNAME [PROCNAME ...]`: Only show analysis for specific procedure(s).
- `--hide-locals`: Hide local variable declaration lists in procedures.
- `--hide-ok`: Hide procedures that do not read or modify any global variables.
- `--trace-var VARIABLE_NAME`: Generate a detailed lifecycle report for a specific global variable.

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
