#!/usr/bin/env python3
"""
Fortran Variable Source Analysis Tool

Analyzes Fortran source files to identify variable sources, assignments,
and cross-references with global variables from mod_parameters.f90
Performs separate analysis for each subroutine and function.
"""

import re
import sys
import argparse
from pathlib import Path
from collections import defaultdict, namedtuple
from typing import Dict, List, Set, Tuple, Optional
from datetime import datetime

# Data structures
VariableInfo = namedtuple('VariableInfo', [
    'name', 'type', 'kind', 'dimensions', 'is_parameter', 
    'is_allocatable', 'initial_value', 'module', 'line_num'
])

AssignmentInfo = namedtuple('AssignmentInfo', [
    'variable', 'line_num', 'assignment_type', 'rhs', 'context'
])

CallInfo = namedtuple('CallInfo', [
    'called_name', 'line_num', 'call_type'
])

ProcedureInfo = namedtuple('ProcedureInfo', [
    'name', 'type', 'start_line', 'end_line', 'module', 'arguments', 'calls'
])

class Color:
    """ANSI color codes for terminal output"""
    HEADER = '\033[95m'
    BLUE = '\033[94m'
    CYAN = '\033[96m'
    GREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'

def cprint(text, color, force_color: bool = False, **kwargs):
    """Prints text in color if stdout is a TTY or color is forced."""
    if force_color or sys.stdout.isatty():
        print(f"{color}{text}{Color.ENDC}", **kwargs)
    else:
        print(text, **kwargs)

# Copy the entire FortranVariableAnalyzer class here from your original file
# (The class definition remains the same)
class FortranVariableAnalyzer:
    def __init__(self):
        self.global_variables = {}
        self.modules = defaultdict(list)
        self.module_locations = {}

        # File-level structures
        self.file_variables = defaultdict(list)
        self.file_assignments = defaultdict(list)
        self.file_reads = defaultdict(list)
        self.use_statements = defaultdict(list)
        self.file_procedures = defaultdict(list)

        # Procedure-level structures
        self.procedure_variables = defaultdict(lambda: defaultdict(list))
        self.procedure_assignments = defaultdict(lambda: defaultdict(list))
        self.procedure_reads = defaultdict(lambda: defaultdict(list))
        self.procedure_calls = defaultdict(lambda: defaultdict(list))
        self.procedure_use_statements = defaultdict(lambda: defaultdict(list))

    def scan_directory_for_globals(self, source_dir: Path):
        """
        Pass 1: A fast scan of all files in a directory to find all module-level
        (global) variables.
        """
        cprint(f"Scanning {source_dir} for global variables...", Color.BOLD)
        fortran_files = []
        extensions = ['*.f90', '*.f', '*.F90', '*.F']
        for ext in extensions:
            fortran_files.extend(source_dir.rglob(ext))

        for file_path in fortran_files:
            try:
                with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                    content = f.read()
            except Exception as e:
                print(f"Warning: Could not read {file_path} during scan: {e}")
                continue
            
            self._discover_globals_in_content(content, file_path)
        
        cprint(f"Found {len(self.global_variables)} global variables in {len(self.modules)} modules.", Color.GREEN)

    def _discover_globals_in_content(self, content: str, file_path: Path):
        """Helper for scan_directory_for_globals to parse content."""
        current_module = None
        lines = content.split('\n')
        
        line_buffer = ""
        for line_num, line in enumerate(lines, 1):
            clean_original = self.clean_line(line)
            
            # Handle line continuations correctly
            if clean_original.endswith('&'):
                line_buffer += clean_original[:-1].rstrip() + " "
                continue
            else:
                line_buffer += clean_original

            line_clean = line_buffer
            line_buffer = "" # Reset buffer
            line_lower = line_clean.lower()

            if not line_clean:
                continue

            module_match = re.match(r'^\s*module\s+(\w+)', line_lower)
            if module_match and not line_lower.startswith('end module'):
                current_module = module_match.group(1)
                if current_module not in self.module_locations:
                    self.module_locations[current_module] = file_path.name
                continue

            if re.match(r'^\s*end\s+module', line_lower):
                current_module = None
                continue
            
            # Only look for globals if we are inside a module
            if current_module:
                # We are only interested in declarations, not procedure bodies
                proc_match = re.match(r'^\s*(subroutine|function|contains)', line_lower)
                if proc_match:
                    current_module = None # Stop parsing this module
                    continue

                var_infos = self.parse_variable_declaration(line_clean, line_num, current_module)
                for var_info in var_infos:
                    if var_info.name not in self.global_variables:
                        self.global_variables[var_info.name] = var_info
                    if var_info.name not in self.modules[current_module]:
                        self.modules[current_module].append(var_info.name)

    def clean_line(self, line: str) -> str:
        """Remove comments and clean up line"""
        # Remove comments (but be careful about quotes)
        in_quotes = False
        quote_char = None
        clean_line = ""

        for i, char in enumerate(line):
            if char in ('"', "'") and not in_quotes:
                in_quotes = True
                quote_char = char
            elif char == quote_char and in_quotes:
                in_quotes = False
                quote_char = None
            elif char == '!' and not in_quotes:
                # Found comment, stop here
                break
            clean_line += char

        return clean_line.strip()

    def parse_procedures(self, filename: str, content: str) -> List[ProcedureInfo]:
        """Parse procedures (subroutines and functions) from file content"""
        procedures = []
        lines = content.split('\n')
        current_module = None

        for i, line in enumerate(lines, 1):
            line_clean = self.clean_line(line)
            line_lower = line_clean.lower()

            # Track module context
            module_match = re.match(r'^\s*module\s+(\w+)', line_lower)
            if module_match and not line_lower.startswith('end module'):
                current_module = module_match.group(1)
                continue

            if re.match(r'^\s*end\s+module', line_lower):
                current_module = None
                continue

            # Find procedure starts
            proc_match = re.match(r'^\s*(subroutine|function)\s+(\w+)\s*(\([^)]*\))?', line_lower)
            if proc_match:
                proc_type = proc_match.group(1)
                proc_name = proc_match.group(2)
                arguments = proc_match.group(3) if proc_match.group(3) else '()'

                # Find the end of this procedure
                end_line = self.find_procedure_end(lines, i, proc_type, proc_name)

                procedures.append(ProcedureInfo(
                    name=proc_name,
                    type=proc_type,
                    start_line=i,
                    end_line=end_line,
                    module=current_module,
                    arguments=arguments,
                    calls=[] # Initialize calls list
                ))

        return procedures

    def find_procedure_end(self, lines: List[str], start_line: int, proc_type: str, proc_name: str) -> int:
        """Find the end line of a procedure"""
        nested_level = 0

        for i in range(start_line, len(lines)):
            line_clean = self.clean_line(lines[i])
            line_lower = line_clean.lower()

            # Check for nested procedures
            if re.match(r'^\s*(subroutine|function)\s+\w+', line_lower):
                if i > start_line - 1:  # Don't count the starting procedure
                    nested_level += 1

            # Check for end statements
            end_match = re.match(r'^\s*end\s*(subroutine|function)?(\s+(\w+))?\s*$', line_lower)
            if end_match:
                end_type = end_match.group(1)
                end_name = end_match.group(3)

                # If it's a generic 'end' or matches our procedure
                if (not end_type or end_type == proc_type) and \
                   (not end_name or end_name == proc_name.lower()):
                    if nested_level == 0:
                        return i + 1  # Convert to 1-based line number
                    else:
                        nested_level -= 1

        return len(lines)  # If no end found, assume end of file

    def analyze_file(self, filename: str):
        """Analyze a single Fortran source file with procedure-level analysis"""
        print(f"Analyzing {filename}...")

        try:
            with open(filename, 'r', encoding='utf-8', errors='ignore') as f:
                content = f.read()
        except Exception as e:
            print(f"Error reading {filename}: {e}")
            return

        lines = content.split('\n')

        # First, identify all procedures in the file
        procedures = self.parse_procedures(filename, content)
        self.file_procedures[filename] = procedures

        # Analyze file-level and procedure-level content
        current_module = None
        current_procedure = None
        
        line_buffer = ""
        line_idx = 0
        start_line_num = 1
        while line_idx < len(lines):
            i = line_idx + 1
            original_line = lines[line_idx]
            line_idx += 1

            if not line_buffer:
                start_line_num = i

            # Handle line continuations correctly
            clean_original = self.clean_line(original_line)
            if clean_original.endswith('&'):
                line_buffer += clean_original[:-1].rstrip() + " "
                continue
            else:
                line_buffer += clean_original

            line_clean = line_buffer
            line_buffer = "" # Reset buffer
            line_lower = line_clean.lower()

            # Skip empty lines
            if not line_clean:
                continue

            # Track module context
            module_match = re.match(r'^\s*module\s+(\w+)', line_lower)
            if module_match and not line_lower.startswith('end module'):
                current_module = module_match.group(1)
                current_procedure = None
                continue

            if re.match(r'^\s*end\s+module', line_lower):
                current_module = None
                current_procedure = None
                continue

            # Track procedure context
            proc_match = re.match(r'^\s*(subroutine|function)\s+(\w+)', line_lower)
            if proc_match:
                proc_name = proc_match.group(2)
                # Find the corresponding procedure info. The line number check is removed
                # because parse_procedures doesn't handle continuations, but the main
                # loop does, causing a mismatch in start_line_num.
                for proc in procedures:
                    if proc.name.lower() == proc_name:
                        # This assumes procedure names are unique within a file, which is a
                        # reasonable assumption for a first-pass fix.
                        current_procedure = proc
                        break
                continue

            # Check if we're at the end of a procedure
            if current_procedure and start_line_num >= current_procedure.end_line:
                current_procedure = None

            # Determine analysis scope (file-level vs procedure-level)
            scope_key = self.get_scope_key(filename, current_procedure)

            # Parse USE statements
            use_match = re.match(r'^\s*use\s+(\w+)(?:\s*,\s*only\s*:\s*(.+))?', line_lower, re.DOTALL)
            if use_match:
                module_name = use_match.group(1)
                only_list = use_match.group(2) if use_match.group(2) else 'all'
                if only_list != 'all':
                    only_list = ' '.join(only_list.split())

                if current_procedure:
                    self.procedure_use_statements[filename][current_procedure.name].append((module_name, only_list, start_line_num))
                else:
                    self.use_statements[filename].append((module_name, only_list, start_line_num))
                continue

            # Parse variable declarations
            var_infos = self.parse_variable_declaration(line_clean, start_line_num, current_module or 'local')
            if current_procedure:
                self.procedure_variables[filename][current_procedure.name].extend(var_infos)
            else:
                self.file_variables[filename].extend(var_infos)
                # If we are in a module but not a procedure, these are global variables
                if current_module:
                    for var_info in var_infos:
                        if var_info.name not in self.global_variables:
                            self.global_variables[var_info.name] = var_info
                        if var_info.name not in self.modules[current_module]:
                            self.modules[current_module].append(var_info.name)


            # Parse assignments
            assignments = self.parse_assignments(line_clean, start_line_num, filename)
            if current_procedure:
                self.procedure_assignments[filename][current_procedure.name].extend(assignments)
            else:
                self.file_assignments[filename].extend(assignments)

            # Parse variable reads
            reads = self.parse_variable_reads(line_clean, start_line_num, filename)
            if current_procedure:
                self.procedure_reads[filename][current_procedure.name].extend(reads)
            else:
                self.file_reads[filename].extend(reads)

            # Parse procedure calls
            if current_procedure:
                calls = self.parse_procedure_calls(line_clean, start_line_num)
                self.procedure_calls[filename][current_procedure.name].extend(calls)

    def get_scope_key(self, filename: str, procedure: Optional[ProcedureInfo]) -> str:
        """Generate a scope key for organizing data"""
        if procedure:
            return f"{filename}::{procedure.name}"
        else:
            return f"{filename}::module_level"

    def parse_variable_declaration(self, line: str, line_num: int, module: str) -> List[VariableInfo]:
        """Parse a single variable declaration line - returns list to handle multiple variables"""
        line_clean = line.strip()
        line_lower = line_clean.lower()
        result = []

        # Skip certain lines
        if (line_lower.startswith('implicit') or line_lower.startswith('save') or
            line_lower.startswith('contains') or line_lower.startswith('use') or
            line_lower.startswith('private') or line_lower.startswith('public') or
            line_lower.startswith('procedure') or line_lower.startswith('class') or
            '::' not in line_clean):
            return result

        try:
            # Parameter declarations (special case)
            param_match = re.search(r'parameter\s*\(\s*(\w+)\s*=\s*(.+?)\s*\)', line_lower)
            if param_match:
                result.append(VariableInfo(
                    name=param_match.group(1),
                    type='parameter',
                    kind='',
                    dimensions='',
                    is_parameter=True,
                    is_allocatable=False,
                    initial_value=param_match.group(2),
                    module=module,
                    line_num=line_num
                ))
                return result

            # Modern Fortran style: type :: variable
            if '::' in line_clean:
                parts = line_clean.split('::', 1)
                if len(parts) == 2:
                    type_part = parts[0].strip()
                    var_part = parts[1].strip()

                    # Parse type information
                    var_type, kind, attributes = self.parse_type_specification(type_part)

                    if var_type:
                        # Parse attributes
                        is_parameter = 'parameter' in attributes.lower()
                        is_allocatable = 'allocatable' in attributes.lower()

                        # Parse variable names and dimensions
                        variables = self.parse_variable_list(var_part)

                        for var_name, dimensions, initial_value in variables:
                            if var_name:  # Make sure we have a valid variable name
                                result.append(VariableInfo(
                                    name=var_name,
                                    type=var_type,
                                    kind=kind,
                                    dimensions=dimensions,
                                    is_parameter=is_parameter,
                                    is_allocatable=is_allocatable,
                                    initial_value=initial_value,
                                    module=module,
                                    line_num=line_num
                                ))

        except Exception as e:
            # If parsing fails, just skip this line
            if '--debug' in sys.argv:
                print(f"Warning: Could not parse line {line_num}: {line_clean[:50]}... ({e})")
            pass

        return result

    def parse_type_specification(self, type_part: str) -> Tuple[str, str, str]:
        """Parse the type specification part before ::"""
        type_part_lower = type_part.lower()

        # Extract base type
        type_match = re.search(r'\b(integer|real|double\s+precision|complex|logical|character|type)\b', type_part_lower)
        if not type_match:
            return None, '', ''

        var_type = type_match.group(1).replace(' ', '_')

        # Extract kind specification
        kind = ''
        kind_match = re.search(r'\(([^)]*)\)', type_part)
        if kind_match:
            kind = kind_match.group(1)

        # Extract attributes (everything else)
        attributes = type_part

        return var_type, kind, attributes

    def parse_variable_list(self, var_part: str) -> List[Tuple[str, str, str]]:
        """Parse variable list from declaration"""
        variables = []

        # Split by commas, but be careful about parentheses and strings
        parts = self.smart_split(var_part, ',')

        for part in parts:
            part = part.strip()
            if not part:
                continue

            # Check for initialization
            initial_value = ''
            if '=' in part:
                var_part_split = part.split('=', 1)
                var_part_name = var_part_split[0].strip()
                initial_value = var_part_split[1].strip()
            else:
                var_part_name = part

            # Check for dimensions
            dimensions = ''
            if '(' in var_part_name:
                paren_idx = var_part_name.index('(')
                var_name = var_part_name[:paren_idx].strip()
                dimensions = var_part_name[paren_idx:].strip()
            else:
                var_name = var_part_name.strip()

            # Clean up variable name - only keep alphanumeric and underscore
            var_name_clean = re.match(r'^([a-zA-Z_]\w*)', var_name)
            if var_name_clean:
                var_name = var_name_clean.group(1)

                if var_name:  # Only add if we have a valid variable name
                    variables.append((var_name, dimensions, initial_value))

        return variables

    def smart_split(self, text: str, delimiter: str) -> List[str]:
        """Split text by delimiter, respecting parentheses and quotes"""
        parts = []
        current = ''
        paren_depth = 0
        in_quotes = False
        quote_char = None

        for char in text:
            if char in ('"', "'") and not in_quotes:
                in_quotes = True
                quote_char = char
            elif char == quote_char and in_quotes:
                in_quotes = False
                quote_char = None
            elif not in_quotes:
                if char == '(':
                    paren_depth += 1
                elif char == ')':
                    paren_depth -= 1
                elif char == delimiter and paren_depth == 0:
                    parts.append(current)
                    current = ''
                    continue

            current += char

        if current:
            parts.append(current)

        return parts

    def parse_assignments(self, line: str, line_num: int, filename: str) -> List[AssignmentInfo]:
        """Enhanced assignment parsing with better pattern detection"""
        assignments = []
        line_clean = line.strip()
        line_lower = line_clean.lower()

        # Skip control structures and I/O statements
        skip_patterns = [
            r'^\s*if\s*\(',
            r'^\s*where\s*\(',
            r'^\s*forall\s*\(',
            r'^\s*select\s+case',
            r'^\s*write\s*\(',
            r'^\s*print\s*[\*\(]',
            r'^\s*call\s+\w+',
            r'^\s*read\s*\(',
            r'==|/=|<=|>=|\.eq\.|\.ne\.|\.le\.|\.ge\.|\.lt\.|\.gt\.',
        ]

        for pattern in skip_patterns:
            if re.search(pattern, line_lower):
                return assignments

        # Enhanced assignment patterns
        assignment_patterns = [
            # Simple assignment: var = value
            r'^(\w+(?:\([^)]*\))?(?:%\w+)*)\s*=\s*(.+)$',
            # Array assignment: var(indices) = value
            r'^(\w+\([^)]*\))\s*=\s*(.+)$',
            # Derived type assignment: var%component = value
            r'^(\w+%\w+(?:%\w+)*)\s*=\s*(.+)$',
        ]

        for pattern in assignment_patterns:
            match = re.match(pattern, line_clean)
            if match:
                lhs = match.group(1).strip()
                rhs = match.group(2).strip()

                # Extract base variable name
                var_name_match = re.match(r'^(\w+)', lhs)
                if var_name_match:
                    var_name = var_name_match.group(1)
                    assignment_type = self.classify_assignment_enhanced(lhs, rhs, line_clean)

                    assignments.append(AssignmentInfo(
                        variable=var_name,
                        line_num=line_num,
                        assignment_type=assignment_type,
                        rhs=rhs,
                        context=lhs
                    ))
                    break

        return assignments

    def classify_assignment_enhanced(self, lhs: str, rhs: str, full_line: str) -> str:
        """Enhanced assignment classification"""
        rhs_lower = rhs.lower()

        # I/O operations
        if any(pattern in rhs_lower for pattern in ['read(', 'get_command_argument', 'command_argument_count']):
            return 'INPUT_READ'

        # Mathematical operations
        if any(func in rhs_lower for func in ['sqrt', 'exp', 'log', 'sin', 'cos', 'tan', 'abs', 'max', 'min', 'sum', 'product']):
            return 'CALCULATED'

        # Arithmetic expressions
        if re.search(r'[\+\-\*/]', rhs) and not re.search(r'^[\+\-]?\d+\.?\d*$', rhs):
            return 'CALCULATED'

        # Literal values
        if re.search(r'^\s*[\d\.\-\+eE_]+\w*\s*$', rhs) or rhs_lower in ['true', 'false', '.true.', '.false.']:
            return 'LITERAL_ASSIGNMENT'

        # String literals
        if re.search(r'^["\'].*["\']$', rhs):
            return 'STRING_LITERAL'

        # Configuration/parameter access
        if 'config%' in rhs_lower or '%' in rhs:
            return 'CONFIG_ACCESS'

        # Array operations
        if '(' in lhs and ')' in lhs:
            return 'ARRAY_ASSIGNMENT'

        # Derived type operations
        if '%' in lhs:
            return 'DERIVED_TYPE_ASSIGNMENT'

        return 'VARIABLE_ASSIGNMENT'

    def parse_variable_reads(self, line: str, line_num: int, filename: str) -> List[Tuple[str, int, str]]:
        """Parse variable reads (usage) from a line - improved version"""
        reads = []
        line_clean = line.strip()

        # Skip empty lines and comments
        if not line_clean or line_clean.startswith('!'):
            return reads

        # Skip certain Fortran constructs
        line_lower = line_clean.lower()
        skip_patterns = [
            r'^\s*use\s+',
            r'^\s*implicit\s+',
            r'^\s*parameter\s*\(',
            r'^\s*dimension\s+',
            r'^\s*allocatable\s+',
            r'^\s*intent\s*\(',
            r'^\s*(subroutine|function|end|contains|module|program)',
            r'^\s*(public|private|save)',
        ]

        for pattern in skip_patterns:
            if re.search(pattern, line_lower):
                return reads

        # For assignment lines, only look at RHS
        if '=' in line_clean and not line_lower.startswith('if'):
            # Split on first = to get RHS
            parts = line_clean.split('=', 1)
            if len(parts) > 1:
                search_text = parts[1]
            else:
                return reads
        else:
            search_text = line_clean

        # Extract variables more carefully
        # Remove function calls first
        search_text = re.sub(r'\b\w+\s*\([^)]*\)', '', search_text)

        # Find variable names
        var_pattern = r'\b([a-zA-Z_]\w*)\b'

        fortran_keywords = {
            'if', 'then', 'else', 'endif', 'end', 'do', 'while', 'call',
            'function', 'subroutine', 'return', 'stop', 'print', 'write',
            'read', 'open', 'close', 'sqrt', 'exp', 'log', 'sin', 'cos',
            'abs', 'max', 'min', 'real', 'integer', 'logical', 'character',
            'parameter', 'dimension', 'allocatable', 'intent', 'kind',
            'selected_real_kind', 'selected_int_kind', 'true', 'false',
            'dp', 'sp', 'only', 'use', 'implicit', 'none', 'save',
            'public', 'private', 'module', 'program', 'contains',
            'where', 'forall', 'select', 'case', 'cycle', 'exit',
            '__date__', '__file__', '__line__', '__time__', '_dp'
        }

        for match in re.finditer(var_pattern, search_text):
            var_name = match.group(1).lower()

            # Skip keywords, very short names, and obvious non-variables
            if (var_name not in fortran_keywords and
                len(var_name) > 1 and
                not var_name.startswith('_') and
                not var_name.isdigit()):
                reads.append((var_name, line_num, 'VARIABLE_READ'))

        return reads

    def parse_procedure_calls(self, line: str, line_num: int) -> List[CallInfo]:
        """Parse procedure calls from a line."""
        calls = []
        line_clean = line.strip()
        line_lower = line_clean.lower()

        # Pattern for 'call subroutine_name(...)'
        call_match = re.match(r'^\s*call\s+([a-zA-Z_]\w*)', line_lower)
        if call_match:
            calls.append(CallInfo(
                called_name=call_match.group(1),
                line_num=line_num,
                call_type='subroutine'
            ))

        # Pattern for function calls: name(...)
        # This is a simple heuristic and may misidentify array access as function calls.
        # It avoids matching names on the LHS of an assignment.
        search_text = line_clean
        if '=' in search_text and not search_text.lstrip().lower().startswith('if'):
             search_text = search_text.split('=', 1)[1]

        func_matches = re.findall(r'\b([a-zA-Z_]\w*)\s*\(', search_text)
        if func_matches:
            # A simple filter to avoid common intrinsic functions
            fortran_intrinsics = {'dble', 'real', 'int', 'abs', 'mod', 'max', 'min', 'sin', 'cos', 'tan', 'exp', 'log', 'sqrt', 'trim', 'adjustl', 'len', 'anint', 'log10', 'dabs', 'dsin', 'allocated'}
            for func_name in func_matches:
                if func_name.lower() not in fortran_intrinsics:
                    calls.append(CallInfo(
                        called_name=func_name,
                        line_num=line_num,
                        call_type='function'
                    ))
        return calls

    def generate_report(self, output_file: str = None, show_all_globals: bool = False, truncate_lists: bool = False, enhanced: bool = False, show_only_file: Optional[List[str]] = None, show_only_proc: Optional[List[str]] = None, hide_locals: bool = False, hide_ok: bool = False, color_mode: str = 'auto'):
        """Generate comprehensive analysis report with procedure-level analysis"""
        report_lines = []
        
        if color_mode == 'always':
            use_color = True
        elif color_mode == 'never':
            use_color = False
        else: # auto
            use_color = sys.stdout.isatty() and output_file is None

        def add_line(line: str = '', indent: int = 0, color: Optional[str] = None):
            if use_color and color:
                report_lines.append('  ' * indent + f"{color}{line}{Color.ENDC}")
            else:
                report_lines.append('  ' * indent + line)

        add_line("FORTRAN VARIABLE SOURCE ANALYSIS REPORT", color=Color.HEADER)
        add_line("=" * 50, color=Color.HEADER)
        add_line(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        add_line(f"Working Directory: {Path.cwd()}")
        add_line()

        # --- Pre-computation for enhanced call graph ---
        proc_to_file_map = {}
        for fname, procs in self.file_procedures.items():
            for p in procs:
                proc_to_file_map[p.name.lower()] = fname
        global_var_names_lower = {name.lower() for name in self.global_variables.keys()}
        # --- End pre-computation ---

        # Global Hotspot Summary (if enhanced)
        if enhanced:
            self._generate_global_hotspot_summary(add_line, use_color)

        # Global Variables Summary
        self._generate_global_summary(add_line, show_all_globals, use_color)

        # File-by-file analysis with procedure breakdown
        for filename in sorted(self.file_variables.keys()):
            # Filtering logic for --show-only-file
            if show_only_file and Path(filename).name not in show_only_file:
                continue

            add_line(f"FILE ANALYSIS: {Path(filename).name}", color=Color.HEADER)
            add_line("=" * 80, color=Color.HEADER)

            # Generate and add the high-level summary for the file
            self._generate_file_summary(filename, add_line, use_color)

            # File-level USE statements
            if filename in self.use_statements:
                add_line("FILE-LEVEL USE Statements:", 1, color=Color.BLUE)
                for module, only_list, line_num in self.use_statements[filename]:
                    add_line(f"Line {line_num:4d}: USE {module}", 2)
                    if only_list != 'all':
                        only_clean = only_list.replace('\n', ' ').replace('\t', ' ')
                        add_line(f"            ONLY: {only_clean}", 2)
                add_line()

            # File-level variable declarations
            if filename in self.file_variables and self.file_variables[filename]:
                add_line("FILE-LEVEL Variable Declarations:", 1, color=Color.BLUE)
                limit = 15 if truncate_lists else len(self.file_variables[filename])
                for var_info in self.file_variables[filename][:limit]:
                    scope_status = self.classify_variable_scope(var_info.name, filename)
                    line_str = self._format_variable_line(var_info, scope_status, use_color)
                    add_line(line_str, 2)

                if truncate_lists and len(self.file_variables[filename]) > 15:
                    add_line(f"  ... and {len(self.file_variables[filename]) - 15} more declarations", 2, color=Color.CYAN)
                add_line()

            # Procedure-by-procedure analysis
            if filename in self.file_procedures:
                procedures = self.file_procedures[filename]
                if procedures:
                    add_line(f"PROCEDURES FOUND: {len(procedures)}", 1, color=Color.BLUE)
                    add_line()

                    for i, proc in enumerate(procedures):
                        if i > 0:
                            add_line("." * 80, 1, color=Color.CYAN)
                            add_line()
                            
                        # --- Start of filtering logic for this procedure ---

                        # Global variable usage within the procedure
                        global_var_names_lower = {name.lower() for name in self.global_variables.keys()}

                        # Globals Read in this procedure
                        proc_reads = self.procedure_reads.get(filename, {}).get(proc.name, [])
                        read_globals_in_proc = sorted(list({var_name.lower() for var_name, _, _ in proc_reads if var_name.lower() in global_var_names_lower}))
                        
                        # Globals Modified in this procedure
                        proc_assignments = self.procedure_assignments.get(filename, {}).get(proc.name, [])
                        modified_globals_in_proc = sorted(list({assign.variable.lower() for assign in proc_assignments if assign.variable.lower() in global_var_names_lower}))

                        # Filtering logic for --hide-ok
                        if hide_ok and not read_globals_in_proc and not modified_globals_in_proc:
                            continue # Skip this procedure from the report

                        # Filtering logic for --show-only-proc
                        if show_only_proc and proc.name.lower() not in [p.lower() for p in show_only_proc]:
                            continue # Skip this procedure if it's not in the requested list
                        
                        # --- End of filtering logic ---

                        proc_header = f"{proc.type.upper()}: {proc.name} (lines {proc.start_line}-{proc.end_line})"
                        add_line(proc_header, 1, color=Color.GREEN)
                        add_line("-" * len(proc_header), 1, color=Color.GREEN)

                        if proc.module:
                            add_line(f"Module: {proc.module}", 2)
                        add_line(f"Arguments: {proc.arguments}", 2)
                        add_line()

                        # Procedure-level USE statements
                        if (filename in self.procedure_use_statements and
                            proc.name in self.procedure_use_statements[filename]):
                            add_line("USE Statements:", 2, color=Color.CYAN)
                            for module, only_list, line_num in self.procedure_use_statements[filename][proc.name]:
                                add_line(f"Line {line_num:4d}: USE {module}", 3)
                                if only_list != 'all':
                                    only_clean = only_list.replace('\n', ' ').replace('\t', ' ')
                                    add_line(f"            ONLY: {only_clean}", 3)
                            add_line()

                        # Procedure-level variable declarations
                        if (filename in self.procedure_variables and
                            proc.name in self.procedure_variables[filename] and not hide_locals):
                            vars_list = self.procedure_variables[filename][proc.name]
                            if vars_list:
                                add_line("Local Variable Declarations:", 2, color=Color.CYAN)
                                limit = 15 if truncate_lists else len(vars_list)
                                for var_info in vars_list[:limit]:
                                    scope_status = self.classify_variable_scope(var_info.name, filename, proc.name)
                                    line_str = self._format_variable_line(var_info, scope_status, use_color)
                                    add_line(line_str, 3)

                                if truncate_lists and len(vars_list) > 15:
                                    add_line(f"  ... and {len(vars_list) - 15} more declarations", 3, color=Color.CYAN)
                                add_line()

                        # Procedure-level assignments
                        if (filename in self.procedure_assignments and
                            proc.name in self.procedure_assignments[filename]):
                            assignments = self.procedure_assignments[filename][proc.name]
                            if assignments:
                                add_line("Variable Assignments:", 2, color=Color.CYAN)
                                assignment_counts = defaultdict(int)
                                limit = 15 if truncate_lists else len(assignments)

                                for assignment in assignments[:limit]:
                                    assignment_counts[assignment.assignment_type] += 1
                                    scope_status = self.classify_variable_scope(assignment.variable, filename, proc.name)
                                    scope_str = f"[{scope_status}]"
                                    if use_color:
                                        scope_color = Color.GREEN if "GLOBAL" in scope_status else Color.ENDC
                                        scope_str = f"[{scope_color}{scope_status}{Color.ENDC}]"


                                    rhs_display = assignment.rhs[:30] + "..." if len(assignment.rhs) > 30 else assignment.rhs
                                    add_line(f"Line {assignment.line_num:4d}: {assignment.variable:<18} = {rhs_display:<33} [{assignment.assignment_type[:12]}] {scope_str}", 3)

                                if truncate_lists and len(assignments) > 15:
                                    add_line(f"  ... and {len(assignments) - 15} more assignments", 3, color=Color.CYAN)

                                add_line()
                                add_line("Assignment Type Summary:", 2, color=Color.CYAN)
                                for assign_type, count in assignment_counts.items():
                                    add_line(f"  {assign_type:<20}: {count}", 3)
                                add_line()

                        # Procedure-level variable reads
                        if (filename in self.procedure_reads and
                            proc.name in self.procedure_reads[filename]):
                            reads = self.procedure_reads[filename][proc.name]
                            if reads:
                                read_counts = defaultdict(int)
                                for var_name, line_num, read_type in reads:
                                    read_counts[var_name] += 1

                                add_line("Most Frequently Read Variables:", 2, color=Color.CYAN)
                                sorted_reads = sorted(read_counts.items(), key=lambda x: x[1], reverse=True)
                                limit = 12 if truncate_lists else len(sorted_reads)
                                for var_name, count in sorted_reads[:limit]:
                                    scope_status = self.classify_variable_scope(var_name, filename, proc.name)
                                    scope_str = f"[{scope_status}]"
                                    if use_color:
                                        scope_color = Color.GREEN if "GLOBAL" in scope_status else Color.ENDC
                                        scope_str = f"[{scope_color}{scope_status}{Color.ENDC}]"
                                    add_line(f"{var_name:<20}: {count:3d} reads {scope_str}", 3)
                                add_line()

                        # Procedure-level calls
                        if (filename in self.procedure_calls and
                            proc.name in self.procedure_calls[filename]):
                            calls = self.procedure_calls[filename][proc.name]
                            if calls:
                                add_line("Procedure Calls:", 2, color=Color.CYAN)
                                called_procs_info = defaultdict(lambda: {'reads': set(), 'modifies': set()})

                                for call in calls:
                                    called_name_lower = call.called_name.lower()
                                    called_filename = proc_to_file_map.get(called_name_lower)

                                    if called_filename:
                                        # Globals Read in called procedure
                                        called_reads = self.procedure_reads.get(called_filename, {}).get(called_name_lower, [])
                                        read_globals = {var.lower() for var, _, _ in called_reads if var.lower() in global_var_names_lower}
                                        called_procs_info[called_name_lower]['reads'].update(read_globals)

                                        # Globals Modified in called procedure
                                        called_assignments = self.procedure_assignments.get(called_filename, {}).get(called_name_lower, [])
                                        modified_globals = {assign.variable.lower() for assign in called_assignments if assign.variable.lower() in global_var_names_lower}
                                        called_procs_info[called_name_lower]['modifies'].update(modified_globals)
                                
                                for called_name, data_flow in sorted(called_procs_info.items()):
                                    flow_parts = []
                                    if data_flow['modifies']:
                                        mod_str = f"Modifies: {', '.join(sorted(list(data_flow['modifies'])))}"
                                        if use_color:
                                            flow_parts.append(f"{Color.FAIL}{mod_str}{Color.ENDC}")
                                        else:
                                            flow_parts.append(mod_str)
                                    
                                    if data_flow['reads']:
                                        read_str = f"Reads: {', '.join(sorted(list(data_flow['reads'])))}"
                                        if use_color:
                                            flow_parts.append(f"{Color.WARNING}{read_str}{Color.ENDC}")
                                        else:
                                            flow_parts.append(read_str)
                                    
                                    flow_display = f" ({'; '.join(flow_parts)})" if flow_parts else ""
                                    add_line(f"- {called_name}{flow_display}", 3)

                                add_line()

                        # Global variable usage within the procedure (already calculated above)
                        if read_globals_in_proc:
                            add_line("Global Variables Read:", 2, color=Color.WARNING)
                            for line in self._format_list_in_columns(read_globals_in_proc):
                                add_line(line, 3)
                            add_line()

                        if modified_globals_in_proc:
                            add_line("Global Variables Modified:", 2, color=Color.FAIL)
                            for line in self._format_list_in_columns(modified_globals_in_proc):
                                add_line(line, 3)
                            add_line()

                        add_line()

            add_line()

        # Cross-reference analysis
        self._generate_cross_reference_analysis(add_line, use_color)

        # Call Graph Analysis
        self._generate_call_graph_analysis(add_line, use_color)

        report_text = '\n'.join(report_lines)

        if output_file:
            # Strip color codes when writing to a file
            report_text_no_color = re.sub(r'\033\[[0-9;]*m', '', report_text)
            with open(output_file, 'w') as f:
                f.write(report_text_no_color)
            cprint(f"Report written to {output_file}", Color.GREEN)
        else:
            print(report_text)

    def _format_list_in_columns(self, items: List[str], num_columns: int = 4, col_width: int = 22) -> List[str]:
        """Formats a list of strings into a multi-column layout."""
        if not items:
            return []

        # Pad list to be a multiple of num_columns for even distribution
        padded_items = items + [""] * ((num_columns - len(items) % num_columns) % num_columns)
        num_rows = len(padded_items) // num_columns
        
        lines = []
        for r in range(num_rows):
            row_items = []
            for c in range(num_columns):
                # Get item using column-major order for better balancing
                item_index = c * num_rows + r
                if item_index < len(padded_items):
                    row_items.append(padded_items[item_index])
            
            # Format the row
            line = "".join(f"{item:<{col_width}}" for item in row_items)
            lines.append(line.rstrip())
            
        return lines

    def _generate_global_hotspot_summary(self, add_line, use_color: bool):
        """Generate a summary of global variable hotspots across the entire codebase."""
        add_line("GLOBAL HOTSPOT SUMMARY", color=Color.HEADER)
        add_line("-" * 50, color=Color.HEADER)

        global_writes_count = defaultdict(int)
        proc_global_write_counts = defaultdict(int)
        global_var_names_lower = {name.lower() for name in self.global_variables.keys()}

        # Aggregate writes from all procedures in all analyzed files
        for filename in self.procedure_assignments:
            for proc_name, assignments in self.procedure_assignments[filename].items():
                modified_globals_in_proc = {assign.variable.lower() for assign in assignments if assign.variable.lower() in global_var_names_lower}
                if modified_globals_in_proc:
                    proc_full_name = f"{proc_name} (in {Path(filename).name})"
                    proc_global_write_counts[proc_full_name] += len(modified_globals_in_proc)
                    for var_name in modified_globals_in_proc:
                        global_writes_count[var_name] += 1
        
        sorted_global_writes = sorted(global_writes_count.items(), key=lambda item: item[1], reverse=True)
        sorted_hotspots = sorted(proc_global_write_counts.items(), key=lambda item: item[1], reverse=True)

        if sorted_hotspots:
            add_line("Top Global Variable Modifying Procedures (Codebase-wide):", 1, color=Color.FAIL)
            for proc_name, count in sorted_hotspots[:10]:
                add_line(f"{proc_name:<40}: Modifies {count} unique global variables", 2)
            add_line()

        if sorted_global_writes:
            add_line("Most Frequently Modified Global Variables (Codebase-wide):", 1, color=Color.FAIL)
            for var_name, count in sorted_global_writes[:10]:
                add_line(f"{var_name:<40}: Modified by {count} procedure(s)", 2)
            add_line()
        
        if not sorted_hotspots and not sorted_global_writes:
            add_line("No global variable modifications found.", 1)
            add_line()

    def _format_variable_line(self, var_info: VariableInfo, scope_or_usage: str, use_color: bool) -> str:
        """Formats a single line for a variable declaration with consistent padding."""
        line_prefix = f"Line {var_info.line_num:4d}:"
        name_str = f"{var_info.name:<20}"
        type_str = f"{var_info.type:<15}"
        
        # Combine dimensions and attributes
        attributes = []
        if var_info.dimensions:
            attributes.append(var_info.dimensions)
        if var_info.is_parameter:
            attributes.append("[PARAM]")
        if var_info.is_allocatable:
            attributes.append("[ALLOCATABLE]")
        
        attr_str = ' '.join(attributes)

        if use_color:
            scope_color = Color.ENDC
            if "MODIFIED" in scope_or_usage:
                scope_color = Color.FAIL
            elif "USED" in scope_or_usage:
                scope_color = Color.GREEN
            elif "UNUSED" in scope_or_usage:
                scope_color = Color.WARNING
            elif "GLOBAL" in scope_or_usage:
                scope_color = Color.BLUE
            
            name_str = f"{Color.BOLD}{name_str}{Color.ENDC}"
            type_str = f"{Color.CYAN}{type_str}{Color.ENDC}"
            scope_str = f"[{scope_color}{scope_or_usage}{Color.ENDC}]"
        else:
            scope_str = f"[{scope_or_usage}]"
        
        return f"{line_prefix} {name_str} {type_str} {attr_str:<25} {scope_str}"

    def _generate_file_summary(self, filename: str, add_line, use_color: bool):
        """Generate a high-level summary for a single file."""
        procedures = self.file_procedures.get(filename, [])
        if not procedures:
            return

        global_reads_count = defaultdict(int)
        global_writes_count = defaultdict(int)
        proc_global_write_counts = []
        global_var_names_lower = {name.lower() for name in self.global_variables.keys()}

        for proc in procedures:
            # Aggregate global reads
            proc_reads = self.procedure_reads.get(filename, {}).get(proc.name, [])
            for var_name, _, _ in proc_reads:
                if var_name.lower() in global_var_names_lower:
                    global_reads_count[var_name.lower()] += 1

            # Aggregate global writes and count per-procedure modifications
            proc_assignments = self.procedure_assignments.get(filename, {}).get(proc.name, [])
            modified_globals_in_proc = {assign.variable.lower() for assign in proc_assignments if assign.variable.lower() in global_var_names_lower}
            
            if modified_globals_in_proc:
                proc_global_write_counts.append((proc.name, len(modified_globals_in_proc)))
                for var_name in modified_globals_in_proc:
                    global_writes_count[var_name] += 1

        sorted_global_reads = sorted(global_reads_count.items(), key=lambda item: item[1], reverse=True)
        sorted_global_writes = sorted(global_writes_count.items(), key=lambda item: item[1], reverse=True)
        sorted_hotspots = sorted(proc_global_write_counts, key=lambda item: item[1], reverse=True)

        add_line("High-Level Summary", 1, color=Color.BLUE)
        add_line("------------------", 1, color=Color.BLUE)
        add_line(f"Total Procedures: {len(procedures)}", 2)
        add_line()

        if sorted_hotspots:
            add_line("Top Global Variable Modifiers (Hotspots):", 2, color=Color.WARNING)
            for proc_name, count in sorted_hotspots[:5]:
                add_line(f"{proc_name:<30}: Modifies {count} global variables", 3)
            add_line()

        if sorted_global_writes:
            add_line("Most Frequently Modified Global Variables:", 2, color=Color.FAIL)
            for var_name, count in sorted_global_writes[:5]:
                add_line(f"{var_name:<30}: Modified in {count} procedure(s)", 3)
            add_line()

        if sorted_global_reads:
            add_line("Most Frequently Read Global Variables:", 2, color=Color.WARNING)
            for var_name, count in sorted_global_reads[:5]:
                add_line(f"{var_name:<30}: Read {count} time(s)", 3)
            add_line()
        
        add_line("-" * 80)
        add_line()

    def _generate_global_summary(self, add_line, show_all_globals: bool, use_color: bool):
        """Generate global variables summary"""
        add_line("GLOBAL VARIABLES SUMMARY", color=Color.HEADER)
        add_line("-" * 50, color=Color.HEADER)
        add_line(f"Total modules found: {len(self.modules)}")
        add_line(f"Total global variables: {len(self.global_variables)}")

        if show_all_globals:
            add_line("Showing ALL global variables (--show-all-globals specified)", color=Color.CYAN)
        else:
            add_line("Showing first 10 variables per module (use --show-all-globals for complete list)", color=Color.CYAN)
        add_line()

        for module_name, variables in self.modules.items():
            location = self.module_locations.get(module_name, 'unknown file')
            add_line(f"Module: {module_name} (in {location}) ({len(variables)} variables)", color=Color.BLUE)

            # Show all variables if requested, otherwise limit to 10
            display_vars = variables if show_all_globals else variables[:10]

            # Count usage statistics for this module
            used_count = 0
            modified_count = 0

            for var_name in display_vars:
                if var_name in self.global_variables:
                    var_info = self.global_variables[var_name]
                    usage_status = self._check_global_variable_usage(var_name)
                    line_str = self._format_variable_line(var_info, usage_status, use_color)
                    add_line(line_str, 1)

                    if "USED" in usage_status:
                        used_count += 1
                    if "MODIFIED" in usage_status:
                        modified_count += 1

            if not show_all_globals and len(variables) > 10:
                add_line(f"  ... and {len(variables) - 10} more variables (use --show-all-globals to see all)", 1, color=Color.CYAN)

            # Add module usage summary
            usage_pct = (used_count / len(display_vars)) * 100 if display_vars else 0
            add_line(f"  Module Usage: {used_count}/{len(display_vars)} variables used ({usage_pct:.1f}%), {modified_count} modified", 1)
            add_line()

    def _generate_cross_reference_analysis(self, add_line, use_color: bool):
        """Generate cross-reference analysis"""
        add_line("CROSS-REFERENCE ANALYSIS", color=Color.HEADER)
        add_line("-" * 30, color=Color.HEADER)

        # Collect all assignments and reads from both file and procedure level
        all_assigned_vars = set()
        all_read_vars = set()

        # File-level
        for assignments in self.file_assignments.values():
            for assignment in assignments:
                all_assigned_vars.add(assignment.variable.lower())

        for reads in self.file_reads.values():
            for var_name, _, _ in reads:
                all_read_vars.add(var_name.lower())

        # Procedure-level
        for filename_procedures in self.procedure_assignments.values():
            for procedure_assignments in filename_procedures.values():
                for assignment in procedure_assignments:
                    all_assigned_vars.add(assignment.variable.lower())

        for filename_procedures in self.procedure_reads.values():
            for procedure_reads in filename_procedures.values():
                for var_name, _, _ in procedure_reads:
                    all_read_vars.add(var_name.lower())

        global_var_names = {name.lower() for name in self.global_variables.keys()}

        add_line("Global variables being modified:", 1, color=Color.FAIL)
        modified_globals = sorted(list(all_assigned_vars.intersection(global_var_names)))
        for line in self._format_list_in_columns(modified_globals):
            add_line(line, 2)

        add_line()
        add_line("Global variables being read:", 1, color=Color.WARNING)
        read_globals = sorted(list(all_read_vars.intersection(global_var_names)))
        for line in self._format_list_in_columns(read_globals):
            add_line(line, 2)

        add_line()
        add_line("Variables assigned but not in global scope:", 1, color=Color.CYAN)
        local_assigned = sorted(list(all_assigned_vars - global_var_names))
        for line in self._format_list_in_columns(local_assigned):
            add_line(line, 2)

    def _generate_call_graph_analysis(self, add_line, use_color: bool):
        """Generate a reverse call graph (who calls whom)."""
        add_line()
        add_line("CALL GRAPH ANALYSIS (Who calls whom)", color=Color.HEADER)
        add_line("-" * 40, color=Color.HEADER)

        # Build a reverse map: called_proc -> [calling_proc1, calling_proc2, ...]
        reverse_call_map = defaultdict(list)
        all_known_procs = set()

        for filename, procs in self.file_procedures.items():
            for proc in procs:
                all_known_procs.add(proc.name.lower())
                if filename in self.procedure_calls and proc.name in self.procedure_calls[filename]:
                    calls = self.procedure_calls[filename][proc.name]
                    for call in calls:
                        reverse_call_map[call.called_name.lower()].append(f"{proc.name} (in {Path(filename).name})")

        # Sort by procedure name
        sorted_called_procs = sorted(reverse_call_map.keys())

        if not sorted_called_procs:
            add_line("No procedure calls were found.", 1)
            return

        for called_proc in sorted_called_procs:
            callers = sorted(list(set(reverse_call_map[called_proc])))
            
            # Check if the called procedure is defined in the analyzed files
            is_defined = called_proc in all_known_procs
            status = "[DEFINED]" if is_defined else "[EXTERNAL/UNDEFINED]"
            if use_color:
                color = Color.GREEN if is_defined else Color.WARNING
                status = f"{color}{status}{Color.ENDC}"

            add_line(f"Procedure: {called_proc} {status}", 1)
            add_line(f"  Called by ({len(callers)}):", 1)
            for caller in callers:
                add_line(caller, 2)
            add_line()

    def classify_variable_scope(self, var_name: str, filename: str, proc_name: Optional[str] = None) -> str:
        """Enhanced scope classification"""
        var_lower = var_name.lower()

        # Check if it's a global variable
        if var_lower in [v.lower() for v in self.global_variables.keys()]:
            return "GLOBAL"

        # Check procedure-level variables first if proc_name is given
        if proc_name and filename in self.procedure_variables:
            if proc_name in self.procedure_variables[filename]:
                proc_var_names = [v.name.lower() for v in self.procedure_variables[filename][proc_name]]
                if var_lower in proc_var_names:
                    return f"LOCAL"

        # Check if it's a local variable in current file
        if filename in self.file_variables:
            local_vars = [v.name.lower() for v in self.file_variables[filename]]
            if var_lower in local_vars:
                return "FILE_LOCAL"

        # Check if it's from a USE statement (procedure scope)
        if proc_name and filename in self.procedure_use_statements:
            if proc_name in self.procedure_use_statements[filename]:
                for module, only_list, _ in self.procedure_use_statements[filename][proc_name]:
                    if only_list == 'all' and module in self.modules:
                        if var_lower in [v.lower() for v in self.modules[module]]:
                             return f"IMPORTED"
                    elif only_list != 'all':
                        only_vars = [v.strip().lower() for v in only_list.split(',')]
                        if var_lower in only_vars:
                            return f"IMPORTED"

        # Check if it's from a USE statement (file scope)
        if filename in self.use_statements:
            for module, only_list, _ in self.use_statements[filename]:
                if only_list == 'all' and module in self.modules:
                    if var_lower in [v.lower() for v in self.modules[module]]:
                        return f"IMPORTED"
                elif only_list != 'all':
                    only_vars = [v.strip().lower() for v in only_list.split(',')]
                    if var_lower in only_vars:
                        return f"IMPORTED"

        return "UNKNOWN"

    def _check_global_variable_usage(self, var_name: str) -> str:
        """Check if a global variable is used in any analyzed file"""
        var_lower = var_name.lower()

        # Check if variable is read (file-level and procedure-level)
        is_read = False
        for reads in self.file_reads.values():
            for read_var, _, _ in reads:
                if read_var.lower() == var_lower:
                    is_read = True
                    break
            if is_read:
                break

        if not is_read:
            for filename_procedures in self.procedure_reads.values():
                for procedure_reads in filename_procedures.values():
                    for read_var, _, _ in procedure_reads:
                        if read_var.lower() == var_lower:
                            is_read = True
                            break
                    if is_read:
                        break
                if is_read:
                    break

        # Check if variable is assigned (file-level and procedure-level)
        is_assigned = False
        for assignments in self.file_assignments.values():
            for assignment in assignments:
                if assignment.variable.lower() == var_lower:
                    is_assigned = True
                    break
            if is_assigned:
                break

        if not is_assigned:
            for filename_procedures in self.procedure_assignments.values():
                for procedure_assignments in filename_procedures.values():
                    for assignment in procedure_assignments:
                        if assignment.variable.lower() == var_lower:
                            is_assigned = True
                            break
                    if is_assigned:
                        break
                if is_assigned:
                    break

        if is_read and is_assigned:
            return "USED+MODIFIED"
        elif is_read:
            return "USED"
        elif is_assigned:
            return "MODIFIED"
        else:
            return "UNUSED"

def main():
    """Main entry point for the command line interface"""
    parser = argparse.ArgumentParser(
        description="Analyze Fortran source files for variable sources and dependencies. "
                    "Scans a codebase for globals then analyzes specific files.",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument('codebase_dir', help='Directory containing the full Fortran codebase to scan for globals.')
    parser.add_argument('files_to_analyze', nargs='*', 
                       help='(Optional) Specific Fortran files to analyze in detail. If omitted, all files in codebase_dir are analyzed.')
    parser.add_argument('--output', '-o',
                       help='Output file for report (default: print to stdout)')
    parser.add_argument('--verbose', '-v', action='store_true',
                       help='Verbose output')
    parser.add_argument('--debug', action='store_true',
                       help='Debug mode - show parsing warnings')
    parser.add_argument('--show-all-globals', action='store_true',
                       help='Show all global variables instead of limiting to 10')
    parser.add_argument('--enhanced', action='store_true',
                       help='Generate enhanced report with statistics and recommendations')
    parser.add_argument('--truncate', action='store_true',
                       help='Truncate long lists (default: show full lists)')
    parser.add_argument('--color', choices=['auto', 'always', 'never'], default='auto',
                       help='Control colorized output. `always` is useful for piping to `less -R`.')
    
    # New filtering options
    parser.add_argument('--show-only-file', nargs='+', metavar='FILENAME',
                       help='Only show analysis for specific file(s) by name.')
    parser.add_argument('--show-only-proc', nargs='+', metavar='PROCNAME',
                       help='Only show analysis for specific procedure(s) by name.')
    parser.add_argument('--hide-locals', action='store_true',
                       help='Hide local variable declaration lists in procedures.')
    parser.add_argument('--hide-ok', action='store_true',
                       help='Hide procedures that do not read or modify any global variables.')

    args = parser.parse_args()
    
    codebase_path = Path(args.codebase_dir)
    if not codebase_path.is_dir():
        cprint(f"Error: Codebase directory not found: {args.codebase_dir}", Color.FAIL, force_color=args.color=='always')
        return 1

    # Determine which files to run the detailed analysis on
    if args.files_to_analyze:
        analysis_targets = [Path(f) for f in args.files_to_analyze]
    else:
        # If no specific files are given, analyze all Fortran files in the codebase dir
        cprint("No specific files provided; analyzing all Fortran files in the codebase directory.", Color.WARNING, force_color=args.color=='always')
        analysis_targets = []
        extensions = ['*.f90', '*.f', '*.F90', '*.F']
        for ext in extensions:
            analysis_targets.extend(codebase_path.rglob(ext))

    if not analysis_targets:
        cprint(f"Error: No Fortran files found to analyze.", Color.FAIL, force_color=args.color=='always')
        return 1
    
    # Initialize analyzer
    try:
        analyzer = FortranVariableAnalyzer()
    except Exception as e:
        cprint(f"Error initializing analyzer: {e}", Color.FAIL, force_color=args.color=='always')
        if args.debug:
            import traceback
            traceback.print_exc()
        return 1

    # Pass 1: Scan the entire codebase for global variables
    analyzer.scan_directory_for_globals(codebase_path)
    
    # Pass 2: Analyze each target input file in detail
    analyzed_count = 0
    cprint("\nStarting detailed analysis...", Color.BOLD, force_color=args.color=='always')
    for file_path in analysis_targets:
        try:
            analyzer.analyze_file(str(file_path))
            analyzed_count += 1
        except Exception as e:
            cprint(f"Error analyzing {file_path}: {e}", Color.FAIL, force_color=args.color=='always')
            if args.debug:
                import traceback
                traceback.print_exc()
    
    if analyzed_count == 0:
        cprint("Error: No files were successfully analyzed", Color.FAIL, force_color=args.color=='always')
        return 1
    
    cprint(f"\nSuccessfully analyzed {analyzed_count}/{len(analysis_targets)} files", Color.GREEN, force_color=args.color=='always')
    
    # Generate report
    try:
        analyzer.generate_report(
            args.output, 
            args.show_all_globals, 
            args.truncate, 
            args.enhanced,
            show_only_file=args.show_only_file,
            show_only_proc=args.show_only_proc,
            hide_locals=args.hide_locals,
            hide_ok=args.hide_ok,
            color_mode=args.color
        )
        return 0
        
    except Exception as e:
        cprint(f"Error generating report: {e}", Color.FAIL, force_color=args.color=='always')
        if args.debug:
            import traceback
            traceback.print_exc()
        return 1

if __name__ == "__main__":
    sys.exit(main())