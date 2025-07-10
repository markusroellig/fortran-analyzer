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

ArgumentInfo = namedtuple('ArgumentInfo', [
    'name', 'intent', 'type_info'
])

CallInfo = namedtuple('CallInfo', [
    'called_name', 'line_num', 'call_type', 'arguments'
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
        in_interface_block = False
        interface_depth = 0
        in_procedure_def = False
        proc_buffer = ""
        proc_start_line = 0

        # This regex handles multi-line arguments
        proc_regex = re.compile(r'^\s*(subroutine|function)\s+([a-zA-Z_]\w*)\s*(?:\((.*?)\))?', re.IGNORECASE | re.DOTALL)

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

            # Track interface blocks to skip them
            if re.match(r'^\s*interface\b', line_lower):
                in_interface_block = True
                interface_depth += 1
                continue
            
            if re.match(r'^\s*end\s+interface\b', line_lower):
                interface_depth -= 1
                if interface_depth == 0:
                    in_interface_block = False
                continue

            # Skip procedures inside interface blocks
            if in_interface_block:
                continue

            # Handle procedure definitions with continuations
            if not in_procedure_def and re.match(r'^\s*(subroutine|function)\s+([a-zA-Z_]\w*)', line_clean, re.IGNORECASE):
                in_procedure_def = True
                proc_start_line = i
                proc_buffer = line_clean.rstrip('&').rstrip()
            elif in_procedure_def and line_clean.strip():
                # Append continued lines
                proc_buffer += ' ' + line_clean.lstrip('&').strip()
            
            # Check if procedure definition is complete (no continuation or found closing paren)
            if in_procedure_def and (not line_clean.endswith('&') or ')' in line_clean):
                in_procedure_def = False
                proc_match = proc_regex.match(proc_buffer)
                
                if proc_match:
                    proc_type = proc_match.group(1).lower()
                    proc_name = proc_match.group(2).lower()
                    arg_string = proc_match.group(3)

                    # Parse arguments more carefully
                    arguments = []
                    if arg_string:
                        # Clean up argument string
                        arg_string_clean = arg_string.replace('&', ' ').replace('\n', ' ').strip()
                        # Split by comma, respecting parentheses
                        arg_parts = self.smart_split(arg_string_clean, ',')
                        
                        for arg_part in arg_parts:
                            arg_part = arg_part.strip()
                            if not arg_part:
                                continue
                            
                            # Extract argument name (might have dimensions)
                            arg_name_match = re.match(r'^(\w+)', arg_part)
                            if arg_name_match:
                                arg_name = arg_name_match.group(1).lower()
                                # Skip if it's a keyword that might appear in complex declarations
                                if arg_name not in ['real', 'integer', 'character', 'logical', 'type', 'class', 'double', 'precision']:
                                    arguments.append(ArgumentInfo(
                                        name=arg_name,
                                        intent='UNKNOWN',
                                        type_info=''
                                    ))

                    # Find the end of this procedure
                    end_line = self.find_procedure_end(lines, proc_start_line, proc_type, proc_name)

                    procedures.append(ProcedureInfo(
                        name=proc_name,
                        type=proc_type,
                        start_line=proc_start_line,
                        end_line=end_line,
                        module=current_module,
                        arguments=arguments,
                        calls=[]
                    ))
                proc_buffer = ""  # Reset buffer

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
                end_name = end_match.group(3) if end_match.group(3) else ''

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
        in_interface_block = False
        interface_depth = 0
        
        line_buffer = ""
        line_idx = 0
        start_line_num = 1
        
        while line_idx < len(lines):
            original_line = lines[line_idx]
            
            # Track where multi-line statements start
            if not line_buffer:
                start_line_num = line_idx + 1

            # Handle line continuations
            clean_original = self.clean_line(original_line)
            if clean_original.endswith('&'):
                line_buffer += clean_original[:-1].rstrip() + " "
                line_idx += 1
                continue
            else:
                line_buffer += clean_original

            line_clean = line_buffer.strip()
            line_buffer = ""  # Reset buffer
            line_lower = line_clean.lower()
            line_idx += 1

            # Skip empty lines
            if not line_clean:
                continue

            # Track module context
            module_match = re.match(r'^\s*module\s+(\w+)', line_lower)
            if module_match and not line_lower.startswith('end module'):
                current_module = module_match.group(1)
                continue

            if re.match(r'^\s*end\s+module', line_lower):
                current_module = None
                current_procedure = None
                continue

            # Track interface blocks
            if re.match(r'^\s*interface\b', line_lower) and not re.match(r'^\s*end\s+interface', line_lower):
                in_interface_block = True
                interface_depth += 1
                continue
            
            if re.match(r'^\s*end\s+interface\b', line_lower):
                interface_depth -= 1
                if interface_depth == 0:
                    in_interface_block = False
                continue

            # Skip parsing inside interface blocks
            if in_interface_block:
                continue

            # Find current procedure based on line number
            current_procedure = None
            for proc in procedures:
                if proc.start_line <= start_line_num <= proc.end_line:
                    current_procedure = proc
                    break

            # Skip the procedure definition line itself
            if current_procedure and start_line_num == current_procedure.start_line:
                continue

            # Parse USE statements
            use_match = re.match(r'^\s*use\s+(\w+)(?:\s*,\s*only\s*:\s*(.+))?', line_lower, re.DOTALL)
            if use_match:
                module_name = use_match.group(1)
                only_list = use_match.group(2) if use_match.group(2) else 'all'
                if only_list != 'all':
                    # Clean up the only list
                    only_list = ' '.join(only_list.split())

                if current_procedure:
                    self.procedure_use_statements[filename][current_procedure.name].append(
                        (module_name, only_list, start_line_num))
                else:
                    self.use_statements[filename].append((module_name, only_list, start_line_num))
                continue

            # Parse INTENT statements for procedure arguments
            if current_procedure:
                # Look for intent declarations (both inline and separate)
                intent_pattern = r'intent\s*\(\s*(in|out|inout)\s*\)'
                intent_matches = re.finditer(intent_pattern, line_lower)
                
                for intent_match in intent_matches:
                    intent = intent_match.group(1).upper()
                    
                    # Find variables with this intent on this line
                    # Look for :: followed by variable list
                    var_list_match = re.search(r'::\s*(.+)', line_clean[intent_match.end():])
                    if not var_list_match:
                        # Try looking before the intent
                        var_list_match = re.search(r'^\s*([^:]+?)\s*,\s*intent', line_lower)
                    
                    if var_list_match:
                        var_string = var_list_match.group(1)
                        var_names = []
                        
                        # Parse variable names from the list
                        var_parts = self.smart_split(var_string, ',')
                        for var_part in var_parts:
                            var_part = var_part.strip()
                            # Extract just the variable name
                            var_name_match = re.match(r'^(\w+)', var_part)
                            if var_name_match:
                                var_names.append(var_name_match.group(1).lower())
                        
                        # Update the arguments in the procedure info
                        updated_args = []
                        for arg in current_procedure.arguments:
                            if arg.name in var_names:
                                updated_args.append(arg._replace(intent=intent))
                            else:
                                updated_args.append(arg)
                        
                        # Update the procedure info
                        proc_list = self.file_procedures[filename]
                        for idx, proc in enumerate(proc_list):
                            if proc.name == current_procedure.name and proc.start_line == current_procedure.start_line:
                                new_proc_info = proc._replace(arguments=updated_args)
                                proc_list[idx] = new_proc_info
                                current_procedure = new_proc_info
                                break

            # Parse variable declarations
            var_infos = self.parse_variable_declaration(line_clean, start_line_num, current_module or 'local')
            if current_procedure:
                # Filter out variables that are procedure arguments
                arg_names = {arg.name for arg in current_procedure.arguments}
                var_infos = [v for v in var_infos if v.name not in arg_names]
                if var_infos:
                    self.procedure_variables[filename][current_procedure.name].extend(var_infos)
            else:
                self.file_variables[filename].extend(var_infos)
                # If we are in a module but not a procedure, these are global variables
                if current_module and not in_interface_block:
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
                calls = self.parse_procedure_calls(line_clean, start_line_num, filename, current_procedure.name)
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

        # Skip certain lines that are definitely not declarations
        if (line_lower.startswith('implicit') or line_lower.startswith('save') or
            line_lower.startswith('contains') or line_lower.startswith('use') or
            line_lower.startswith('private') or line_lower.startswith('public') or
            line_lower.startswith('procedure') or line_lower.startswith('class')):
            return result
        
        # In parse_variable_declaration, add a check to skip interface parameters:
        if re.match(r'^\s*interface\b', line_lower):
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
            # F77-style declaration: type variable_list
            else:
                f77_type_pattern = r'^\s*(integer|real|double\s+precision|complex|logical|character(?:\s*\*[\s\d\w\(\)]+)?)\s+(.*)'
                match = re.match(f77_type_pattern, line_clean, re.IGNORECASE)
                if match:
                    type_part = match.group(1)
                    var_part = match.group(2)

                    # Avoid misinterpreting a function definition as a variable declaration
                    if re.match(r'^\s*function\b', var_part, re.IGNORECASE):
                        return result

                    type_part_lower = type_part.lower()
                    kind = ''
                    var_type = ''

                    if type_part_lower.startswith('character'):
                        var_type = 'character'
                        kind_match = re.search(r'(\*.*)', type_part, re.IGNORECASE)
                        if kind_match:
                            kind = kind_match.group(1).strip()
                    elif type_part_lower.startswith('double'):
                        var_type = 'double_precision'
                    else:
                        type_match = re.match(r'(\w+)', type_part_lower)
                        var_type = type_match.group(1)
                        kind_match = re.search(r'\*(\s*\d+)', type_part_lower)
                        if kind_match:
                            kind = kind_match.group(1).strip()
                    
                    variables = self.parse_variable_list(var_part)
                    for var_name, dimensions, initial_value in variables:
                        if var_name:
                            result.append(VariableInfo(
                                name=var_name, type=var_type, kind=kind,
                                dimensions=dimensions, is_parameter=False,
                                is_allocatable=False, initial_value=initial_value,
                                module=module, line_num=line_num
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
        """Parse variable reads (usage) from a line with better procedure name handling."""
        reads = []
        line_clean = line.strip()
        
        # Skip declarations, interface blocks, etc.
        if not line_clean or re.match(r'^\s*(interface|end\s+interface|contains|implicit|subroutine|function|module|program)', line_clean.lower()):
            return reads

        # For CALL statements, don't count the procedure name as a variable read
        if re.search(r'\bcall\s+\w+', line_clean.lower()):
            # Extract just the part inside the parentheses
            match = re.search(r'\bcall\s+\w+\s*\((.*)\)', line_clean.lower())
            search_text = match.group(1) if match else ""
        # For assignment lines, only look at RHS
        elif '=' in line_clean and not line_clean.lower().strip().startswith('if'):
            # Split on first = to get RHS
            search_text = line_clean.split('=', 1)[1]
        else:
            search_text = line_clean

        # Find potential procedure names (anything followed by parentheses)
        proc_pattern = r'\b([a-zA-Z_]\w*)\s*\('
        proc_names = {match.group(1).lower() for match in re.finditer(proc_pattern, search_text.lower())}
        
        # Find variable names
        var_pattern = r'\b([a-zA-Z_]\w*)\b'
        fortran_keywords = {
            'if', 'then', 'else', 'endif', 'end', 'do', 'while', 'call', 'exit', 'cycle',
            'function', 'subroutine', 'return', 'stop', 'print', 'write', 'save',
            'read', 'open', 'close', 'sqrt', 'exp', 'log', 'sin', 'cos', 'tan',
            'abs', 'max', 'min', 'real', 'integer', 'logical', 'character', 'dp',
            'in', 'out', 'inout', 'intent', 'parameter', 'allocatable', 'dimension',
            'true', 'false', 'and', 'or', 'not', 'eq', 'ne', 'gt', 'lt', 'ge', 'le'
        }

        for match in re.finditer(var_pattern, search_text):
            var_name = match.group(1).lower()

            # Skip keywords, procedure names, and obvious non-variables
            if (var_name not in fortran_keywords and 
                var_name not in proc_names and
                not var_name.isdigit()):
                reads.append((var_name, line_num, 'VARIABLE_READ'))

        return reads

    def parse_procedure_calls(self, line: str, line_num: int, filename: str, proc_name: str) -> List[CallInfo]:
        """Parse procedure calls from a line with improved declaration filtering."""
        calls = []
        line_clean = line.strip()
        line_lower = line_clean.lower()

        # Skip empty lines and comments
        if not line_clean or line_clean.startswith('!'):
            return calls

        # Comprehensive detection of declaration lines to skip
        declaration_patterns = [
            r'^\s*(integer|real|double\s+precision|complex|logical|character|type)\s*[\(\::]',
            r'^\s*\w+\s*,\s*(intent|dimension|parameter|allocatable|pointer|target|save)',
            r'^\s*(intent|dimension|parameter|allocatable|pointer|target|save)\s*[\(\::]',
            r'^\s*interface\b',
            r'^\s*end\s+interface\b',
            r'^\s*contains\b',
            r'^\s*implicit\s+none\b',
            r'^\s*use\s+\w+',
            r'^\s*module\s+\w+',
            r'^\s*program\s+\w+',
            r'^\s*subroutine\s+\w+',
            r'^\s*function\s+\w+',
            r'^\s*end\s+(subroutine|function|program|module)',
            r'^\s*\w+\s*::\s*',  # Any variable with :: declaration syntax
        ]
        
        # Skip lines that match declaration patterns
        for pattern in declaration_patterns:
            if re.match(pattern, line_lower):
                return calls

        # Skip lines that are clearly control structures or other non-call statements
        control_patterns = [
            r'^\s*(if|then|else|elseif|endif|select|case|end\s+select)',
            r'^\s*(do|enddo|end\s+do|while|forall|where)',
            r'^\s*(stop|return|cycle|exit)',
            r'^\s*(write|print|read|open|close)\s*[\(\*]',
            r'^\s*\w+\s*=',  # Assignment statements
        ]
        
        for pattern in control_patterns:
            if re.match(pattern, line_lower):
                return calls

        # Find CALL statements (subroutine calls)
        call_matches = re.finditer(r'\bcall\s+([a-zA-Z_]\w*)\s*(?:\((.*?)\))?', line_lower, re.DOTALL)
        for call_match in call_matches:
            called_name = call_match.group(1)
            arg_string = call_match.group(2) if call_match.group(2) else ""
            
            arguments = self.smart_split(arg_string, ',') if arg_string else []
            
            calls.append(CallInfo(
                called_name=called_name,
                line_num=line_num,
                call_type='subroutine',
                arguments=[arg.strip() for arg in arguments if arg.strip()]
            ))

        # For function calls, be much more restrictive
        # Only look for function calls in assignment RHS or other expression contexts
        search_text = line_clean
        
        # If it's an assignment, only look at the RHS
        if '=' in line_clean and not line_lower.strip().startswith('if'):
            parts = line_clean.split('=', 1)
            if len(parts) == 2:
                search_text = parts[1]
        
        # More restrictive function call pattern that avoids type declarations
        # This pattern looks for function-like calls but excludes common declaration contexts
        func_pattern = r'(?<![a-zA-Z_])(assert_eq|cumsum|iminlocdp|arth|maxval|minval|max|min|abs|size|shape|allocated|present)\s*\('
        
        # Find intrinsic/library function calls that we know are functions
        known_functions = {
            'assert_eq', 'cumsum', 'iminlocdp', 'arth', 'maxval', 'minval', 
            'max', 'min', 'abs', 'size', 'shape', 'allocated', 'present',
            'sqrt', 'exp', 'log', 'sin', 'cos', 'tan', 'real', 'int', 'nint'
        }
        
        # Look for known function patterns
        for func_match in re.finditer(func_pattern, search_text, re.IGNORECASE):
            func_name = func_match.group(1).lower()
            if func_name in known_functions:
                # Find the arguments for this function call
                start_pos = func_match.end() - 1  # Position of opening parenthesis
                arg_string = self.extract_parenthesized_content(search_text, start_pos)
                if arg_string is not None:
                    arguments = self.smart_split(arg_string, ',') if arg_string else []
                    calls.append(CallInfo(
                        called_name=func_name,
                        line_num=line_num,
                        call_type='function',
                        arguments=[arg.strip() for arg in arguments if arg.strip()]
                    ))

        return calls

    def extract_parenthesized_content(self, text: str, start_pos: int) -> Optional[str]:
        """Extract content between balanced parentheses starting at start_pos."""
        if start_pos >= len(text) or text[start_pos] != '(':
            return None
        
        paren_count = 0
        content_start = start_pos + 1
        
        for i in range(start_pos, len(text)):
            if text[i] == '(':
                paren_count += 1
            elif text[i] == ')':
                paren_count -= 1
                if paren_count == 0:
                    return text[content_start:i]
        
        return None  # Unbalanced parentheses
    
    def correlate_call_arguments(self, call_info: CallInfo, called_proc_info: ProcedureInfo) -> List[Tuple[str, str, str]]:
        """
        Correlate actual arguments in a call with dummy arguments in procedure definition.
        Returns: [(actual_arg, dummy_arg, intent), ...]
        """
        correlations = []
        
        try:
            # Match actual arguments with dummy arguments by position
            for i, actual_arg in enumerate(call_info.arguments):
                if i < len(called_proc_info.arguments):
                    dummy_arg = called_proc_info.arguments[i]
                    
                    # Extract variable name from actual argument (remove array indices, etc.)
                    actual_var = re.match(r'^(\w+)', actual_arg.strip())
                    if actual_var and hasattr(dummy_arg, 'name') and hasattr(dummy_arg, 'intent'):
                        correlations.append((
                            actual_var.group(1).lower(),
                            dummy_arg.name,
                            dummy_arg.intent
                        ))
        except Exception as e:
            print(f"Debug: Error in correlate_call_arguments: {e}")
            return []
        
        return correlations
    
    def find_procedure_definition(self, proc_name: str) -> Optional[ProcedureInfo]:
        """Find the definition of a procedure by name across all analyzed files."""
        proc_name_lower = proc_name.lower()
        
        for filename, procedures in self.file_procedures.items():
            for proc in procedures:
                if proc.name == proc_name_lower:
                    return proc
        
        return None
    
    def analyze_indirect_modifications(self, filename: str, proc_name: str):
        """Detect global variables potentially modified through procedure calls."""
        indirect_mods = []
        
        try:
            if filename not in self.procedure_calls or proc_name not in self.procedure_calls[filename]:
                return indirect_mods
            
            calls = self.procedure_calls[filename][proc_name]
            
            for call in calls:
                try:
                    # Find the definition of the called procedure
                    called_proc = self.find_procedure_definition(call.called_name)
                    
                    if called_proc:
                        # Correlate arguments
                        correlations = self.correlate_call_arguments(call, called_proc)
                        
                        for correlation in correlations:
                            try:
                                # Ensure we have a 3-tuple
                                if len(correlation) == 3:
                                    actual_arg, dummy_arg, intent = correlation
                                    
                                    # Check if actual argument is a global variable
                                    if actual_arg in self.global_variables:
                                        # Check if it can be modified (OUT or INOUT)
                                        if intent in ['OUT', 'INOUT']:
                                            indirect_mods.append({
                                                'variable': actual_arg,
                                                'call_line': call.line_num,
                                                'called_proc': call.called_name,
                                                'dummy_arg': dummy_arg,
                                                'intent': intent
                                            })
                            except Exception as e:
                                print(f"Debug: Error processing correlation {correlation}: {e}")
                                continue
                except Exception as e:
                    print(f"Debug: Error processing call {call.called_name}: {e}")
                    continue
                    
        except Exception as e:
            print(f"Debug: Error in analyze_indirect_modifications: {e}")
            return []
        
        return indirect_mods

    def _infer_argument_intents(self):
        """
        Iterate through all procedures and infer the intent of arguments
        if they are not explicitly defined.
        """
        for filename, procedures in self.file_procedures.items():
            updated_procedures = []
            for proc in procedures:
                updated_args = []
                # Only infer for arguments with unknown intent
                if not any(arg.intent != 'UNKNOWN' for arg in proc.arguments):
                    proc_assignments = self.procedure_assignments.get(filename, {}).get(proc.name, [])
                    proc_reads = self.procedure_reads.get(filename, {}).get(proc.name, [])
                    
                    assigned_vars = {a.variable.lower() for a in proc_assignments}
                    read_vars = {r[0].lower() for r in proc_reads}

                    for arg in proc.arguments:
                        arg_lower = arg.name.lower()
                        is_written = arg_lower in assigned_vars
                        is_read = arg_lower in read_vars
                        
                        inferred_intent = arg.intent
                        if inferred_intent == 'UNKNOWN':
                            if is_written and is_read:
                                inferred_intent = 'INOUT'
                            elif is_written:
                                inferred_intent = 'OUT'
                            elif is_read:
                                inferred_intent = 'IN'
                        
                        updated_args.append(arg._replace(intent=inferred_intent))
                else:
                    # If any argument has an explicit intent, trust the existing parsing
                    updated_args = proc.arguments

                updated_procedures.append(proc._replace(arguments=updated_args))
            
            self.file_procedures[filename] = updated_procedures

    def generate_report(self, output_file: str = None, show_all_globals: bool = False, truncate_lists: bool = False, enhanced: bool = False, show_only_file: Optional[List[str]] = None, show_only_proc: Optional[List[str]] = None, hide_locals: bool = False, hide_ok: bool = False, color_mode: str = 'auto'):
        """Generate comprehensive analysis report with procedure-level analysis"""
        
        # Infer argument intents before generating the report
        self._infer_argument_intents()

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
        for filename in sorted(self.file_procedures.keys()):
            # Filtering logic for --show-only-file
            if show_only_file and Path(filename).name not in show_only_file:
                continue

            # Read the file content to get the lines for intent checking
            try:
                with open(filename, 'r', encoding='utf-8', errors='ignore') as f:
                    lines = f.read().splitlines()
            except IOError:
                lines = [] # If file can't be read, we can't check for explicit intents

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
                        
                        # Format arguments with intent
                        arg_parts = []
                        for arg in proc.arguments:
                            intent_str = ""
                            if arg.intent != 'UNKNOWN':
                                # Add a '*' to indicate inferred intent
                                is_inferred = arg.intent in ['IN', 'OUT', 'INOUT'] and not re.search(r'intent\s*\(\s*' + arg.intent.lower(), ' '.join(lines[proc.start_line:proc.end_line]), re.IGNORECASE)
                                inferred_marker = '*' if is_inferred else ''
                                intent_str = f"[{arg.intent}{inferred_marker}]"
                            arg_parts.append(f"{arg.name}{intent_str}")
                        add_line(f"Arguments: {', '.join(arg_parts)}", 2)
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
                                
                                # Group calls by called procedure name
                                call_counts = defaultdict(list)
                                for call in calls:
                                    call_counts[call.called_name].append(call)
                                
                                # Display each called procedure with its details
                                for called_name, call_list in sorted(call_counts.items()):
                                    call_count = len(call_list)
                                    call_type = call_list[0].call_type
                                    first_line = call_list[0].line_num
                                    
                                    # Show basic call info
                                    add_line(f"- {called_name} ({call_type}) called {call_count} time(s), first at line {first_line}", 3)
                                    
                                    # For enhanced reports, show argument patterns
                                    if enhanced and call_count <= 3:  # Only show details for a few calls
                                        for call in call_list[:3]:
                                            args_str = ', '.join(call.arguments[:3])  # Show first 3 arguments
                                            if len(call.arguments) > 3:
                                                args_str += ', ...'
                                            add_line(f"  Line {call.line_num}: {called_name}({args_str})", 4)
                                
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
            cprint(f"Report written to {output_file}", Color.GREEN, force_color=color_mode!='never')
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
        all_calls = defaultdict(list)

        # File-level data is not relevant for this summary, focus on procedures
        # Procedure-level
        for filename, procs in self.file_procedures.items():
            for proc in procs:
                # Assignments
                if filename in self.procedure_assignments and proc.name in self.procedure_assignments[filename]:
                    for assignment in self.procedure_assignments[filename][proc.name]:
                        all_assigned_vars.add(assignment.variable.lower())
                # Reads
                if filename in self.procedure_reads and proc.name in self.procedure_reads[filename]:
                    for var_name, _, _ in self.procedure_reads[filename][proc.name]:
                        all_read_vars.add(var_name.lower())
                # Calls
                if filename in self.procedure_calls and proc.name in self.procedure_calls[filename]:
                    for call in self.procedure_calls[filename][proc.name]:
                        all_calls[call.called_name.lower()].append(f"{proc.name} (in {Path(filename).name})")


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
                        # Filter out calls to variables
                        if call.called_name.lower() not in self.global_variables and not re.match(r'^\d', call.called_name):
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
    
    def analyze_global_variable_lifecycle(self, var_name: str):
        """Generate a complete lifecycle report for a global variable."""
        
        var_lower = var_name.lower()
        
        # Find the actual key in global_variables that matches (case-insensitive)
        actual_var_key = None
        for key in self.global_variables.keys():
            if key.lower() == var_lower:
                actual_var_key = key
                break
        
        if actual_var_key is None:
            return f"Variable '{var_name}' not found in global variables."
        
        
        events = []
        
        try:
            # Find declaration using the actual key
            var_info = self.global_variables[actual_var_key]

            
            module_location = self.module_locations.get(var_info.module, 'unknown')
            
            # Build the dictionary step by step
            event_dict = {}
            event_dict['type'] = 'DECLARED'
            event_dict['file'] = module_location
            event_dict['line'] = var_info.line_num
            event_dict['context'] = f"Module {var_info.module}"
            event_dict['details'] = f"{var_info.type} {var_info.name}"
                     
            events.append(event_dict)
 
            
        except Exception as e:
            import traceback
            traceback.print_exc()
            return f"Error processing variable declaration: {e}"
        
        
        # Find all direct reads and writes
        proc_count = 0
        for filename, procs in self.file_procedures.items():
            for proc in procs:
                proc_count += 1
                
                try:
                    # Direct assignments
                    if filename in self.procedure_assignments and proc.name in self.procedure_assignments[filename]:
                        assignments = self.procedure_assignments[filename][proc.name]
                        
                        for i, assignment in enumerate(assignments):
                            try:
                                if assignment.variable.lower() == var_lower:
                                    
                                    # Safely handle potentially None values
                                    rhs = str(assignment.rhs) if assignment.rhs is not None else ""
                                    context = str(assignment.context) if assignment.context is not None else str(assignment.variable)
                                    assignment_type = str(assignment.assignment_type) if assignment.assignment_type is not None else "UNKNOWN"
                                    
                                    rhs_display = rhs[:50] + "..." if len(rhs) > 50 else rhs
                                    
                                    assignment_event = {}
                                    assignment_event['type'] = 'MODIFIED_DIRECT'
                                    assignment_event['file'] = str(filename)
                                    assignment_event['procedure'] = str(proc.name)
                                    assignment_event['line'] = int(assignment.line_num)
                                    assignment_event['context'] = f"{context} = {rhs_display}"
                                    assignment_event['assignment_type'] = assignment_type
                                    
                                    events.append(assignment_event)
                                    
                            except Exception as e:
                                import traceback
                                traceback.print_exc()
                                continue
                    
                    # Direct reads
                    if filename in self.procedure_reads and proc.name in self.procedure_reads[filename]:
                        reads = self.procedure_reads[filename][proc.name]
                        
                        for i, read_item in enumerate(reads):
                            try:
                                if len(read_item) >= 3:
                                    var, line_num, read_type = read_item[0], read_item[1], read_item[2]
                                    if str(var).lower() == var_lower:
                                        
                                        read_event = {}
                                        read_event['type'] = 'READ'
                                        read_event['file'] = str(filename)
                                        read_event['procedure'] = str(proc.name)
                                        read_event['line'] = int(line_num)
                                        read_event['context'] = 'Variable used in expression'
                                        
                                        events.append(read_event)
                                        
                            except Exception as e:
                                import traceback
                                traceback.print_exc()
                                continue
                    
                except Exception as e:
                    import traceback
                    traceback.print_exc()
                    continue
        
        
        # Sort events by file and line number
        try:
            events.sort(key=lambda x: (x.get('file', ''), x.get('line', 0)))
        except Exception as e:
            import traceback
            traceback.print_exc()

        return self.generate_variable_diary_report(actual_var_key, events)

    def list_global_variables_matching(self, partial_name: str) -> List[str]:
        """Find global variables that match a partial name (case-insensitive)."""
        partial_lower = partial_name.lower()
        matches = []
        for var_name in self.global_variables.keys():
            if partial_lower in var_name.lower():
                matches.append(var_name)
        return sorted(matches)
    
    def generate_variable_diary_report(self, var_name: str, events: List[Dict]) -> str:
        """Generate a formatted diary report for a variable."""
        lines = []
        lines.append(f"VARIABLE LIFECYCLE DIARY: {var_name}")
        lines.append("=" * 60)
        lines.append("")  # FIX: was lines.append()
        
        if not events:
            lines.append("No events found for this variable.")
            return '\n'.join(lines)
        
        for event in events:
            try:
                event_type = event.get('type', 'UNKNOWN')
                file_name = event.get('file', 'unknown')
                line_num = event.get('line', 0)
                
                if event_type == 'DECLARED':
                    lines.append(f"📋 DECLARED in {file_name} at line {line_num}")
                    lines.append(f"   {event.get('details', 'No details')}")
                
                elif event_type == 'MODIFIED_DIRECT':
                    procedure = event.get('procedure', 'unknown')
                    lines.append(f"✏️  MODIFIED in {procedure}() at line {line_num}")
                    lines.append(f"   {event.get('context', 'No context')}")
                    lines.append(f"   Type: {event.get('assignment_type', 'UNKNOWN')}")
                
                elif event_type == 'MODIFIED_INDIRECT':
                    procedure = event.get('procedure', 'unknown')
                    lines.append(f"⚠️  POTENTIALLY MODIFIED in {procedure}() at line {line_num}")
                    lines.append(f"   {event.get('context', 'No context')}")
                
                elif event_type == 'READ':
                    procedure = event.get('procedure', 'unknown')
                    lines.append(f"👁️  READ in {procedure}() at line {line_num}")
                
                lines.append("")  # FIX: was lines.append()
                
            except Exception as e:
                lines.append(f"Error processing event: {e}")
                lines.append("")
        
        return '\n'.join(lines)

    def classify_variable_scope(self, var_name: str, filename: str, proc_name: Optional[str] = None) -> str:
        """Enhanced scope classification"""
        var_lower = var_name.lower()

        # Check if it's a procedure argument first
        if proc_name and filename in self.file_procedures:
            for proc in self.file_procedures[filename]:
                if proc.name == proc_name:
                    arg_names = [arg.name.lower() for arg in proc.arguments]
                    if var_lower in arg_names:
                        return "ARGUMENT"
                    break

        # Check if it's from a USE statement first
        if proc_name and filename in self.procedure_use_statements:
            if proc_name in self.procedure_use_statements[filename]:
                for module, only_list, _ in self.procedure_use_statements[filename][proc_name]:
                    if only_list == 'all':
                        # Check if this variable is from the imported module
                        if module in self.modules and var_lower in [v.lower() for v in self.modules[module]]:
                            return f"IMPORTED({module})"
                    else:
                        # Check if variable is in the ONLY list
                        only_vars = [v.strip().lower() for v in only_list.split(',')]
                        if var_lower in only_vars:
                            return f"IMPORTED({module})"
        
        # Then check if it's a global variable
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
    
    # NEW: Variable diary feature
    parser.add_argument('--trace-var', metavar='VARIABLE_NAME',
                       help='Generate a detailed lifecycle report for a specific global variable.')


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
    
    # Initialize analyzer and run analysis (existing code)
    analyzer = FortranVariableAnalyzer()
    analyzer.scan_directory_for_globals(codebase_path)
    
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
    
    # NEW: Handle variable diary request
    if args.trace_var:
        try:
            # Debug: show what global variables contain 'h'
            matches = analyzer.list_global_variables_matching(args.trace_var)
            diary_report = analyzer.analyze_global_variable_lifecycle(args.trace_var)
            if args.output:
                with open(args.output, 'w') as f:
                    f.write(diary_report)
                cprint(f"Variable diary written to {args.output}", Color.GREEN, force_color=args.color!='never')
            else:
                print(diary_report)
            return 0
        except Exception as e:
            cprint(f"Error generating variable diary: {e}", Color.FAIL, force_color=args.color=='always')
            if args.debug:
                import traceback
                traceback.print_exc()
            return 1
    
    # Generate regular report (existing code)
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