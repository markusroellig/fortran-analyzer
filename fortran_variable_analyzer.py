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

ProcedureInfo = namedtuple('ProcedureInfo', [
    'name', 'type', 'start_line', 'end_line', 'module', 'arguments'
])

class FortranVariableAnalyzer:
    def __init__(self, mod_parameters_file: str = None):
        self.global_variables = {}
        self.modules = {}
        
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
        self.procedure_use_statements = defaultdict(lambda: defaultdict(list))
        
        if mod_parameters_file:
            self.parse_global_variables(mod_parameters_file)
    
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
    
    def parse_global_variables(self, filename: str):
        """Parse mod_parameters.f90 to extract global variable definitions"""
        print(f"Parsing global variables from {filename}...")
        
        try:
            with open(filename, 'r', encoding='utf-8', errors='ignore') as f:
                content = f.read()
        except Exception as e:
            print(f"Error reading {filename}: {e}")
            return
        
        current_module = None
        in_type_def = False
        
        lines = content.split('\n')
        for i, line in enumerate(lines, 1):
            line_clean = self.clean_line(line)
            line_lower = line_clean.lower()
            
            # Skip empty lines
            if not line_clean:
                continue
            
            # Module detection
            module_match = re.match(r'^\s*module\s+(\w+)', line_lower)
            if module_match and not line_lower.startswith('end module'):
                current_module = module_match.group(1)
                self.modules[current_module] = []
                continue
            
            # End module
            if re.match(r'^\s*end\s+module', line_lower):
                current_module = None
                continue
            
            # Type definition detection
            type_match = re.match(r'^\s*type\s*::\s*(\w+)', line_lower)
            if type_match:
                in_type_def = True
                continue
            
            if re.match(r'^\s*end\s+type', line_lower):
                in_type_def = False
                continue
            
            if current_module and not in_type_def:
                var_infos = self.parse_variable_declaration(line_clean, i, current_module)
                for var_info in var_infos:
                    self.global_variables[var_info.name] = var_info
                    self.modules[current_module].append(var_info.name)
    
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
                    arguments=arguments
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
        
        for i, line in enumerate(lines, 1):
            line_clean = self.clean_line(line)
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
                # Find the corresponding procedure info
                for proc in procedures:
                    if proc.name.lower() == proc_name and proc.start_line == i:
                        current_procedure = proc
                        break
                continue
            
            # Check if we're at the end of a procedure
            if current_procedure and i >= current_procedure.end_line:
                current_procedure = None
            
            # Determine analysis scope (file-level vs procedure-level)
            scope_key = self.get_scope_key(filename, current_procedure)
            
            # Parse USE statements
            use_match = re.match(r'^\s*use\s+(\w+)(?:\s*,\s*only\s*:\s*(.+))?', line_lower)
            if use_match:
                module_name = use_match.group(1)
                only_list = use_match.group(2) if use_match.group(2) else 'all'
                
                if current_procedure:
                    self.procedure_use_statements[filename][current_procedure.name].append((module_name, only_list, i))
                else:
                    self.use_statements[filename].append((module_name, only_list, i))
                continue
            
            # Parse variable declarations
            var_infos = self.parse_variable_declaration(line_clean, i, current_module or 'local')
            if current_procedure:
                self.procedure_variables[filename][current_procedure.name].extend(var_infos)
            else:
                self.file_variables[filename].extend(var_infos)
            
            # Parse assignments
            assignments = self.parse_assignments(line_clean, i, filename)
            if current_procedure:
                self.procedure_assignments[filename][current_procedure.name].extend(assignments)
            else:
                self.file_assignments[filename].extend(assignments)
            
            # Parse variable reads
            reads = self.parse_variable_reads(line_clean, i, filename)
            if current_procedure:
                self.procedure_reads[filename][current_procedure.name].extend(reads)
            else:
                self.file_reads[filename].extend(reads)
    
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
    
    def generate_report(self, output_file: str = None, show_all_globals: bool = False, truncate_lists: bool = False):
        """Generate comprehensive analysis report with procedure-level analysis"""
        report_lines = []
        
        def add_line(line: str = '', indent: int = 0):
            report_lines.append('  ' * indent + line)
        
        add_line("FORTRAN VARIABLE SOURCE ANALYSIS REPORT")
        add_line("=" * 50)
        add_line(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        add_line(f"Working Directory: {Path.cwd()}")
        add_line()
        
        # Global Variables Summary
        self._generate_global_summary(add_line, show_all_globals)
        
        # File-by-file analysis with procedure breakdown
        for filename in sorted(self.file_variables.keys()):
            add_line(f"FILE ANALYSIS: {Path(filename).name}")
            add_line("=" * 80)
            
            # File-level USE statements
            if filename in self.use_statements:
                add_line("FILE-LEVEL USE Statements:", 1)
                for module, only_list, line_num in self.use_statements[filename]:
                    add_line(f"Line {line_num:4d}: USE {module}", 2)
                    if only_list != 'all':
                        only_clean = only_list.replace('\n', ' ').replace('\t', ' ')
                        add_line(f"            ONLY: {only_clean[:60]}", 2)
                add_line()
            
            # File-level variable declarations
            if filename in self.file_variables and self.file_variables[filename]:
                add_line("FILE-LEVEL Variable Declarations:", 1)
                for var_info in self.file_variables[filename][:15]:
                    scope_status = self.classify_variable_scope(var_info.name, filename)
                    param_str = " [PARAM]" if var_info.is_parameter else ""
                    dims_str = f" {var_info.dimensions}" if var_info.dimensions else ""
                    add_line(f"Line {var_info.line_num:4d}: {var_info.name:<20} {var_info.type:<15}{dims_str} [{scope_status}]{param_str}", 2)
                
                if len(self.file_variables[filename]) > 15:
                    add_line(f"  ... and {len(self.file_variables[filename]) - 15} more declarations", 2)
                add_line()
            
            # Procedure-by-procedure analysis
            if filename in self.file_procedures:
                procedures = self.file_procedures[filename]
                if procedures:
                    add_line(f"PROCEDURES FOUND: {len(procedures)}", 1)
                    add_line()
                    
                    for proc in procedures:
                        add_line(f"{proc.type.upper()}: {proc.name} (lines {proc.start_line}-{proc.end_line})", 1)
                        add_line("-" * 60, 1)
                        
                        if proc.module:
                            add_line(f"Module: {proc.module}", 2)
                        add_line(f"Arguments: {proc.arguments}", 2)
                        add_line()
                        
                        # Procedure-level USE statements
                        if (filename in self.procedure_use_statements and 
                            proc.name in self.procedure_use_statements[filename]):
                            add_line("USE Statements:", 2)
                            for module, only_list, line_num in self.procedure_use_statements[filename][proc.name]:
                                add_line(f"Line {line_num:4d}: USE {module}", 3)
                                if only_list != 'all':
                                    only_clean = only_list.replace('\n', ' ').replace('\t', ' ')
                                    add_line(f"            ONLY: {only_clean[:50]}", 3)
                            add_line()
                        
                        # Procedure-level variable declarations
                        if (filename in self.procedure_variables and 
                            proc.name in self.procedure_variables[filename]):
                            vars_list = self.procedure_variables[filename][proc.name]
                            if vars_list:
                                add_line("Local Variable Declarations:", 2)
                                for var_info in vars_list[:15]:
                                    scope_status = self.classify_variable_scope(var_info.name, filename)
                                    param_str = " [PARAM]" if var_info.is_parameter else ""
                                    dims_str = f" {var_info.dimensions}" if var_info.dimensions else ""
                                    add_line(f"Line {var_info.line_num:4d}: {var_info.name:<18} {var_info.type:<12}{dims_str} [{scope_status}]{param_str}", 3)
                                
                                if len(vars_list) > 15:
                                    add_line(f"  ... and {len(vars_list) - 15} more declarations", 3)
                                add_line()
                        
                        # Procedure-level assignments
                        if (filename in self.procedure_assignments and 
                            proc.name in self.procedure_assignments[filename]):
                            assignments = self.procedure_assignments[filename][proc.name]
                            if assignments:
                                add_line("Variable Assignments:", 2)
                                assignment_counts = defaultdict(int)
                                
                                for assignment in assignments[:15]:
                                    assignment_counts[assignment.assignment_type] += 1
                                    scope_status = self.classify_variable_scope(assignment.variable, filename)
                                    
                                    rhs_display = assignment.rhs[:30] + "..." if len(assignment.rhs) > 30 else assignment.rhs
                                    add_line(f"Line {assignment.line_num:4d}: {assignment.variable:<18} = {rhs_display:<33} [{assignment.assignment_type[:12]}] [{scope_status}]", 3)
                                
                                if len(assignments) > 15:
                                    add_line(f"  ... and {len(assignments) - 15} more assignments", 3)
                                
                                add_line()
                                add_line("Assignment Type Summary:", 2)
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
                                
                                add_line("Most Frequently Read Variables:", 2)
                                sorted_reads = sorted(read_counts.items(), key=lambda x: x[1], reverse=True)
                                for var_name, count in sorted_reads[:12]:
                                    scope_status = self.classify_variable_scope(var_name, filename)
                                    add_line(f"  {var_name:<18}: {count:3d} reads [{scope_status}]", 3)
                                add_line()
                        
                        add_line()
            
            add_line()
        
        # Cross-reference analysis
        self._generate_cross_reference_analysis(add_line)
        
        report_text = '\n'.join(report_lines)
        
        if output_file:
            with open(output_file, 'w') as f:
                f.write(report_text)
            print(f"Report written to {output_file}")
        else:
            print(report_text)
    
    def _generate_global_summary(self, add_line, show_all_globals: bool):
        """Generate global variables summary"""
        add_line("GLOBAL VARIABLES SUMMARY (from mod_parameters.f90)")
        add_line("-" * 50)
        add_line(f"Total modules found: {len(self.modules)}")
        add_line(f"Total global variables: {len(self.global_variables)}")
        
        if show_all_globals:
            add_line("Showing ALL global variables (--show-all-globals specified)")
        else:
            add_line("Showing first 10 variables per module (use --show-all-globals for complete list)")
        add_line()
        
        for module_name, variables in self.modules.items():
            add_line(f"Module: {module_name} ({len(variables)} variables)")
            
            # Show all variables if requested, otherwise limit to 10
            display_vars = variables if show_all_globals else variables[:10]
            
            # Count usage statistics for this module
            used_count = 0
            modified_count = 0
            
            for var_name in display_vars:
                if var_name in self.global_variables:
                    var_info = self.global_variables[var_name]
                    param_str = " [PARAMETER]" if var_info.is_parameter else ""
                    alloc_str = " [ALLOCATABLE]" if var_info.is_allocatable else ""
                    dims_str = f" {var_info.dimensions}" if var_info.dimensions else ""
                    
                    # Check if variable is used in any analyzed file
                    usage_status = self._check_global_variable_usage(var_name)
                    
                    if "USED" in usage_status:
                        used_count += 1
                    if "MODIFIED" in usage_status:
                        modified_count += 1
                    
                    add_line(f"  {var_info.name:<20} {var_info.type:<15}{dims_str}{param_str}{alloc_str} [{usage_status}]", 1)
            
            if not show_all_globals and len(variables) > 10:
                add_line(f"  ... and {len(variables) - 10} more variables (use --show-all-globals to see all)", 1)
            
            # Add module usage summary
            usage_pct = (used_count / len(display_vars)) * 100 if display_vars else 0
            add_line(f"  Module Usage: {used_count}/{len(display_vars)} variables used ({usage_pct:.1f}%), {modified_count} modified", 1)
            add_line()
    
    def _generate_cross_reference_analysis(self, add_line):
        """Generate cross-reference analysis"""
        add_line("CROSS-REFERENCE ANALYSIS")
        add_line("-" * 30)
        
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
        
        add_line("Global variables being modified:", 1)
        modified_globals = all_assigned_vars.intersection(global_var_names)
        for var in sorted(modified_globals):
            add_line(f"  {var}", 2)
        
        add_line()
        add_line("Global variables being read:", 1)
        read_globals = all_read_vars.intersection(global_var_names)
        for var in sorted(read_globals)[:30]:
            add_line(f"  {var}", 2)
        
        add_line()
        add_line("Variables assigned but not in global scope:", 1)
        local_assigned = all_assigned_vars - global_var_names
        for var in sorted(local_assigned)[:30]:
            add_line(f"  {var}", 2)
    
    def classify_variable_scope(self, var_name: str, filename: str) -> str:
        """Enhanced scope classification"""
        var_lower = var_name.lower()
        
        # Check if it's a global variable
        if var_lower in [v.lower() for v in self.global_variables.keys()]:
            return "GLOBAL"
        
        # Check if it's a local variable in current file
        if filename in self.file_variables:
            local_vars = [v.name.lower() for v in self.file_variables[filename]]
            if var_lower in local_vars:
                return "FILE_LOCAL"
        
        # Check procedure-level variables
        if filename in self.procedure_variables:
            for proc_name, proc_vars in self.procedure_variables[filename].items():
                proc_var_names = [v.name.lower() for v in proc_vars]
                if var_lower in proc_var_names:
                    return f"PROC_LOCAL({proc_name})"
        
        # Check if it's from a USE statement
        if filename in self.use_statements:
            for module, only_list, _ in self.use_statements[filename]:
                if only_list != 'all':
                    only_vars = [v.strip().lower() for v in only_list.split(',')]
                    if var_lower in only_vars:
                        return f"IMPORTED_FROM_{module.upper()}"
        
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
    parser = argparse.ArgumentParser(
        description="Analyze Fortran source files for variable sources and dependencies with procedure-level analysis",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument('files', nargs='+', help='Fortran source files to analyze')
    parser.add_argument('--mod-parameters', '-m', 
                       default='mod_parameters.f90',
                       help='Path to mod_parameters.f90 file (default: mod_parameters.f90)')
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
    
    args = parser.parse_args()
    
    # Validate input files exist
    missing_files = [f for f in args.files if not Path(f).exists()]
    if missing_files:
        print(f"Error: The following files were not found: {missing_files}")
        return 1
    
    # Validate mod_parameters file
    if not Path(args.mod_parameters).exists():
        print(f"Warning: mod_parameters file not found: {args.mod_parameters}")
        args.mod_parameters = None
    
    # Initialize analyzer
    try:
        analyzer = FortranVariableAnalyzer(args.mod_parameters)
    except Exception as e:
        print(f"Error initializing analyzer: {e}")
        if args.debug:
            import traceback
            traceback.print_exc()
        return 1
    
    # Analyze each input file
    analyzed_count = 0
    for filename in args.files:
        try:
            analyzer.analyze_file(filename)
            analyzed_count += 1
        except Exception as e:
            print(f"Error analyzing {filename}: {e}")
            if args.debug:
                import traceback
                traceback.print_exc()
    
    if analyzed_count == 0:
        print("Error: No files were successfully analyzed")
        return 1
    
    print(f"Successfully analyzed {analyzed_count}/{len(args.files)} files")
    
    # Generate report
    try:
        analyzer.generate_report(args.output, args.show_all_globals, args.truncate)
        return 0
        
    except Exception as e:
        print(f"Error generating report: {e}")
        if args.debug:
            import traceback
            traceback.print_exc()
        return 1

if __name__ == "__main__":
    sys.exit(main())