import unittest
import sys
import os

# Add current directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from fortran_variable_analyzer import FortranVariableAnalyzer

class TestFortranVariableAnalyzer(unittest.TestCase):
    
    def setUp(self):
        self.analyzer = FortranVariableAnalyzer()
    
    def test_clean_line(self):
        """Test comment removal"""
        line = "real :: var  ! This is a comment"
        cleaned = self.analyzer.clean_line(line)
        self.assertEqual(cleaned, "real :: var")
    
    def test_clean_line_with_quotes(self):
        """Test comment removal with quotes"""
        line = 'write(*,*) "This ! is not a comment"  ! But this is'
        cleaned = self.analyzer.clean_line(line)
        self.assertEqual(cleaned, 'write(*,*) "This ! is not a comment"')
    
    # Add more tests as needed

if __name__ == '__main__':
    unittest.main()
