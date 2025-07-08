module mod_parameters
  use iso_fortran_env, only: real64
  implicit none
  
  ! Mathematical constants
  real(real64), parameter :: pi = 3.14159265359_real64
  real(real64), parameter :: e = 2.71828182846_real64
  
  ! Physical constants
  real(real64), parameter :: c_light = 2.99792458e8_real64  ! m/s
  real(real64), parameter :: h_planck = 6.62607015e-34_real64  ! J⋅s
  
  ! Configuration variables
  integer :: max_iterations = 1000
  real(real64) :: tolerance = 1.0e-6_real64
  logical :: debug_mode = .false.
  
end module mod_parameters
