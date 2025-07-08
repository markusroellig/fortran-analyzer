module sample_module
  use iso_fortran_env, only: real64
  implicit none
  
  real(real64), parameter :: pi = 3.14159265359_real64
  integer :: global_counter = 0

contains

  subroutine test_subroutine(input_val, output_val)
    real(real64), intent(in) :: input_val
    real(real64), intent(out) :: output_val
    
    real(real64) :: local_var
    integer :: i
    
    local_var = input_val * pi
    output_val = 0.0_real64
    
    do i = 1, 10
      output_val = output_val + local_var / real(i, real64)
    end do
    
    global_counter = global_counter + 1
  end subroutine test_subroutine

end module sample_module
