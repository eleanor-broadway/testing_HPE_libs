! example.f90
program main
    implicit none
    ! integer, parameter :: dp = selected_real_kind(15,300)
    real :: blas_scale
    real, allocatable, dimension(:) :: blas_array, B
    integer, allocatable, dimension(:) :: pivot
    real, allocatable, dimension(:,:) :: A
    integer :: blas_size, istat, ok

    !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    ! X = a*X
    !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

    print*, " "
    print*, " "
    print*, "Testing the BLAS library"
    print*, " "

    blas_scale = 5
    blas_size = 3

    allocate(blas_array(blas_size), stat=istat)
    if (istat.ne.0) stop 'Error: allocating blas'

    blas_array(1) = 1
    blas_array(2) = 2
    blas_array(3) = 3

    print*, "Original array =", blas_array(:)

    print*, " "
    print*, "Scaling the array using SSCAL:"
    call sscal(blas_size, blas_scale, blas_array, 1)
    print*, " "

    print*, "Array scaled by,", int(blas_scale), "=", blas_array(:)
    print*, " "
    print*, "DONE"
    print*, " "


    !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    ! A*X = B
    ! Where A, X and B are matrices
    ! A = 3.0 -1.0  B = 4.0
    !     1.0  6.0      6.0
    !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

    print*, " "
    print*, " "
    print*, "Testing the LAPACK library"
    print*, " "


    allocate(A(2,2), stat=istat)
    if (istat.ne.0) stop 'Error: allocating A'
    allocate(B(2), stat=istat)
    if (istat.ne.0) stop 'Error: allocating B'
    allocate(pivot(2), stat=istat)
    if (istat.ne.0) stop 'Error: allocating pivot'

    A(1,1) = 3.0
    A(1,2) = -1.0
    A(2,1) = 1.0
    A(2,2) = 6.0

    B(1) = 4.0
    B(2) = 6.0

    ! print*, A(:,:)
    ! print*, B(:)

    print*, "Solving a set of linear equations using SGESV"
    call SGESV(2,1,A,2,pivot,B,2,ok)
    print*, " "

    print*, "Correct answer: 1.57894742,  0.736842036"
    print*, "Answer:", B(:)
    print*, " "
    print*, "DONE"
    print*, " "


    deallocate(blas_array)
    deallocate(A)
    deallocate(B)
    deallocate(pivot)
end program main
