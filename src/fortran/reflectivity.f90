module reflectivity_mod
  use iso_fortran_env, only : real64
  use omp_lib
  implicit none
  private
  public :: compute_reflectivity

  integer, parameter :: dp = real64

contains

  ! Square root with positive imaginary part convention
  pure function csqrt_pos(z) result(r)
    complex(dp), intent(in) :: z
    complex(dp) :: r
    r = sqrt(z)
    if (aimag(r) < 0.0_dp) r = -r
  end function csqrt_pos

  subroutine compute_reflectivity( &
      h, vp, rho, omegas, p, free_surface, zr, zs, R, &
      nlay, nkq, nw )

    !f2py intent(hide) :: nlay, nkq, nw
    !f2py intent(in)  :: h, vp, rho, omegas, p, free_surface, zr, zs
    !f2py intent(out) :: R
    !f2py real(8)     :: h, vp, rho, p, zr, zs
    !f2py integer     :: free_surface
    !f2py complex(16):: omegas
    !f2py complex(16):: R

    integer, intent(in) :: nlay, nkq, nw, free_surface
    real(dp), intent(in) :: h(nlay), vp(nlay), rho(nlay)
    complex(dp), intent(in) :: omegas(nw)
    real(dp), intent(in) :: p(nkq)
    real(dp), intent(in) :: zr, zs
    complex(dp), intent(out) :: R(nw, nkq)

    ! Precomputed arrays
    real(dp), allocatable :: vp_inv2(:), p2(:)

    integer :: iw, ik, ell
    real(dp) :: dzminus, dzplus, abs_denom
    complex(dp) :: omega, omega2
    complex(dp) :: kz_cur, kz_next, Z_cur, Z_next
    complex(dp) :: Rval, rint, phase
    complex(dp) :: direct, image, Gsr, roundtrip, denominator
    complex(dp) :: k02_term, numerator, denom_update

    complex(dp), parameter :: zero    = (0.0_dp, 0.0_dp)
    complex(dp), parameter :: one     = (1.0_dp, 0.0_dp)
    complex(dp), parameter :: neg_one = (-1.0_dp, 0.0_dp)
    complex(dp), parameter :: i_unit  = (0.0_dp, 1.0_dp)
    complex(dp), parameter :: two_i   = (0.0_dp, 2.0_dp)
    real(dp),    parameter :: eps_denom = 1.0e-12_dp

    if (nlay < 1) return

    allocate(vp_inv2(nlay), p2(nkq))

    ! Precompute layer properties
    !$OMP PARALLEL DO SIMD
    do ell = 1, nlay
      vp_inv2(ell) = 1.0_dp / (vp(ell) * vp(ell))
    end do
    !$OMP END PARALLEL DO SIMD

    ! Precompute p^2
    !$OMP PARALLEL DO SIMD
    do ik = 1, nkq
      p2(ik) = p(ik) * p(ik)
    end do
    !$OMP END PARALLEL DO SIMD

    !$OMP PARALLEL DO PRIVATE( &
    !$OMP   ik, ell, omega, omega2, &
    !$OMP   k02_term, kz_cur, kz_next, Z_cur, Z_next, &
    !$OMP   Rval, rint, phase, numerator, denom_update, denominator, abs_denom, &
    !$OMP   dzminus, dzplus, direct, image, Gsr, roundtrip ) &
    !$OMP SCHEDULE(static) COLLAPSE(2)

    do iw = 1, nw
      do ik = 1, nkq

        omega  = omegas(iw)
        omega2 = omega * omega

        ! ---- Bottom layer ----
        ! Mimic numpy: sqrt(omega2 * (1/vp^2 - p^2) + 0j)
        k02_term = omega2 * cmplx(vp_inv2(nlay) - p2(ik), 0.0_dp, dp)
        kz_next = csqrt_pos(k02_term)
        
        ! Z = omega * rho / kz using native complex division like numpy
        Z_next = omega * cmplx(rho(nlay), 0.0_dp, dp) / kz_next
        Rval   = zero

        ! ---- Upward recursion through layers ----
        do ell = nlay - 1, 1, -1
          k02_term = omega2 * cmplx(vp_inv2(ell) - p2(ik), 0.0_dp, dp)
          kz_cur = csqrt_pos(k02_term)
          
          Z_cur = omega * cmplx(rho(ell), 0.0_dp, dp) / kz_cur

          ! Interface reflection coefficient
          rint = (Z_next - Z_cur) / (Z_next + Z_cur)

          ! Phase shift through layer
          phase = exp(two_i * kz_next * h(ell+1))
          
          ! Update reflection coefficient
          numerator = rint + Rval * phase
          denom_update = one + rint * Rval * phase
          Rval = numerator / denom_update

          kz_next = kz_cur
          Z_next  = Z_cur
        end do

        ! ---- Free surface boundary condition ----
        if (free_surface == 1) then
          dzminus = abs(zr - zs)
          dzplus  = abs(zr + zs)

          direct = exp(i_unit * kz_next * dzminus)
          image  = neg_one * exp(i_unit * kz_next * dzplus)
          Gsr    = direct + image

          roundtrip = Rval * neg_one * exp(two_i * kz_next * h(1))
          denominator = one - roundtrip

          ! Match numpy's threshold: replace small denominators with eps
          abs_denom = abs(denominator)
          if (abs_denom < eps_denom) then
            denominator = cmplx(eps_denom, 0.0_dp, dp)
          end if

          ! Apply free surface correction
          Rval = (Rval / denominator) * Gsr - direct
        end if

        R(iw, ik) = Rval

      end do
    end do
    !$OMP END PARALLEL DO

    deallocate(vp_inv2, p2)

  end subroutine compute_reflectivity

end module reflectivity_mod