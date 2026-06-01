module reflectivity_mod
  use iso_fortran_env, only : real64
  use omp_lib
  implicit none
  private
  public :: compute_reflectivity
  integer, parameter :: dp = real64
contains

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
    !f2py intent(in)   :: h, vp, rho, omegas, p, free_surface, zr, zs
    !f2py intent(out)  :: R
    !f2py real(8)      :: h, vp, rho, p, zr, zs
    !f2py integer      :: free_surface
    !f2py complex(16)  :: omegas
    !f2py complex(16)  :: R

    integer,     intent(in)  :: nlay, nkq, nw, free_surface
    real(dp),    intent(in)  :: h(nlay), vp(nlay), rho(nlay)
    complex(dp), intent(in)  :: omegas(nw)
    real(dp),    intent(in)  :: p(nkq)
    real(dp),    intent(in)  :: zr, zs
    complex(dp), intent(out) :: R(nw, nkq)

    ! Scalar temporaries (thread-private)
    integer     :: iw, ik, ell
    complex(dp) :: omega, omega2
    complex(dp) :: kz_cur, kz_next, Z_cur, Z_next, inv_kz
    complex(dp) :: Rval, rint, phase
    complex(dp) :: ghost, cavity
    complex(dp) :: k02_term, numerator, denom_update

    ! Thread-local - declared in parallel region via PRIVATE
    real(dp), allocatable :: vp_inv2(:)
    real(dp)              :: p2_ik

    complex(dp), parameter :: zero    = (0.0_dp, 0.0_dp)
    complex(dp), parameter :: one     = (1.0_dp, 0.0_dp)
    complex(dp), parameter :: two_i   = (0.0_dp, 2.0_dp)

    if (nlay < 1) return

    ! Outer loop over frequencies - each thread owns a full iw slice of R
    ! so writes to R(iw, :) never collide across threads.
    ! vp_inv2 is allocated once per thread (PRIVATE + local allocate).
    !$OMP PARALLEL DEFAULT(NONE) &
    !$OMP   SHARED(h, vp, rho, omegas, p, R, nlay, nkq, nw, &
    !$OMP          free_surface, zr, zs) &
    !$OMP   PRIVATE(iw, ik, ell, omega, omega2, k02_term, &
    !$OMP           kz_cur, kz_next, Z_cur, Z_next, inv_kz, &
    !$OMP           Rval, rint, phase, numerator, denom_update, &
    !$OMP           ghost, cavity, vp_inv2, p2_ik)

    ! Each thread allocates its own private vp_inv2
    allocate(vp_inv2(nlay))
    do ell = 1, nlay
      vp_inv2(ell) = 1.0_dp / (vp(ell) * vp(ell))
    end do

    !$OMP DO SCHEDULE(dynamic, 4)
    do iw = 1, nw
      omega  = omegas(iw)
      omega2 = omega * omega

      do ik = 1, nkq
        p2_ik = p(ik) * p(ik)   ! scalar computed on the fly

        ! ---- Bottom half-space ----
        k02_term = omega2 * cmplx(vp_inv2(nlay) - p2_ik, 0.0_dp, dp)
        kz_next  = csqrt_pos(k02_term)
        Z_next   = omega * cmplx(rho(nlay), 0.0_dp, dp) / kz_next
        Rval     = zero

        ! ---- Upward recursion through layers ----
        do ell = nlay - 1, 1, -1
          k02_term = omega2 * cmplx(vp_inv2(ell) - p2_ik, 0.0_dp, dp)
          kz_cur   = csqrt_pos(k02_term)
          Z_cur    = omega * cmplx(rho(ell), 0.0_dp, dp) / kz_cur

          rint  = (Z_next - Z_cur) / (Z_next + Z_cur)
          phase = exp(two_i * kz_next * h(ell+1))

          numerator    = rint + Rval * phase
          denom_update = one  + rint * Rval * phase
          Rval    = numerator / denom_update
          kz_next = kz_cur
          Z_next  = Z_cur
        end do

        ! ---- Free surface boundary condition ----
        if (free_surface == 1) then
          cavity = one / (one + Rval * exp(two_i * kz_next * h(1)))
          ghost  = cmplx(-4.0_dp * sin(aimag(kz_next * cmplx(0,1,dp)) + &
                   real(kz_next,dp)*zs) * &  ! keep real kz branch
                   sin(real(kz_next,dp)*zr), 0.0_dp, dp)
          ghost = cmplx(-4.0_dp, 0.0_dp, dp) * &
                  sin(kz_next * cmplx(zs, 0.0_dp, dp)) * &
                  sin(kz_next * cmplx(zr, 0.0_dp, dp))
          Rval = cavity * Rval * ghost
        end if

        R(iw, ik) = Rval
      end do   ! ik
    end do     ! iw
    !$OMP END DO

    deallocate(vp_inv2)
    !$OMP END PARALLEL

  end subroutine compute_reflectivity
end module reflectivity_mod