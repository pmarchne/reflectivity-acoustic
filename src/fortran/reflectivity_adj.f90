module reflectivity_adj_mod
  use iso_fortran_env, only : real64
  use omp_lib
  implicit none
  private
  public :: compute_reflectivity_adj

  integer, parameter :: dp = real64

contains

  pure function csqrt_pos(z) result(r)
    complex(dp), intent(in) :: z
    complex(dp) :: r
    r = sqrt(z)
    if (aimag(r) < 0.0_dp) r = -r
  end function csqrt_pos

  subroutine compute_reflectivity_adj( &
      h, vp, rho, omegas, p, free_surface, zr, zs, &
      R, dR_dvp, dR_drho, nlay, nkq, nw )

    !f2py intent(hide) :: nlay, nkq, nw
    !f2py intent(in)  :: h, vp, rho, omegas, p, free_surface, zr, zs
    !f2py intent(out) :: R, dR_dvp, dR_drho
    !f2py real(8)     :: h, vp, rho, p, zr, zs
    !f2py integer     :: free_surface
    !f2py complex(16) :: omegas
    !f2py complex(16) :: R, dR_dvp, dR_drho

    integer, intent(in) :: nlay, nkq, nw, free_surface
    real(dp),    intent(in)  :: h(nlay), vp(nlay), rho(nlay)
    complex(dp),  intent(in)  :: omegas(nw)
    real(dp),     intent(in)  :: p(nkq)
    real(dp),     intent(in)  :: zr, zs
    complex(dp),  intent(out) :: R(nw, nkq)
    complex(dp),  intent(out) :: dR_dvp(nw, nkq, nlay)
    complex(dp),  intent(out) :: dR_drho(nw, nkq, nlay)

    real(dp), allocatable :: vp_inv2(:), vp_inv3(:), p2(:)

    integer :: iw, ik, ell
    complex(dp) :: omega, omega2
    complex(dp) :: k02_term
    complex(dp) :: Rval, rint, numerator, denom_update
    complex(dp) :: ghost, cavity
    complex(dp) :: dkz_dvp, dZ_dvp, dZ_drho, inv_kz

    complex(dp), parameter :: zero   = (0.0_dp, 0.0_dp)
    complex(dp), parameter :: one    = (1.0_dp, 0.0_dp)
    complex(dp), parameter :: two_i  = (0.0_dp, 2.0_dp)

    if (nlay < 1) return

    allocate(vp_inv2(nlay), vp_inv3(nlay), p2(nkq))

    !$OMP PARALLEL DO SIMD
    do ell = 1, nlay
      vp_inv2(ell) = 1.0_dp / (vp(ell) * vp(ell))
      vp_inv3(ell) = 1.0_dp / (vp(ell) * vp(ell) * vp(ell))
    end do
    !$OMP END PARALLEL DO SIMD

    !$OMP PARALLEL DO SIMD
    do ik = 1, nkq
      p2(ik) = p(ik) * p(ik)
    end do
    !$OMP END PARALLEL DO SIMD

    !$OMP PARALLEL DEFAULT(shared) PRIVATE( &
    !$OMP   iw, ik, ell, omega, omega2, k02_term, Rval, rint, numerator, denom_update, &
    !$OMP   ghost, cavity, dkz_dvp, dZ_dvp, dZ_drho, inv_kz )
    block
      complex(dp), allocatable :: kz(:), Z(:), Rstep(:), phase(:)
      complex(dp), allocatable :: adj_kz(:), adj_Z(:)
      complex(dp) :: adj_out, adj_cavity, adj_Rstep0, adj_ghost, adj_current
      complex(dp) :: adj_t, adj_s0
      complex(dp) :: t, q, x, D, B, invB2
      complex(dp) :: adj_rloc, adj_x, adj_q
      complex(dp) :: dghost_dkz

      allocate(kz(nlay), Z(nlay), Rstep(nlay), phase(nlay), adj_kz(nlay), adj_Z(nlay))

      !$OMP DO COLLAPSE(2) SCHEDULE(static)
      do iw = 1, nw
        do ik = 1, nkq

          adj_kz = zero
          adj_Z  = zero

          omega  = omegas(iw)
          omega2 = omega * omega

          ! ---- Bottom layer ----
          k02_term = omega2 * cmplx(vp_inv2(nlay) - p2(ik), 0.0_dp, dp)
          kz(nlay) = csqrt_pos(k02_term)
          inv_kz   = one / kz(nlay)
          Z(nlay)  = omega * cmplx(rho(nlay), 0.0_dp, dp) * inv_kz
          phase(nlay) = exp(two_i * kz(nlay) * h(nlay))
          Rstep(nlay) = zero

          ! ---- Upward recursion ----
          do ell = nlay - 1, 1, -1
            k02_term = omega2 * cmplx(vp_inv2(ell) - p2(ik), 0.0_dp, dp)
            kz(ell)  = csqrt_pos(k02_term)
            inv_kz   = one / kz(ell)
            Z(ell)   = omega * cmplx(rho(ell), 0.0_dp, dp) * inv_kz
            phase(ell) = exp(two_i * kz(ell) * h(ell))

            rint = (Z(ell + 1) - Z(ell)) / (Z(ell + 1) + Z(ell))
            q = phase(ell + 1)
            denom_update = one + rint * Rstep(ell + 1) * q
            numerator    = rint + Rstep(ell + 1) * q
            Rstep(ell)   = numerator / denom_update
          end do

          ! ---- Free surface boundary condition ----
          if (free_surface == 1) then
            cavity = one / (one + Rstep(1) * phase(1))
            ghost  = -4.0_dp * sin(kz(1) * zs) * sin(kz(1) * zr)
            Rval   = cavity * Rstep(1) * ghost
          else
            Rval = Rstep(1)
          end if

          R(iw, ik) = Rval

          ! ---- Reverse tape ----
          if (free_surface == 1) then
            adj_out = one

            adj_cavity = adj_out * Rstep(1) * ghost
            adj_Rstep0 = adj_out * cavity * ghost
            adj_ghost  = adj_out * cavity * Rstep(1)

            t = one + Rstep(1) * phase(1)
            adj_t = -adj_cavity / (t * t)

            adj_Rstep0 = adj_Rstep0 + adj_t * phase(1)
            adj_s0 = adj_t * Rstep(1)

            dghost_dkz = -4.0_dp * ( &
                cos(kz(1) * zs) * zs * sin(kz(1) * zr) + &
                sin(kz(1) * zs) * cos(kz(1) * zr) * zr )

            adj_kz(1) = adj_kz(1) + &
                adj_s0 * (two_i * h(1) * phase(1)) + &
                adj_ghost * dghost_dkz

            adj_current = adj_Rstep0
          else
            adj_current = one
          end if

          ! ---- Backward recursion over interfaces ----
          do ell = 1, nlay - 1
            x = Rstep(ell + 1)
            rint = (Z(ell + 1) - Z(ell)) / (Z(ell + 1) + Z(ell))
            q = phase(ell + 1)
            D = one + rint * x * q

            adj_rloc = adj_current * (one - (x * q) * (x * q)) / (D * D)
            adj_x    = adj_current * q * (one - rint * rint) / (D * D)
            adj_q    = adj_current * x * (one - rint * rint) / (D * D)

            B = Z(ell + 1) + Z(ell)
            invB2 = one / (B * B)

            adj_Z(ell + 1) = adj_Z(ell + 1) + adj_rloc * (2.0_dp * Z(ell)) * invB2
            adj_Z(ell)     = adj_Z(ell)     + adj_rloc * (-2.0_dp * Z(ell + 1)) * invB2

            adj_kz(ell + 1) = adj_kz(ell + 1) + adj_q * (two_i * h(ell + 1) * q)

            adj_current = adj_x
          end do

          ! ---- Convert layer adjoints to parameter derivatives ----
          do ell = 1, nlay
            inv_kz   = one / kz(ell)
            dkz_dvp  = -omega2 * vp_inv3(ell) * inv_kz
            dZ_dvp   = -Z(ell) * inv_kz * dkz_dvp
            dZ_drho  = omega * inv_kz

            dR_dvp(iw, ik, ell)  = adj_kz(ell) * dkz_dvp + adj_Z(ell) * dZ_dvp
            dR_drho(iw, ik, ell) = adj_Z(ell) * dZ_drho
          end do

        end do
      end do
      !$OMP END DO

      deallocate(kz, Z, Rstep, phase, adj_kz, adj_Z)
    end block
    !$OMP END PARALLEL

    deallocate(vp_inv2, vp_inv3, p2)

  end subroutine compute_reflectivity_adj

end module reflectivity_adj_mod