# Reflectivity-Acoustic

This repository implements the **reflectivity method** for computing synthetic seismograms in the acoustic approximation. The reflectivity method is a semi-analytical method that allows to solve the wave equation in **vertically layered media**. The goal is to leverage this assumption in order to propose a fast alternative to spectral elements or finite difference implementations.  

> This project is currently in progress

# Methodology 
The method operates in the Fourier in both space and time. The steps are as follows:
1. generate recursively a multi-layer reflection coefficient for all incidence angles (i.e. all spatial wavenumbers),
2. integrate over all incidence angles to obtain the frequency-domain response of the multi-layer stack. This amounts to evaluate a *Sommerfeld integral*.
3. Multiply the result by the source spectrum, and go back to the time domain via inverse Fourier transform (iFFT).

Additional details on how the method works are given below.

## 2D Green function in the wavenumber domain

To obtain the Green function of the layer stack in the frequency domain, we must evaluate a **Sommerfeld integral** over the horizontal wavenumber space $k_x$. This takes the form

$$
G(x,z,\omega)
= \int_{-\infty}^{\infty}
R(k_x, \omega) \,
\frac{e^{i k_z z} \, e^{i k_x x}}{2 i k_z} \,
\mathrm{d}k_x,
$$

where the vertical wavenumber is given by the dispersion relation

$$
k_z = \sqrt{k_0^2 - k_x^2},
\qquad k_0 = \frac{\omega}{v_p}.
$$

We select the **principal branch** of the square root so that evanescent waves satisfy $\mathrm{Im}(k_z) > 0$.  
The function $R(k_x,\omega)$ is called the **reflectivity map** of the layered stack. If $R=1$, we have the standard Green kernel solution $G(x, z, \omega) = \frac{i}{4} H_0^{(1)} (kr), \, r=\sqrt{x^2+z^2}$.

## Reflectivity of the Layer Stack

Using interface conditions (continuity of pressure and vertical velocity) at each layer interface, we can construct recursively the reflectivity map $R(k_x,\omega)$, which incorporates all the multiple reflections of the stack.

In addition, we can set a free surface condition at $z = 0$, which corresponds to impose a reflection coefficient $R_{\text{surface}} = -1.$ on the top of the stack.

## Numerical Considerations

- The integral is **singular** at $k_x = k_0$; propagating $k_x < k_0$ and evanescent $k_x > k_0$ regimes must be treated separately.
- introducing a small damping in the frequency range $\omega = \omega + i \alpha, \; \alpha > 0$ facilitates the quadrature. This artificial damping is removed _a posteriori_ after the iFFT. 
- The integrand is **highly oscillatory** at high frequencies and for large offsets. This calls the need for specialized quadrature schemes tailored to highly oscillatory integrands.
- The free surface introduces **resonances** in the reflectivity map.  

## Things to do 
- Extend the implementation to **3D physics**.
- Improve the quadrature for the Sommerfeld integral by using complex contour deformation.
- Extend to the elastic case

## References
1. Mallick, S., & Frazer, L. N. (1987). Practical aspects of reflectivity modeling. Geophysics, 52(10), 1355-1364.
2. Muller, G. (1985). The reflectivity method: a tutorial. Journal of Geophysics, 58(1), 153-174.

## License
This project is licensed under the MIT License - see the LICENSE file for details.

  

