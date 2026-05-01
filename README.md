# Reflectivity-Acoustic

This repository implements the **reflectivity method** for computing synthetic seismograms within the acoustic approximation.

The reflectivity method is a semi-analytical approach for solving the wave equation in **vertically layered media**. By leveraging the 1D nature of the velocity model, this implementation provides a fast alternative to grid-based (FD, SEM) methods.

> **Status:** Project in progress

---

## Methodology

The solver operates in the Fourier frequency-wavenumber ($\omega-k_x$) domain. The three primary steps are:

1. **Recursive reflection coefficient:** Compute a multi-layer reflection coefficient for all incidence angles (spatial wavenumbers).
2. **Sommerfeld Integration:** Integrate over all incidence angles to obtain the frequency-domain response of the stack.
3. **Time-Domain Transformation:** Multiply by the source spectrum and apply an Inverse Fast Fourier Transform (iFFT).

### 2D Green's Function in the Wavenumber Domain
The frequency-domain Green's function for a layered stack is obtained by evaluating a **Sommerfeld integral**. or 2D physics, it reads:

$$
G(x, z, \omega) = \int_{-\infty}^{\infty} R(k_x, \omega) \frac{e^{i k_z z} e^{i k_x x}}{2 i k_z} \mathrm{d}k_x
$$

The vertical wavenumber $k_z$ is defined by the dispersion relation:
$$k_z = \sqrt{k_0^2 - k_x^2}, \quad k_0 = \frac{\omega}{v_p}$$
We select the **principal branch** of the square root to ensure evanescent waves satisfy $\mathrm{Im}(k_z) > 0$. 

> **Note:** If $R=1$ (homogeneous space), the integral simplifies to the standard 2D Green's kernel: $G(x, z, \omega) = \frac{i}{4} H_0^{(1)} (k r)$.

### Reflectivity of the Layer Stack
The reflectivity map $R(k_x, \omega)$ is constructed recursively using interface conditions (continuity of pressure and vertical velocity). 
- **Multiples:** The recursive formulation naturally incorporates all internal multiple reflections.
- **Free Surface:** A free-surface condition at $z=0$ can be enabled, imposing a reflection coefficient of $R = -1$.

---

## Numerical Considerations

* **Singularities:** The integrand is singular at the critical wavenumber $k_x = k_0$. Propagating $k_x < k_0$ and evanescent $k_x > k_0$ regimes are treated separately in the code.
* **Complex Frequencies:** Small damping ($\omega = \omega + i \alpha$) is introduced to facilitate quadrature. This artificial damping is removed after the iFFT.
* **Oscillations:** The integrand becomes highly oscillatory at high frequencies and large offsets, requiring specialized quadrature schemes.
* **Resonances:** The free surface introduces strong resonances in the reflectivity map that must be handled carefully.

---

## Roadmap
- Extend implementation to **3D physics**.
- Implement **complex contour deformation** for the Sommerfeld integral.
- Extend to **Elastic** wave propagation.
- scale correctly with the density
  
## References
1. Mallick, S., & Frazer, L. N. (1987). Practical aspects of reflectivity modeling. Geophysics, 52(10), 1355-1364.
2. Muller, G. (1985). The reflectivity method: a tutorial. Journal of Geophysics, 58(1), 153-174.

## License
This project is licensed under the MIT License - see the LICENSE file for details.

  

