# Reflectivity-Acoustic

This repository implements the **reflectivity method** for computing synthetic seismograms in the acoustic approximation. The reflectivity method is a semi-analytical approach tailored to **vertically layered media**.   
The goal is to revisit the method in order to propose a cost-efficient alternative to spectral elements or finite difference schemes.  

The method works as follows:  
1. generate recursively a multi-layer reflection coefficient for all incidence angles,
2. integrate over all incidence angles to obtain the frequency-domain response of the multi-layer stack,
3. go back to the time domain via inverse Fourier transform.

Once installed, you should be able to directly run the notebooks and generated seismograms.

---

## Green’s Function in the Wavenumber Domain

To obtain the Green function of the layer stack in the frequency domain, we evaluate a **Sommerfeld integral** over the horizontal wavenumber $k_x$:

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
We select the **principal branch** of the square root so that evanescent waves satisfy  
$\operatorname{Im}(k_z) > 0$.  
The function $R(k_x,\omega)$ is the **reflectivity map** of the layered stack.

## Reflectivity of the Layer Stack

Using interface conditions (continuity of pressure and vertical velocity), we construct recursively the reflectivity and obtain the **effective reflection coefficient** \(R(k_x,\omega)\), which incorporates all multiple reflections and transmissions.

At the free surface \(z = 0\), we impose total reflection, which corresponds to a top reflectivity $R_{\text{surface}} = -1.$

This boundary condition is introduced at the top of the recursive reflectivity computation.

## High-Level Procedure

Given a set of frequencies \(\omega\) and horizontal wavenumbers \(k_x\):

1. Compute the reflectivity \(R(k_x,\omega)\) at the source depth.  
2. Assemble the Sommerfeld integral using numerical quadrature.  
3. Evaluate the integral for all source–receiver pairs.  
4. Multiply by the source spectrum and apply an inverse FFT to obtain time-domain seismograms.

---

## Numerical Considerations

- The integral is **singular**; introducing a small damping in frequency (complex \(\omega\)) stabilizes the integral.  
- Propagating (\(k_x < k_0\)) and evanescent (\(k_x > k_0\)) regimes must be treated separately.  
- The integrand becomes **highly oscillatory** at high frequencies and large offsets.  
- The free surface introduces **resonances** in the reflectivity map.  

---

## Roadmap

1. Compare results with a finite-difference wave solver.  
2. Extend the implementation to **3D physics**.  
3. Generate seismograms for large receiver arrays using **FFT acceleration**.  
4. Optimize computation of the reflectivity map.  
5. Compute the **exact gradient** of the forward model.  
6. Prepare for **MCMC sampling** by plotting cost functions.

