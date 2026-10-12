####################################################################################################
#                                          processing.py                                           #
####################################################################################################
#                                                                                                  #
# Authors: J. P. Merkofer (j.p.merkofer@tue.nl)                                                    #
#                                                                                                  #
# Created: 12/10/26                                                                                #
#                                                                                                  #
# Purpose: The raw spectroscopy pipeline of Augmentrum's torch engine (FSL-MRS estimates: wSVD     #
#          coil combination, spectral registration, unlike-transient removal, averaging, eddy      #
#          current correction, shift to creatine, zero-order phase) written once on                #
#          nifti_mrs_plus.ops, so it runs batched on NumPy, PyTorch, JAX and TensorFlow arrays.    #
#                                                                                                  #
####################################################################################################


#*************#
# imports                                                                                          #
#*************#
from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np
from nifti_mrs_plus import ops

NOISE_FRACTION = 0.1  # FSL-MRS estimate_noise_cov: the last tenth of every FID is noise
MIN_SAMPLES_PER_COIL = 10  # fewer noise samples per coil: no prewhitening (FSL-MRS)
PROTON_REFERENCE = 4.65  # ppm at the spectrometer frequency (FSL-MRS default for 1H)


#**************************************************************************************************#
#                                              Backend                                             #
#**************************************************************************************************#
#                                                                                                  #
# The few operations nifti_mrs_plus.ops does not have, on whichever backend an array lives:        #
# sorting, cumulative sums, arg-minima, minima and maxima, norms and phasors.                      #
#                                                                                                  #
#**************************************************************************************************#
class Backend:
    @staticmethod
    def sort(x, axis):
        if ops.is_torch(x):
            return x.sort(dim=axis).values
        if ops.is_tf(x):
            import tensorflow as tf
            return tf.sort(x, axis=axis)
        return ops.namespace(x).sort(x, axis=axis)

    @staticmethod
    def cumsum(x, axis):
        if ops.is_torch(x):
            return x.cumsum(dim=axis)
        if ops.is_tf(x):
            import tensorflow as tf
            return tf.math.cumsum(x, axis=axis)
        return ops.namespace(x).cumsum(x, axis=axis)

    @staticmethod
    def argmin(x, axis):
        if ops.is_torch(x):
            return x.argmin(dim=axis)
        if ops.is_tf(x):
            import tensorflow as tf
            return tf.argmin(x, axis=axis)
        return ops.namespace(x).argmin(x, axis=axis)

    @staticmethod
    def minimum(a, b):
        return ops.namespace(a).minimum(a, b)

    @staticmethod
    def maximum(a, b):
        return ops.namespace(a).maximum(a, b)

    @staticmethod
    def sign(x):
        return ops.namespace(x).sign(x)

    @staticmethod
    def remainder(x, y):
        if ops.is_tf(x):
            import tensorflow as tf
            return tf.math.floormod(x, y)
        if ops.is_torch(x):
            return x.remainder(y)
        return ops.namespace(x).remainder(x, y)

    @staticmethod
    def norm(x, axis, keepdims=False):
        return ops.sqrt(ops.sum(ops.real(x * ops.conj(x)) if ops.is_complex(x) else x * x,
                                axis=axis, keepdims=keepdims))

    @staticmethod
    def phasor(angle):
        """exp(i angle) for real *angle*."""
        return ops.complex_from(ops.cos(angle), ops.sin(angle))

    @staticmethod
    def zeros(ref, shape, dtype="float64"):
        return ops.full_like_shape(ref, shape, 0.0, dtype=dtype)

    @staticmethod
    def const(ref, values, dtype="float64"):
        """A NumPy-built constant on *ref*'s backend."""
        return ops.asarray_like(ref, np.asarray(values), dtype=dtype)


#**************************************************************************************************#
#                                             Spectral                                             #
#**************************************************************************************************#
#                                                                                                  #
# FSL-MRS's transform and axis: FIDToSpec halves the first point and takes the orthonormal FFT,    #
# fftshifted; the ppm axis is linspace(-sw / 2, sw / 2, n) / sf + reference.                       #
#                                                                                                  #
#**************************************************************************************************#
class Spectral:
    @staticmethod
    def halve_first(fids):
        n = ops.shape(fids)[-1]
        scale = np.ones(n)
        scale[0] = 0.5
        return fids * ops.cast_like(Backend.const(fids, scale), fids)

    @classmethod
    def fid_to_spec(cls, fids):
        n = ops.shape(fids)[-1]
        return ops.fftshift(ops.fft(cls.halve_first(fids))) / math.sqrt(n)

    @staticmethod
    def zero_fill(fids, factor):
        shape = ops.shape(fids)
        zeros = ops.cast_like(Backend.zeros(fids, shape[:-1] + ((factor - 1) * shape[-1],)), fids)
        return ops.concatenate([fids, zeros], axis=-1)

    @staticmethod
    def ppm_axis(n, sw_hz, sf_mhz, reference=PROTON_REFERENCE):
        return np.linspace(-sw_hz / 2, sw_hz / 2, n) / sf_mhz + reference

    @classmethod
    def window(cls, n, sw_hz, sf_mhz, limits, reference=PROTON_REFERENCE):
        """First and last bin of *limits* (ppm) on the axis (FSL-MRS limit_to_range)."""
        axis = cls.ppm_axis(n, sw_hz, sf_mhz, reference)
        first = int(np.argmin(np.abs(axis - limits[0])))
        last = int(np.argmin(np.abs(axis - limits[1])))
        return (first, last) if first <= last else (last, first)


#**************************************************************************************************#
#                                         CoilCombination                                          #
#**************************************************************************************************#
#                                                                                                  #
# FSL-MRS wSVD: the noise covariance C from the last tenth of every FID (np.cov over voxels,       #
# transients and points); a whitening L with L L^H = C (here from the eigendecomposition; the      #
# weights do not depend on which L is used); the principal eigenvector v of L^-1 G L^-H, G the     #
# Gram matrix X^H X of the reference (the water averaged over its transients, else each transient  #
# itself); weights L^-H v, scaled by |L v| and with the phase of the first coil's amplitude        #
# removed. Without enough noise samples (fewer than ten per coil) there is no whitening.           #
#                                                                                                  #
#**************************************************************************************************#
class CoilCombination:
    @staticmethod
    def noise_covariance(x):
        """x: (B, V, C, D, T) -> covariance (B, C, C) and the number of samples."""
        b, v, c, d, n = ops.shape(x)
        tail = n - int((1 - NOISE_FRACTION) * n)
        samples = ops.reshape(ops.transpose(x[..., n - tail:], (0, 1, 3, 4, 2)),
                              (b, v * d * tail, c))
        centred = samples - ops.mean(samples, axis=1, keepdims=True)
        cov = ops.matmul(ops.transpose(centred, (0, 2, 1)), ops.conj(centred))
        return cov / max(v * d * tail - 1, 1), v * d * tail

    @staticmethod
    def gram(reference):
        """X^H X of reference FIDs laid out (..., T, C)."""
        perm = tuple(range(len(ops.shape(reference)) - 2)) + (-1, -2)
        perm = tuple(p % len(ops.shape(reference)) for p in perm)
        return ops.matmul(ops.conj(ops.transpose(reference, perm)), reference)

    @staticmethod
    def weights(gram, cov, whiten):
        """Weights (..., C) from Gram matrices (B, ..., C, C) and covariances (B, C, C)."""
        c = ops.shape(cov)[-1]
        lead = len(ops.shape(gram)) - 3
        if c == 1:
            return ops.cast_like(ops.full_like_shape(gram, ops.shape(gram)[:-1], 1.0), gram)
        if whiten:
            values, vectors = ops.eigh(cov)
            values = ops.clip(ops.real(values), 1e-300, None)
            root = ops.sqrt(values)
        else:
            vectors = ops.cast_like(Backend.const(cov, np.broadcast_to(np.eye(c), ops.shape(cov))),
                                    cov)
            root = ops.full_like_shape(cov, ops.shape(cov)[:-1], 1.0)
        # L = U diag(root): broadcast over the reference's leading axes
        shape = (ops.shape(cov)[0],) + (1,) * lead + (c, c)
        u = ops.reshape(vectors, shape)
        r = ops.cast_like(ops.reshape(root, shape[:-1]), gram)
        uh = ops.conj(ops.transpose(u, tuple(range(len(shape) - 2)) + (len(shape) - 1,
                                                                        len(shape) - 2)))
        whitened = ops.matmul(ops.matmul(uh, gram), u) / (r[..., :, None] * r[..., None, :])
        _, eigenvectors = ops.eigh(whitened)
        principal = eigenvectors[..., -1]  # largest eigenvalue last (ascending)
        amplitude = ops.matmul(u, (r * principal)[..., None])[..., 0]  # L v
        first = amplitude[..., :1]
        rescale = Backend.norm(amplitude, -1, keepdims=True) * ops.conj(first) / ops.abs(first)
        return ops.matmul(u, (principal / r)[..., None])[..., 0] * rescale  # L^-H v

    @classmethod
    def combine(cls, x, w):
        """x (B, V, C, D, T), water w like it or None -> both combined (C = 1), weights."""
        b, v, c, d, n = ops.shape(x)
        cov, samples = cls.noise_covariance(x)
        whiten = samples >= MIN_SAMPLES_PER_COIL * c
        if w is None:
            gram = cls.gram(ops.transpose(x, (0, 1, 3, 4, 2)))            # (B, V, D, C, C)
            weights = cls.weights(gram, cov, whiten)                       # (B, V, D, C)
            combined = ops.sum(x * ops.transpose(weights, (0, 1, 3, 2))[..., None], axis=2)
            return combined[:, :, None], None, weights
        gram = cls.gram(ops.transpose(ops.mean(w, axis=3), (0, 1, 3, 2)))  # (B, V, C, C)
        weights = cls.weights(gram, cov, whiten)                           # (B, V, C)
        scale = weights[:, :, :, None, None]
        return (ops.sum(x * scale, axis=2)[:, :, None], ops.sum(w * scale, axis=2)[:, :, None],
                weights)


#**************************************************************************************************#
#                                           ShiftProfile                                           #
#**************************************************************************************************#
#                                                                                                  #
# The FSL-MRS alignment cost of every transient as an exact function of its frequency shift nu     #
# (cycles per sample): cost^2 = E(nu) + |Y|^2 - 2 Re(e^{-i phi} K(nu)), K the in-window cross-     #
# correlation with the target, E the in-window energy of the shifted transient; both are           #
# trigonometric polynomials evaluated as dot products of weighted rows with cos and sin.           #
#                                                                                                  #
#**************************************************************************************************#
class ShiftProfile:
    def __init__(self, x, target, first, last):
        n = ops.shape(x)[-1]
        mask = np.zeros(n)
        mask[first:last] = 1.0
        spec = Spectral.fid_to_spec(target)
        windowed = spec * ops.cast_like(Backend.const(spec, mask), spec)
        self.y_energy = ops.sum(ops.real(windowed * ops.conj(windowed)), axis=-1)
        self.norm = Backend.norm(target, -1)
        window = math.sqrt(n) * ops.ifft(ops.ifftshift(windowed))
        xt = Spectral.halve_first(x)
        h = xt * ops.conj(window)[:, None, :]
        spectrum = ops.fft(Spectral.zero_fill(xt, 2))
        power = ops.cast_like(ops.real(spectrum) ** 2 + ops.imag(spectrum) ** 2, spectrum)
        autocorr = ops.ifft(power)[..., :n]
        q = autocorr * ops.cast_like(Backend.const(x, self.kernel(n, first, last),
                                                   "complex128"), autocorr)
        self.q0 = ops.real(q[..., 0])
        rest = np.ones(n)
        rest[0] = 0.0
        q = q * ops.cast_like(Backend.const(q, rest), q)
        self.lag = Backend.const(ops.real(x), np.arange(n, dtype=float))
        index = self.lag + 1
        rows = []
        for c, w in ((h, index), (q, self.lag)):
            for part in (ops.real(c), ops.imag(c)):
                rows += [part, part * w, part * w ** 2]
        self.rows = ops.stack(rows, axis=-2)                               # (B, D, 12, T)
        self.plain = ops.stack(rows[0::3], axis=-2)                        # moment 0 only

    @staticmethod
    def kernel(n, first, last):
        """w_l = (1/n) sum over the window's bins of e^{-2 pi i l kappa / n}."""
        indicator = np.zeros(n)
        indicator[np.fft.fftshift(np.arange(n))[first:last]] = 1.0
        return np.fft.fft(indicator) / n

    @staticmethod
    def combine(cos, sin, moments):
        """The complex moment sums: (re + i im) e^{-i theta} = (re cos + im sin) + i (...)."""
        def sums(offset):
            return [ops.complex_from(cos[..., offset + j] + sin[..., offset + moments + j],
                                     cos[..., offset + moments + j] - sin[..., offset + j])
                    for j in range(moments)]
        return sums(0), sums(2 * moments)

    def shared(self, nus):
        """K and E at shifts every transient shares, (P,) -> each (B, D, P)."""
        theta = 2 * math.pi * self.lag[:, None] * nus[None, :]
        cos = ops.transpose(ops.matmul(self.plain, ops.cos(theta)), (0, 1, 3, 2))
        sin = ops.transpose(ops.matmul(self.plain, ops.sin(theta)), (0, 1, 3, 2))
        k, e = self.combine(cos, sin, 1)
        lead = Backend.phasor(-2 * math.pi * nus)
        return lead * k[0], self.q0[..., None] + 2 * ops.real(e[0])

    def at(self, nu, derivatives=False):
        """K and E at one shift per transient (B, D); with derivatives ((K, K', K''), (E, ...))."""
        theta = 2 * math.pi * self.lag * nu[..., None]
        rows = self.rows if derivatives else self.plain
        cos = ops.matmul(rows, ops.cos(theta)[..., None])[..., 0]
        sin = ops.matmul(rows, ops.sin(theta)[..., None])[..., 0]
        k, e = self.combine(cos, sin, 3 if derivatives else 1)
        rate = 2 * math.pi
        lead = Backend.phasor(-rate * nu)
        k0, e0 = lead * k[0], self.q0 + 2 * ops.real(e[0])
        if not derivatives:
            return k0, e0
        return ((k0, lead * k[1] * (-1j * rate), lead * k[2] * (-rate ** 2)),
                (e0, 2 * ops.real(e[1] * (-1j * rate)), 2 * ops.real(e[2]) * (-rate ** 2)))

    def locked(self, phi, k, e):
        y = ops.reshape(self.y_energy, (-1,) + (1,) * (len(ops.shape(k)) - 1))
        return e + y - 2 * ops.real(Backend.phasor(-phi) * k)

    def cost(self, phi, k, e):
        norm = ops.reshape(self.norm, (-1,) + (1,) * (len(ops.shape(k)) - 1))
        return ops.sqrt(ops.clip(self.locked(phi, k, e), 0.0, None)) / norm


#**************************************************************************************************#
#                                           Registration                                           #
#**************************************************************************************************#
#                                                                                                  #
# FSL-MRS phase_freq_align: each transient against the one nearest the mean, Powell from zero, run #
# twice. Its decisive steps are followed exactly (the phase line search ends at the closed-form    #
# phase; the shift line search opens scipy's bracket from (0, 1 Hz) and hands it to Brent), then   #
# Newton steps converge within a bin; every cost is exact (ShiftProfile).                          #
#                                                                                                  #
#**************************************************************************************************#
class Registration:
    GOLD = 1.618034  # scipy.optimize.bracket
    GROW_LIMIT = 110.0
    CGOLD = 0.3819660  # scipy Brent
    BRENT_MINTOL = 1.0e-11
    BRENT_TOL = 1.0e-2

    @classmethod
    def align(cls, fids, sw_hz, sf_mhz, ppmlim, reference=PROTON_REFERENCE, passes=2,
              free_steps=(2, 3)):
        """fids (B, D, T) -> (phi rad, eps Hz), each (B, D)."""
        x = ops.cast(fids, "complex128")
        b, d, n = ops.shape(x)
        first, last = Spectral.window(n, sw_hz, sf_mhz, ppmlim, reference)
        reach = (sw_hz / 4) / sw_hz
        avg = ops.mean(x, axis=1, keepdims=True)
        dist = Backend.norm(x - avg, -1)
        # the first of the transients nearest the mean (ties within rounding, as with two)
        near = dist <= -ops.amax(-dist, axis=-1, keepdims=True) * (1 + 1e-9)
        nearest = ops.argmax(ops.cast(near, "int32"), axis=-1)
        target = ops.take_along(x, ops.reshape(nearest, (b, 1, 1)) * ops.cast(
            ops.full_like_shape(dist, (b, 1, n), 1.0), "int64"), axis=1)[:, 0]
        profile = ShiftProfile(x, target, first, last)
        nu = Backend.zeros(dist, (b, d))
        k, _ = profile.at(nu)
        for step in range(passes):
            phi = ops.angle(k)
            base = None if step == 0 else nu
            xa, xb, xc, fa, fb, fc, expanding = cls.bracket(profile, phi, sw_hz, base)
            valid = ((((fb < fc) & (fb <= fa)) | ((fb < fa) & (fb <= fc)))
                     & (((xa < xb) & (xb < xc)) | ((xc < xb) & (xb < xa))))
            points, costs = ops.stack([xa, xb, xc]), ops.stack([fa, fb, fc])
            best = ops.take_along(points, Backend.argmin(costs, 0)[None], axis=0)[0]
            offset = nu if step else Backend.zeros(dist, (b, d))
            usable = valid | expanding
            start = offset + ops.where(usable, xb, best) / sw_hz
            low = ops.where(usable, offset + Backend.minimum(xa, xc) / sw_hz, start)
            high = ops.where(usable, offset + Backend.maximum(xa, xc) / sw_hz, start)
            low = ops.where(expanding & (xc < xa), low * 0 - reach, low)
            high = ops.where(expanding & (xc > xa), high * 0 + reach, high)
            found, left, right = cls.brent(profile, phi, sw_hz, (xa, xb, xc, fb), base)
            start = ops.where(valid, offset + found / sw_hz, start)
            low = ops.where(valid, offset + left / sw_hz, low)
            high = ops.where(valid, offset + right / sw_hz, high)
            nu = start
            steps = free_steps[min(step, len(free_steps) - 1)]
            nu = cls.newton(profile, nu, nu - 1.0 / n, nu + 1.0 / n, 0.5 / sw_hz, steps)
            k, _ = profile.at(nu)
        phi, eps = ops.angle(k), nu * sw_hz
        if d < 2:
            return phi * 0, eps * 0
        return phi, eps

    @classmethod
    def bracket(cls, profile, phi, sw_hz, base):
        """scipy.optimize.bracket from (0, 1 Hz) on the phase-locked cost (one expansion)."""
        def cost(eps):
            nu = eps / sw_hz if base is None else base + eps / sw_hz
            return profile.cost(phi, *profile.at(nu))

        gold = cls.GOLD
        if base is None:
            probes = Backend.const(phi, np.array([0.0, 1.0, 1.0 + gold, -gold]) / sw_hz)
            k, e = profile.shared(probes)
            values = profile.cost(phi[..., None], k, e)
            f0, f1, f_pos, f_neg = (values[..., i] for i in range(4))
        else:
            f0, f1 = cost(base * 0), cost(base * 0 + 1)
        swap = f0 < f1
        xa = ops.cast(swap, "float64")
        xb = 1.0 - xa
        xc = (1.0 + gold) - (1.0 + 2 * gold) * xa
        fa, fb = ops.where(swap, f1, f0), ops.where(swap, f0, f1)
        fc = ops.where(swap, f_neg, f_pos) if base is None else cost(xc)
        expanding = fc < fb

        tmp1 = (xb - xa) * (fb - fc)
        tmp2 = (xb - xc) * (fb - fa)
        val = tmp2 - tmp1
        denom = ops.where(ops.abs(val) < 1e-21, val * 0 + 2e-21, 2.0 * val)
        w = xb - ((xb - xc) * tmp2 - (xb - xa) * tmp1) / denom
        wlim = xb + cls.GROW_LIMIT * (xc - xb)
        inside = (w - xc) * (xb - w) > 0
        limit = ~inside & ((w - wlim) * (wlim - xc) >= 0)
        beyond = ~inside & ~limit & ((w - wlim) * (xc - w) > 0)
        p1 = ops.where(inside | beyond, w, ops.where(limit, wlim, xc + gold * (xc - xb)))
        f_p1 = cost(p1)
        take_b = inside & (f_p1 < fc)
        take_c = inside & ~take_b & (f_p1 > fb)
        closed = take_b | take_c
        falling = beyond & (f_p1 < fc)
        second = (inside & ~closed) | falling
        mid_b, mid_c = ops.where(falling, xc, xb), ops.where(falling, p1, xc)
        mid_fb, mid_fc = ops.where(falling, fc, fb), ops.where(falling, f_p1, fc)
        p2 = mid_c + gold * (mid_c - mid_b)
        f_p2 = cost(p2)
        w, fw = ops.where(second, p2, p1), ops.where(second, f_p2, f_p1)
        na = ops.where(closed, ops.where(take_b, xb, xa), mid_b)
        nb = ops.where(closed, ops.where(take_b, p1, xb), mid_c)
        nc = ops.where(closed, ops.where(take_c, p1, xc), w)
        nfa = ops.where(closed, ops.where(take_b, fb, fa), mid_fb)
        nfb = ops.where(closed, ops.where(take_b, f_p1, fb), mid_fc)
        nfc = ops.where(closed, ops.where(take_c, f_p1, fc), fw)
        xa, xb, xc = (ops.where(expanding, new, old) for new, old in ((na, xa), (nb, xb), (nc, xc)))
        fa, fb, fc = (ops.where(expanding, new, old)
                      for new, old in ((nfa, fa), (nfb, fb), (nfc, fc)))
        expanding = expanding & ~closed & (fc < fb)
        return xa, xb, xc, fa, fb, fc, expanding

    @classmethod
    def brent(cls, profile, phi, sw_hz, bracket, base, iterations=2):
        """scipy's Brent inside *bracket*, its first *iterations* steps (in Hz from base)."""
        xa, xb, xc, fb = bracket

        def cost(eps):
            nu = eps / sw_hz if base is None else base + eps / sw_hz
            return profile.cost(phi, *profile.at(nu))

        x = w = v = xb
        fx = fw = fv = fb
        a, b = Backend.minimum(xa, xc), Backend.maximum(xa, xc)
        deltax = xb * 0
        rat = xb * 0
        done = xb != xb  # all False
        for _ in range(iterations):
            tol1 = cls.BRENT_TOL * ops.abs(x) + cls.BRENT_MINTOL
            tol2 = 2.0 * tol1
            xmid = 0.5 * (a + b)
            done = done | (ops.abs(x - xmid) < (tol2 - 0.5 * (b - a)))
            far = ops.where(x >= xmid, a - x, b - x)
            tmp1 = (x - w) * (fx - fv)
            tmp2 = (x - v) * (fx - fw)
            p = (x - v) * tmp2 - (x - w) * tmp1
            q = 2.0 * (tmp2 - tmp1)
            p = ops.where(q > 0, -p, p)
            q = ops.abs(q)
            usable = ((p > q * (a - x)) & (p < q * (b - x))
                      & (ops.abs(p) < ops.abs(0.5 * q * deltax)))
            golden_first = ops.abs(deltax) <= tol1
            parabolic = ~golden_first & usable
            step = ops.where(parabolic, p / ops.where(q > 0, q, q * 0 + 1.0), cls.CGOLD * far)
            near_edge = ((x + step - a) < tol2) | ((b - x - step) < tol2)
            step = ops.where(parabolic & near_edge, ops.where(xmid - x >= 0, tol1, -tol1), step)
            deltax = ops.where(golden_first | ~usable, far, rat)
            rat = step
            u = x + ops.where(ops.abs(rat) < tol1, ops.where(rat >= 0, tol1, -tol1), rat)
            fu = cost(u)
            worse = fu > fx
            a_new = ops.where(worse, ops.where(u < x, u, a), ops.where(u >= x, x, a))
            b_new = ops.where(worse, ops.where(u < x, b, u), ops.where(u >= x, b, x))
            shift_w = worse & ((fu <= fw) | (w == x))
            shift_v = worse & ~shift_w & ((fu <= fv) | (v == x) | (v == w))
            v_new = ops.where(worse, ops.where(shift_w, w, ops.where(shift_v, u, v)), w)
            fv_new = ops.where(worse, ops.where(shift_w, fw, ops.where(shift_v, fu, fv)), fw)
            w_new = ops.where(worse, ops.where(shift_w, u, w), x)
            fw_new = ops.where(worse, ops.where(shift_w, fu, fw), fx)
            x_new, fx_new = ops.where(worse, x, u), ops.where(worse, fx, fu)
            a, b = ops.where(done, a, a_new), ops.where(done, b, b_new)
            v, fv = ops.where(done, v, v_new), ops.where(done, fv, fv_new)
            w, fw = ops.where(done, w, w_new), ops.where(done, fw, fw_new)
            x, fx = ops.where(done, x, x_new), ops.where(done, fx, fx_new)
        return x, a, b

    @staticmethod
    def newton(profile, nu, low, high, step_cap, steps):
        """Newton descent with the phase optimal at every shift (E + |Y|^2 - 2 |K|)."""
        for _ in range(steps):
            (k, k1, k2), (_, e1, e2) = profile.at(nu, derivatives=True)
            mag = ops.clip(ops.abs(k), 1e-300, None)
            slope = ops.real(ops.conj(k) * k1) / mag
            curve = ((ops.abs(k1) ** 2 + ops.real(ops.conj(k) * k2)) - slope ** 2) / mag
            grad, hess = e1 - 2 * slope, e2 - 2 * curve
            step = ops.where(hess > 0, -grad / ops.where(hess > 0, hess, hess * 0 + 1.0),
                             -Backend.sign(grad) * step_cap)
            nu = Backend.minimum(Backend.maximum(nu + ops.clip(step, -step_cap, step_cap), low),
                                 high)
        return nu

    @staticmethod
    def phasor(phi, eps, n, sw_hz):
        """e^{-i phi} e^{-2 pi i t eps} on FSL-MRS's time axis (dwell .. n dwell)."""
        t = Backend.const(phi, np.linspace(1.0 / sw_hz, n / sw_hz, n))
        return Backend.phasor(-phi[..., None] - 2 * math.pi * t * eps[..., None])


#**************************************************************************************************#
#                                              Unlike                                              #
#**************************************************************************************************#
#                                                                                                  #
# FSL-MRS identifyUnlikeFIDs: the distance of every transient's spectrum to the spectrum of the    #
# complex median (taken in the time domain, first point halved, by Parseval); transients further   #
# than 1.96 standard deviations from the mean distance are unlike; the median is recomputed once   #
# from the alike ones.                                                                             #
#                                                                                                  #
#**************************************************************************************************#
class Unlike:
    @staticmethod
    def median(parts, keep):
        """NumPy's median over the kept transients: parts (B, D, T) real, keep (B, D)."""
        d = ops.shape(parts)[1]
        count = ops.sum(ops.cast(keep, "int64"), axis=1)
        inf = parts * 0 + np.inf
        ordered = Backend.sort(ops.where(keep[..., None], parts, inf), 1)
        low = ops.reshape((count - 1) // 2, (-1, 1, 1)) * ops.cast(parts[:, :1] * 0 + 1, "int64")
        high = ops.reshape(count // 2, (-1, 1, 1)) * ops.cast(parts[:, :1] * 0 + 1, "int64")
        del d
        return 0.5 * (ops.take_along(ordered, low, axis=1) + ops.take_along(ordered, high, axis=1))

    @classmethod
    def keep(cls, fids, sdlimit=1.96, niter=2):
        """fids (B, D, T) -> keep mask (B, D)."""
        x = ops.cast(fids, "complex128")
        b, d, n = ops.shape(x)
        halved = Spectral.halve_first(x)
        energy = ops.sum(ops.real(halved * ops.conj(halved)), axis=-1)
        keep = ops.real(x[..., 0]) == ops.real(x[..., 0])  # all True
        re, im = ops.real(x), ops.imag(x)
        target = ops.complex_from(cls.median(re, keep), cls.median(im, keep))  # (B, 1, T)
        for step in range(niter):
            target_h = Spectral.halve_first(target)
            cross = ops.sum(ops.real(halved * ops.conj(target_h)), axis=-1)
            distance = energy - 2 * cross + ops.sum(ops.real(target_h * ops.conj(target_h)),
                                                    axis=-1)
            metric = ops.sqrt(ops.clip(distance, 0.0, None))
            avg = ops.mean(metric, axis=-1, keepdims=True)
            std = ops.sqrt(ops.mean((metric - avg) ** 2, axis=-1, keepdims=True))
            keep = ops.abs(metric - avg) <= sdlimit * std
            if step < niter - 1:
                target = ops.complex_from(cls.median(re, keep), cls.median(im, keep))
        return keep


#**************************************************************************************************#
#                                           Corrections                                            #
#**************************************************************************************************#
#                                                                                                  #
# After averaging: eddy current correction (Klose) with the smoothed unwrapped phase of the water  #
# (suspect's sliding Gaussian, 32 points; the data itself without water), the shift moving the     #
# tallest peak in 2.9-3.1 ppm to 3.027 ppm (creatine), and the zero-order phase making the tallest #
# peak in 2.9-3.1 ppm (water: 4.55-4.7 ppm) real, both searched on the four-fold zero-filled       #
# spectrum (FSL-MRS shiftToRef, phaseCorrect).                                                     #
#                                                                                                  #
#**************************************************************************************************#
class Corrections:
    @staticmethod
    def unwrap(phase):
        diff = phase[..., 1:] - phase[..., :-1]
        wrapped = Backend.remainder(diff + math.pi, 2 * math.pi) - math.pi
        wrapped = ops.where((wrapped == -math.pi) & (diff > 0), wrapped * 0 + math.pi, wrapped)
        correct = ops.where(ops.abs(diff) < math.pi, diff * 0, wrapped - diff)
        return ops.concatenate([phase[..., :1], phase[..., 1:] + Backend.cumsum(correct, -1)],
                               axis=-1)

    @classmethod
    def ecc_phase(cls, refs, width=32):
        phase = cls.unwrap(ops.angle(ops.cast(refs, "complex128")))
        n = ops.shape(phase)[-1]
        window = np.exp(-np.linspace(-3, 3, width) ** 2)
        window = window / window.sum()
        offset = (width - 1) // 2
        lead = ops.shape(phase)[:-1]
        left = ops.mean(phase[..., :10], axis=-1, keepdims=True) * Backend.const(
            phase, np.ones(lead + (offset,)))
        right = ops.mean(phase[..., -10:], axis=-1, keepdims=True) * Backend.const(
            phase, np.ones(lead + (width - 1 - offset,)))
        padded = ops.concatenate([left, phase, right], axis=-1)
        return sum(float(window[k]) * padded[..., k:k + n] for k in range(width))

    @staticmethod
    def peak(fids, sw_hz, sf_mhz, limits, reference):
        """The four-fold zero-filled spectrum inside *limits*, its bounds and its axis."""
        n = 4 * ops.shape(fids)[-1]
        first, last = Spectral.window(n, sw_hz, sf_mhz, limits, reference)
        spec = Spectral.fid_to_spec(Spectral.zero_fill(ops.cast(fids, "complex128"), 4))
        return spec[..., first:last], Spectral.ppm_axis(n, sw_hz, sf_mhz, reference)[first:last]

    @classmethod
    def phase(cls, fids, sw_hz, sf_mhz, limits, reference):
        spec, _ = cls.peak(fids, sw_hz, sf_mhz, limits, reference)
        index = ops.argmax(ops.abs(spec), axis=-1)[..., None]
        return -ops.angle(ops.take_along(spec, index, axis=-1))[..., 0]

    @classmethod
    def shift_hz(cls, fids, sw_hz, sf_mhz, limits, target_ppm, reference):
        spec, axis = cls.peak(fids, sw_hz, sf_mhz, limits, reference)
        index = ops.argmax(ops.abs(spec), axis=-1)
        return (ops.take(Backend.const(ops.real(spec), axis), index, 0) - target_ppm) * sf_mhz

    @staticmethod
    def shift_phasor(shift_hz, n, sw_hz):
        t = Backend.const(shift_hz, np.linspace(0, n / sw_hz, n))
        return Backend.phasor(-2 * math.pi * t * shift_hz[..., None])


#**************************************************************************************************#
#                                              Result                                              #
#**************************************************************************************************#
#                                                                                                  #
# What the pipeline gives: the processed FID (and water), and what it did on the way - coil        #
# weights, per-transient phase and frequency, the kept transients, the transients after alignment, #
# the eddy current phase, the shift and the zero-order phase - with the steps named.               #
#                                                                                                  #
#**************************************************************************************************#
@dataclass
class Result:
    fid: object
    water: object = None
    weights: object = None
    phi: object = None
    eps: object = None
    keep: object = None
    combined: object = None
    aligned: object = None
    shift_hz: object = None
    phase: object = None
    steps: list[str] = field(default_factory=list)


#**************************************************************************************************#
#                                           RawPipeline                                            #
#**************************************************************************************************#
#                                                                                                  #
# Augmentrum's RawProcessor defaults on one single-voxel dataset laid out (B, V, C, D, T): wSVD    #
# coil combination, alignment of the transients in 0.2-4.2 ppm (water 0-8 ppm), removal of unlike  #
# transients, mean, eddy current correction, shift to creatine, zero-order phase. Other nuclei     #
# than 1H skip the ppm windows of 1H: alignment over the whole spectrum, no shift, phase on the    #
# tallest peak. Without water, eddy currents are corrected with the data itself, as Augmentrum     #
# does (ecc_without_water=False leaves the step out, as FSL-MRS does). A water scan on its own     #
# takes the water path (run_water).                                                                #
#                                                                                                  #
#**************************************************************************************************#
class RawPipeline:
    def __init__(self, sw_hz: float, sf_mhz: float, proton: bool = True,
                 reference: float = PROTON_REFERENCE, ecc_without_water: bool = True):
        self.sw, self.sf, self.proton, self.reference = sw_hz, sf_mhz, proton, reference
        self.ecc_without_water = ecc_without_water

    def full(self) -> tuple[float, float]:
        axis = Spectral.ppm_axis(2, self.sw, self.sf, self.reference)
        return (float(axis.min()), float(axis.max()))

    def run(self, x, w=None) -> Result:
        """x: data (B, V, C, D, T); w: water (B, V, C or 1, Dw, T) or None; both in FSL-MRS's
        orientation, the conjugate of NIfTI-MRS as stored (what nifti_mrs hands out)."""
        steps = []
        b, v, c, d, n = ops.shape(x)
        if w is not None and ops.shape(w)[2] != c:
            steps.append("water has no coils to combine with: coils weighted from the data")
            w = None if c > 1 else w
        weights = None
        if c > 1:
            x, w, weights = CoilCombination.combine(x, w)
            steps.append(f"{c} coils combined (wSVD{', water weights' if w is not None else ''})")
        phi = eps = keep = aligned = None
        combined = x
        dropped = 0
        ppmlim = (0.2, 4.2) if self.proton else self.full()
        if d > 1:
            flat = ops.reshape(x[:, :, 0], (b * v, d, n))
            phi, eps = Registration.align(flat, self.sw, self.sf, ppmlim, self.reference)
            flat = flat * Registration.phasor(phi, eps, n, self.sw)
            aligned = ops.reshape(flat, (b, v, 1, d, n))
            x = aligned
            steps.append(f"{d} transients aligned (spectral registration "
                         f"{ppmlim[0]:g}-{ppmlim[1]:g} ppm)")
            if v == 1:
                keep = Unlike.keep(x[:, 0, 0])
                kept = ops.cast(keep, "float64")
                weights_d = kept / ops.sum(kept, axis=1, keepdims=True)
                x = ops.sum(x * ops.cast_like(weights_d, x)[:, None, None, :, None], axis=3,
                            keepdims=True)
                dropped = int(np.sum(~np.asarray(ops.to_numpy(keep))))
                steps.append(f"{dropped} unlike transients removed" if dropped
                             else "no unlike transients")
            else:
                x = ops.mean(x, axis=3, keepdims=True)
            steps.append(f"mean of {d - dropped} transients")
        if w is not None and ops.shape(w)[3] > 1:
            wd = ops.shape(w)[3]
            wflat = ops.reshape(w[:, :, 0], (b * v, wd, n))
            wphi, weps = Registration.align(wflat, self.sw, self.sf,
                                            (0, 8) if self.proton else self.full(),
                                            self.reference)
            w = ops.reshape(wflat * Registration.phasor(wphi, weps, n, self.sw), (b, v, 1, wd, n))
            w = ops.mean(w, axis=3, keepdims=True)
        if w is not None or self.ecc_without_water:
            ecc = Corrections.ecc_phase(w if w is not None else x)
            x = x * Backend.phasor(-ecc)
            w = w * Backend.phasor(-ecc) if w is not None else None
            steps.append("eddy currents corrected (" + ("water" if w is not None else
                                                        "data itself") + ", smoothed phase)")
        shift = None
        flat = ops.reshape(x, (-1, n))
        if self.proton:
            shift = Corrections.shift_hz(flat, self.sw, self.sf, (2.9, 3.1), 3.027,
                                         self.reference)
            flat = flat * Corrections.shift_phasor(shift, n, self.sw)
            steps.append("shifted to creatine at 3.027 ppm")
        limits = (2.9, 3.1) if self.proton else self.full()
        phase = Corrections.phase(flat, self.sw, self.sf, limits, self.reference)
        flat = flat * Backend.phasor(phase)[..., None]
        x = ops.reshape(flat, ops.shape(x))
        steps.append("zero-order phase on " + ("creatine" if self.proton else "the tallest peak"))
        if w is not None:
            wflat = ops.reshape(w, (-1, n))
            wlim = (4.55, 4.7) if self.proton else self.full()
            wphase = Corrections.phase(wflat, self.sw, self.sf, wlim, self.reference)
            w = ops.reshape(wflat * Backend.phasor(wphase)[..., None], ops.shape(w))
        return Result(x, w, weights, phi, eps, keep, combined, aligned, shift, phase, steps)

    def run_water(self, x) -> Result:
        """A water scan (B, V, C, D, T) on Augmentrum's water path: coils combined with their own
        weights, transients aligned in 0-8 ppm and averaged, eddy currents corrected with the
        water itself, zero-order phase on the water peak; no unlike removal, no creatine shift."""
        steps = []
        b, v, c, d, n = ops.shape(x)
        weights = phi = eps = aligned = None
        if c > 1:
            x, _, weights = CoilCombination.combine(x, x)
            steps.append(f"{c} coils combined (wSVD)")
        combined = x
        if d > 1:
            ppmlim = (0, 8) if self.proton else self.full()
            flat = ops.reshape(x[:, :, 0], (b * v, d, n))
            phi, eps = Registration.align(flat, self.sw, self.sf, ppmlim, self.reference)
            aligned = ops.reshape(flat * Registration.phasor(phi, eps, n, self.sw), (b, v, 1, d, n))
            x = ops.mean(aligned, axis=3, keepdims=True)
            steps.append(f"{d} transients aligned (spectral registration "
                         f"{ppmlim[0]:g}-{ppmlim[1]:g} ppm)")
            steps.append(f"mean of {d} transients")
        x = x * Backend.phasor(-Corrections.ecc_phase(x))
        steps.append("eddy currents corrected (water itself, smoothed phase)")
        flat = ops.reshape(x, (-1, n))
        limits = (4.55, 4.7) if self.proton else self.full()
        phase = Corrections.phase(flat, self.sw, self.sf, limits, self.reference)
        x = ops.reshape(flat * Backend.phasor(phase)[..., None], ops.shape(x))
        steps.append("zero-order phase on water")
        return Result(x, None, weights, phi, eps, None, combined, aligned, None, phase, steps)
