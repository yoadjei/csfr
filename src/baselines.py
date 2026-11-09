"""Classical reconstruction baselines (no external prior).

The first four (zero-fill, bilinear, Telea, Navier-Stokes, dictionary learning)
are moved verbatim from scripts/run_csfr_sweep.py, which re-exports them, so
the released numbers and the round-2 harness use one implementation. The
round-2 additions answer the request for a stronger classical set: Guleryuz-type
iterative DCT thresholding, biharmonic, frequency-selective reconstruction and
shift-map (OpenCV contrib), and exemplar-based inpainting after Criminisi.

Every function takes (y, M) as (B, H, W) float arrays with M = 1 on observed
pixels, returns (B, H, W) float32 in [0, 255], and leaves observed pixels exact.
BASELINES maps the harness name to a callable f(y, M, seed=0).
"""
from __future__ import annotations

import numpy as np


def baseline_zero(y: np.ndarray, M: np.ndarray) -> np.ndarray:
    out = y.copy()
    out[M == 0] = 0.0
    return out


def baseline_bilinear(y: np.ndarray, M: np.ndarray) -> np.ndarray:
    from scipy.interpolate import griddata
    b, h, w = y.shape
    out = np.zeros_like(y)
    yy, xx = np.mgrid[0:h, 0:w]
    for i in range(b):
        Mi = M[i] == 1
        if not Mi.any():
            continue
        pts = np.stack([yy[Mi], xx[Mi]], axis=1)
        vals = y[i][Mi]
        grid = griddata(pts, vals, (yy, xx), method="linear", fill_value=0.0)
        out[i] = grid
    return out


def baseline_inpaint_cv(y: np.ndarray, M: np.ndarray, flag: int) -> np.ndarray:
    """Classical PDE inpainting (Telea / Navier-Stokes) via OpenCV."""
    import cv2
    out = np.empty_like(y)
    for i in range(y.shape[0]):
        img8 = np.clip(y[i], 0, 255).astype(np.uint8)
        hole = (M[i] == 0).astype(np.uint8)
        out[i] = cv2.inpaint(img8, hole, inpaintRadius=3, flags=flag).astype(np.float32)
    return out


def baseline_dictlearn(y: np.ndarray, M: np.ndarray, rng_seed: int,
                       tile: int = 8, stride: int = 8,
                       n_atoms: int = 64, n_nonzero: int = 8) -> np.ndarray:
    """Per-image dictionary-learning inpainting (online dictionary learning,
    Mairal et al. 2009 style): learn atoms from fully-observed tiles of the
    SAME corrupted image, then reconstruct each incomplete tile by
    orthogonal matching pursuit on the observed pixels only. Tiles with no
    observations fall back to the observed-region mean."""
    from sklearn.decomposition import MiniBatchDictionaryLearning

    def omp_numpy(D_obs: np.ndarray, target: np.ndarray, k: int) -> np.ndarray:
        """Bounded orthogonal matching pursuit (numpy only — sklearn's OMP can
        deadlock under mixed torch/OpenMP threadpools on Windows)."""
        code = np.zeros(D_obs.shape[1])
        resid = target.astype(np.float64).copy()
        support: list[int] = []
        norms = np.linalg.norm(D_obs, axis=0)
        norms[norms < 1e-10] = 1e-10
        for _ in range(k):
            corr = np.abs(D_obs.T @ resid) / norms
            corr[support] = -1.0
            j = int(np.argmax(corr))
            if corr[j] <= 1e-12:
                break
            support.append(j)
            sub = D_obs[:, support]
            sol, *_ = np.linalg.lstsq(sub, target, rcond=None)
            resid = target - sub @ sol
        if support:
            code[support] = sol
        return code

    b, h, w = y.shape
    out = y.copy()
    for i in range(b):
        img, m = y[i], M[i]
        obs_tiles = []
        for r in range(0, h - tile + 1, 2):
            for c in range(0, w - tile + 1, 2):
                if m[r:r + tile, c:c + tile].all():
                    obs_tiles.append(img[r:r + tile, c:c + tile].reshape(-1))
        obs_mean = float(img[m == 1].mean()) if (m == 1).any() else 127.5
        if len(obs_tiles) < n_atoms:
            out[i][m == 0] = obs_mean  # too little intact area to learn from
            continue
        X = np.asarray(obs_tiles, dtype=np.float64)
        if len(X) > 400:  # cap fit cost; deterministic subsample
            X = X[np.random.default_rng(rng_seed).choice(len(X), 400, replace=False)]
        mu = X.mean(axis=0)
        dl = MiniBatchDictionaryLearning(
            n_components=n_atoms, alpha=1.0, batch_size=32, max_iter=30,
            random_state=rng_seed, fit_algorithm="cd",
            transform_algorithm="lasso_cd", transform_max_iter=200)
        dl.fit(X - mu)
        Dt = dl.components_.T  # (tile*tile, n_atoms)

        acc = np.zeros((h, w), dtype=np.float64)
        cnt = np.zeros((h, w), dtype=np.float64)
        for r in range(0, h - tile + 1, stride):
            for c in range(0, w - tile + 1, stride):
                tm = m[r:r + tile, c:c + tile].reshape(-1)
                ty = img[r:r + tile, c:c + tile].reshape(-1).astype(np.float64)
                if tm.all():
                    rec = ty
                elif not tm.any():
                    rec = np.full(tile * tile, obs_mean)
                else:
                    obs = tm == 1
                    D_obs = Dt[obs]
                    code = omp_numpy(D_obs, (ty - mu)[obs],
                                     min(n_nonzero, int(obs.sum())))
                    rec = Dt @ code + mu
                    rec[obs] = ty[obs]  # keep observed pixels exact
                acc[r:r + tile, c:c + tile] += rec.reshape(tile, tile)
                cnt[r:r + tile, c:c + tile] += 1.0
        filled = np.divide(acc, np.maximum(cnt, 1), out=np.full((h, w), obs_mean), where=cnt > 0)
        res = img.astype(np.float64).copy()
        res[m == 0] = filled[m == 0]
        out[i] = np.clip(res, 0, 255).astype(np.float32)
    return out




# ── round-2 classical baselines ─────────────────────────────────────────
def _block_dct_threshold(x: np.ndarray, tau: float, b: int = 8,
                         shifts: tuple[int, ...] = (0, 2, 4, 6)) -> np.ndarray:
    """Average of hard-thresholded reconstructions over shifted b x b block-DCT
    grids (symmetric padding, DC always kept). x is (B, H, W)."""
    from scipy.fft import dctn, idctn
    B, h, w = x.shape
    pad = np.pad(x, ((0, 0), (b, b), (b, b)), mode="symmetric")
    acc = np.zeros_like(x)
    n = 0
    for sy in shifts:
        for sx in shifts:
            H = ((h + sy + b - 1) // b) * b
            W = ((w + sx + b - 1) // b) * b
            sub = pad[:, b - sy: b - sy + H, b - sx: b - sx + W]
            blk = sub.reshape(B, H // b, b, W // b, b).transpose(0, 1, 3, 2, 4)
            a = dctn(blk, type=2, norm="ortho", axes=(-2, -1))
            keep = np.abs(a) >= tau
            keep[..., 0, 0] = True
            rec = idctn(a * keep, type=2, norm="ortho", axes=(-2, -1))
            rec = rec.transpose(0, 1, 3, 2, 4).reshape(B, H, W)
            acc += rec[:, sy: sy + h, sx: sx + w]
            n += 1
    return acc / n


def baseline_guleryuz(y: np.ndarray, M: np.ndarray, seed: int = 0, n_iter: int = 100,
                      tau0: float = 128.0, tau1: float = 2.0) -> np.ndarray:
    """Iterated denoising by overlapping block-DCT hard thresholding, after Guleryuz
    (2006): 8 x 8 DCT blocks on 16 grid shifts (stride 2), coefficients below the
    threshold zeroed (DC kept), the 16 reconstructions averaged, observed pixels
    restored and the box applied; the threshold decreases geometrically from tau0
    to tau1 over n_iter iterations. The hole starts at the observed mean.
    Deterministic; `seed` is accepted for the common interface and unused."""
    m = M == 1
    x = y.astype(np.float64).copy()
    cnt = np.maximum(m.reshape(m.shape[0], -1).sum(1), 1)
    mean = (x * m).reshape(x.shape[0], -1).sum(1) / cnt
    x = np.where(m, x, mean[:, None, None])
    taus = tau0 * (tau1 / tau0) ** (np.arange(n_iter) / max(n_iter - 1, 1))
    for tau in taus:
        x = _block_dct_threshold(x, float(tau))
        x = np.where(m, y, x)
        np.clip(x, 0, 255, out=x)
    return x.astype(np.float32)


def baseline_biharmonic(y: np.ndarray, M: np.ndarray, seed: int = 0) -> np.ndarray:
    """Biharmonic inpainting (scikit-image), observed pixels restored exactly."""
    from skimage.restoration import inpaint_biharmonic
    out = y.copy()
    for i in range(y.shape[0]):
        hole = M[i] == 0
        if not hole.any():
            continue
        if hole.all():
            out[i][:] = 127.5
            continue
        rec = inpaint_biharmonic(y[i].astype(np.float64), hole)
        out[i] = np.where(hole, np.clip(rec, 0, 255), y[i]).astype(np.float32)
    return out


def xphoto_available() -> bool:
    try:
        import cv2
        return hasattr(cv2, "xphoto") and hasattr(cv2.xphoto, "inpaint")
    except ImportError:
        return False


def _xphoto(y: np.ndarray, M: np.ndarray, flag_name: str) -> np.ndarray:
    """OpenCV-contrib xphoto inpainting. xphoto takes a mask of VALID pixels
    (non-zero = known), the opposite convention to cv2.inpaint."""
    import cv2
    flag = getattr(cv2.xphoto, flag_name)
    out = np.empty_like(y)
    for i in range(y.shape[0]):
        img8 = np.clip(y[i], 0, 255).astype(np.uint8)
        valid = (M[i] == 1).astype(np.uint8) * 255
        dst = np.zeros_like(img8)
        cv2.xphoto.inpaint(img8, valid, dst, flag)
        rec = dst.astype(np.float32)
        out[i] = np.where(M[i] == 1, y[i], rec)
    return out


def baseline_fsr(y: np.ndarray, M: np.ndarray, seed: int = 0) -> np.ndarray:
    """Frequency-selective reconstruction (Seiler, Jonscher, Kaup), OpenCV-contrib, fast mode."""
    return _xphoto(y, M, "INPAINT_FSR_FAST")


def baseline_shiftmap(y: np.ndarray, M: np.ndarray, seed: int = 0) -> np.ndarray:
    """Shift-map inpainting (Pritch, Kav-Venaki, Peleg), OpenCV-contrib: a non-local
    method that copies pixels from elsewhere in the image by graph-cut optimisation."""
    return _xphoto(y, M, "INPAINT_SHIFTMAP")


def baseline_criminisi(y: np.ndarray, M: np.ndarray, seed: int = 0, patch: int = 9,
                       return_info: bool = False):
    """Exemplar-based inpainting after Criminisi, Perez and Toyama (2004).

    Vectorised variant: the confidence term is a box mean of the confidence map
    over the patch window, the data term uses the isophote of a normalised
    convolution of the known pixels at the fill front, and exemplars are windows
    lying entirely inside the originally observed region, matched by SSD on the
    known pixels of the target window. If no fully observed window exists at the
    requested size the size is reduced (9, 7, 5, 3); if none exists at size 3
    (scattered loss at high severity), the remaining hole is filled with the
    observed mean and the patch is counted as a fallback. Deterministic: ties
    resolve to the first index in raster order."""
    out = y.copy()
    flags = []
    for i in range(y.shape[0]):
        rec, fell_back = _criminisi_single(y[i], M[i] == 1, patch)
        out[i] = rec
        flags.append(bool(fell_back))
    if return_info:
        return out, {"fallback_patches": int(sum(flags)), "fallback": flags}
    return out


def _criminisi_single(img0: np.ndarray, known0: np.ndarray, p0: int) -> tuple[np.ndarray, bool]:
    from numpy.lib.stride_tricks import sliding_window_view
    from scipy.ndimage import binary_dilation, sobel, uniform_filter

    if known0.all():
        return img0.copy(), False
    if not known0.any():
        return np.full_like(img0, 127.5), True
    h, w = img0.shape
    src_img = img0.astype(np.float64)
    p, cand = p0, None
    for p in [s for s in (p0, 7, 5, 3) if s <= p0 and s <= min(h, w)]:
        win_known = sliding_window_view(known0, (p, p)).all(axis=(2, 3))
        if win_known.any():
            cand = np.argwhere(win_known)
            break
    if cand is None:
        res = src_img.copy()
        res[~known0] = src_img[known0].mean()
        return res.astype(np.float32), True
    src = sliding_window_view(src_img, (p, p))[cand[:, 0], cand[:, 1]]  # (C, p, p)
    r = p // 2
    img = src_img.copy()
    known = known0.copy()
    conf = known.astype(np.float64)
    cross = np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]], bool)
    while not known.all():
        front = ~known & binary_dilation(known, cross)
        if not front.any():
            break
        kf = known.astype(np.float64)
        C = uniform_filter(conf * kf, size=p, mode="constant")
        wsum = uniform_filter(kf, size=3, mode="constant")
        filled = np.where(known, img, uniform_filter(img * kf, size=3, mode="constant")
                          / np.maximum(wsum, 1e-12))
        gy, gx = sobel(filled, axis=0), sobel(filled, axis=1)
        ny, nx = sobel(kf, axis=0), sobel(kf, axis=1)
        nn = np.hypot(ny, nx)
        D = np.abs(-gy * ny + gx * nx) / (255.0 * np.maximum(nn, 1e-12))
        pr = np.where(front, C * (D + 1e-3), -1.0)
        fy, fx = np.unravel_index(int(np.argmax(pr)), pr.shape)
        y0 = min(max(fy - r, 0), h - p)
        x0 = min(max(fx - r, 0), w - p)
        T = img[y0:y0 + p, x0:x0 + p]
        K = known[y0:y0 + p, x0:x0 + p]
        ssd = (((src - T) ** 2) * K).sum(axis=(1, 2))
        S = src[int(np.argmin(ssd))]
        newc = conf[y0:y0 + p, x0:x0 + p][K].sum() / (p * p)
        img[y0:y0 + p, x0:x0 + p][~K] = S[~K]
        conf[y0:y0 + p, x0:x0 + p][~K] = newc
        known[y0:y0 + p, x0:x0 + p] = True
    res = np.where(known0, src_img, np.clip(img, 0, 255))
    return res.astype(np.float32), False


def _cv_flag(name: str) -> int:
    import cv2
    return getattr(cv2, name)


def _keep_observed(fn):
    """Composite the observed pixels back. cv2.inpaint works on 8-bit images and
    rounds non-integer observed values; the released sweeps used 8-bit patches so
    their numbers are unaffected, but the round-2 contract is exact preservation
    for any input, so every registered method goes through this wrapper."""
    def wrapped(y, M, seed=0, **kw):
        rec = fn(y, M, seed=seed, **kw)
        if isinstance(rec, tuple):
            rec, info = rec
            return np.where(M == 1, y, rec).astype(np.float32), info
        return np.where(M == 1, y, rec).astype(np.float32)
    wrapped.__name__ = getattr(fn, "__name__", "baseline")
    wrapped.__doc__ = fn.__doc__
    return wrapped


BASELINES = {name: _keep_observed(fn) for name, fn in {
    "zero_fill": lambda y, M, seed=0: baseline_zero(y, M),
    "bilinear": lambda y, M, seed=0: baseline_bilinear(y, M),
    "inpaint_telea": lambda y, M, seed=0: baseline_inpaint_cv(y, M, _cv_flag("INPAINT_TELEA")),
    "inpaint_ns": lambda y, M, seed=0: baseline_inpaint_cv(y, M, _cv_flag("INPAINT_NS")),
    "dictlearn": lambda y, M, seed=0: baseline_dictlearn(y, M, seed),
    "biharmonic": baseline_biharmonic,
    "guleryuz": baseline_guleryuz,
    "fsr": baseline_fsr,
    "shiftmap": baseline_shiftmap,
    "criminisi": baseline_criminisi,
}.items()}

# methods whose registry call can also return per-patch diagnostics
_WITH_INFO = {"criminisi"}


def run_chunk(name: str, y: np.ndarray, M: np.ndarray, seed: int) -> tuple[np.ndarray, dict]:
    """One registered baseline on a chunk of patches; returns (output, info).

    The process-pool entry point of the sweep harness. It lives here, not in the
    script, so that it pickles by reference under the spawn start method. Every
    registered method treats patches independently, so the output does not
    depend on how the batch is split (tests/test_sweep_v2.py checks that). The
    output can differ in the last float32 bit between processes whose BLAS was
    initialised with different thread counts, which is why the harness runs every
    baseline in workers initialised by limit_threads.
    """
    if name in _WITH_INFO:
        return BASELINES[name](y, M, seed=seed, return_info=True)
    return BASELINES[name](y, M, seed=seed), {}


_THREAD_LIMITS = None


def limit_threads() -> None:
    """Process-pool initialiser: one BLAS, OpenMP and OpenCV thread per worker,
    so that the workers do not oversubscribe the cores."""
    global _THREAD_LIMITS
    import cv2
    from threadpoolctl import threadpool_limits
    _THREAD_LIMITS = threadpool_limits(1)  # kept alive for the life of the worker
    cv2.setNumThreads(1)
