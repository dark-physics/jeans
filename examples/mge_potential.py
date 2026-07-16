import numpy as np
from scipy.optimize import nnls, lsq_linear
from mgefit.mge_fit_1d import mge_fit_1d
from readData import get_rc_data, get_mass_data
from scipy.interpolate import RegularGridInterpolator

# ---- physical constant and fixed Gauss-Legendre quadrature on [0,1] ----
G = 4.30091e-6                                   # kpc (km/s)^2 / Msun
_x, _w = np.polynomial.legendre.leggauss(100)
_T = 0.5*(_x + 1.0)
_W = 0.5*_w


def _nonneg_lstsq(A, b):
    """Non-negative least squares, robust to ill-conditioning (plain nnls can hang)."""
    try:
        x, _ = nnls(A, b, maxiter=50*A.shape[1])
        return x
    except (RuntimeError, TypeError):
        return lsq_linear(A, b, bounds=(0, np.inf), method="bvls", max_iter=200).x


class BaryonicModel:
    """MGE baryonic potential Phi(r, theta) for one SPARC galaxy."""

    def __init__(self, galaxy, qdisk=0.1, qgas=0.05, qbul=0.8, ngauss=8,
                 verbose=False):
        self.galaxy = galaxy
        self.qdisk, self.qgas, self.qbul = qdisk, qgas, qbul
        self.ngauss = ngauss
        self.verbose = verbose

        # data 
        self.data, _, _ = get_rc_data(galaxy)
        self.R      = self.data["Rad"].values # [kpc]
        self.SBdisk = self.data["SBdisk"].values # [Lsun/pc^2] at M/L=1
        self.SBbul  = self.data["SBbul"].values # [Lsun/pc^2] at M/L=1
        self.Vdisk  = self.data["Vdisk"].values  # [km/s]
        self.Vbul   = self.data["Vbul"].values  # [km/s]
        self.Vgas   = self.data["Vgas"].values  # [km/s]

        mass, _ = get_mass_data(galaxy)
        self.dist, self.dist_err =  mass["D"].values[0], mass["e_D"].values[0] # [Mpc]
        self.incl, self.incl_err = mass["Inc"].values[0], mass["e_Inc"].values[0] # [deg]

        # build all components once, at Upsilon = 1
        # each entry: dict(tag, sigma_k, Sigma0_k, q, phi, vcirc, route)
        self.components = []
        self._add_disk()
        self._add_bulge()     
        self._add_gas()

    # MGE potential math (single component, Upsilon = 1)
    @staticmethod
    def _masses(sigma_k, Sigma0_k):
        """Msun mass of each Gaussian (q-independent).  Sigma0 [Msun/pc^2], sigma [kpc]."""
        sigma_k = np.asarray(sigma_k, float)
        Sigma0_k = np.asarray(Sigma0_k, float)
        return 2*np.pi*sigma_k**2*Sigma0_k*1e6           # 1e6: pc^2 -> kpc^2

    def _make_component_potential(self, sigma_k, Sigma0_k, q):
        """Return (phi(R,z), vcirc(R)) callables for one oblate MGE at Upsilon=1."""
        sigma = np.asarray(sigma_k, float)
        M = self._masses(sigma, Sigma0_k)
        e2 = 1.0 - float(q)**2

        def phi(R, z):
            R = np.atleast_1d(np.asarray(R, float))
            z = np.atleast_1d(np.asarray(z, float))
            R, z = np.broadcast_arrays(R, z)
            out = np.zeros(R.shape)
            for k in range(sigma.size):
                denom = 1.0 - e2*_T**2
                br = R[..., None]**2 + z[..., None]**2/denom
                integ = np.exp(-_T**2*br/(2*sigma[k]**2))/np.sqrt(denom)
                out += (M[k]/sigma[k])*(integ*_W).sum(-1)
            return -np.sqrt(2/np.pi)*G*out

        def vcirc(R):
            R = np.atleast_1d(np.asarray(R, float))
            v2 = np.zeros(R.shape)
            for k in range(sigma.size):
                denom = 1.0 - e2*_T**2
                integ = _T**2*np.exp(-_T**2*R[..., None]**2/(2*sigma[k]**2))/np.sqrt(denom)
                v2 += (M[k]/sigma[k]**3)*R**2*(integ*_W).sum(-1)
            return np.sqrt(np.clip(np.sqrt(2/np.pi)*G*v2, 0, None))

        return phi, vcirc

    def _register(self, tag, sigma_k, Sigma0_k, q, route=""):
        phi, vcirc = self._make_component_potential(sigma_k, Sigma0_k, q)
        self.components.append(dict(tag=tag, sigma_k=sigma_k, Sigma0_k=Sigma0_k,
                                    q=q, phi=phi, vcirc=vcirc, route=route))

    # INTERNAL FITTING ROUTINES
    def _fit_sb_mge(self, SB, ngauss):
        """1D MGE of a surface-brightness profile -> (sigma_k [kpc], Sigma0_k [Lsun/pc^2])."""
        m = np.isfinite(SB) & (SB > 0) & np.isfinite(self.R) & (self.R > 0)
        R, S = self.R[m], SB[m]
        ng = ngauss if R.size >= ngauss + 2 else max(3, R.size - 2)
        p = mge_fit_1d(R, S, ngauss=ng, quiet=True)
        sigma = p.sol[1]
        Sigma0 = p.sol[0]/(np.sqrt(2*np.pi)*sigma)       # peak (sol[0] is the integral)
        if self.verbose:
            model = (Sigma0[None, :]*np.exp(-0.5*(R[:, None]/sigma[None, :])**2)).sum(1)
            rel = np.abs(model - S)/S
            print(f"[{self.galaxy}] SB-MGE: {sigma.size} Gaussians  "
                  f"median rel err {np.median(rel):.2e}")
        return sigma, Sigma0

    def _invert_vcirc(self, Vc, q, sig_min, sig_max, ngauss=12, signed=False):
        """Recover a thin/round MGE whose forward V_c matches Vc (NNLS forward model)."""
        R = self.R
        m = np.isfinite(R) & np.isfinite(Vc) & (R > 0)
        Rf, Vf = R[m], Vc[m]
        target = Vf*np.abs(Vf) if signed else Vf**2
        sig = np.geomspace(sig_min, sig_max, ngauss)
        K = np.empty((Rf.size, ngauss))
        for k in range(ngauss):
            _, vck = self._make_component_potential([sig[k]], [1.0], q)
            K[:, k] = vck(Rf)**2
        w = 1.0/(2*(np.abs(Vf) + 10.0))
        Sigma0 = _nonneg_lstsq(K*w[:, None], target*w)
        keep = Sigma0 > 0
        return sig[keep], Sigma0[keep]

    def _add_disk(self):
        sigma, Sigma0 = self._fit_sb_mge(self.SBdisk, self.ngauss)
        self._register("d", sigma, Sigma0, self.qdisk, route="SBdisk-MGE")

    def _add_gas(self):
        if not np.any(np.abs(self.Vgas) > 1e-3):
            return                                        # no gas contribution
        sigma, Sigma0 = self._invert_vcirc(self.Vgas, self.qgas,
                                           sig_min=1.0, sig_max=self.R.max(),
                                           ngauss=14, signed=True)
        self._register("g", sigma, Sigma0, self.qgas, route="Vgas-inversion")

    def _add_bulge(self, pointmass_tol=0.03, min_sb_points=5):
        Vbul, SBbul = self.Vbul, self.SBbul
        if not np.any(Vbul > 0):
            return # bulgeless galaxy
        good = Vbul > 0
        v2r = (Vbul[good]**2)*self.R[good]
        pointlike = v2r.std()/v2r.mean() < pointmass_tol
        sbpos = SBbul > 0
        n_sb = int(sbpos.sum())
        core = SBbul[sbpos]
        has_turnover = n_sb >= 3 and np.any(np.diff(core[:min(4, n_sb)]) > 0)

        if (not pointlike) and n_sb >= min_sb_points and has_turnover:
            sigma, Sigma0 = self._fit_sb_mge(SBbul, ngauss=6)
            route = "SBbul-MGE (resolved core)"
        else:
            sigma, Sigma0 = self._invert_vcirc(Vbul, self.qbul,
                                               sig_min=0.3, sig_max=self.R.max()/3,
                                               ngauss=12, signed=False)
            why = "point mass" if pointlike else ("few SB pts" if n_sb < min_sb_points
                                                  else "no core turnover")
            route = f"Vbul-inversion ({why})"
        self._register("b", sigma, Sigma0, self.qbul, route=route)

    # CALLABLES
    def _scale(self, Upsilond, Upsilonb):
        return {"d": Upsilond, "b": Upsilonb, "g": 1.0}
 
    def potential(self, r, th, Upsilond=1.0, Upsilonb=1.0, D=None):
        """
        Total baryonic potential Phi(r, theta) in (km/s)^2.  Gas is never scaled.
        D : if given, evaluate at true distance D [Mpc] via the exact scaling
            Phi(r; D) = (D/D0) * Phi_fid(r * D0/D),  D0 = catalog distance.
        """
        a = 1.0 if D is None else float(D)/self.dist
        scalar = np.isscalar(r) and np.isscalar(th)
        r = np.asarray(r, float); th = np.asarray(th, float)
        R, z = (r/a)*np.sin(th), (r/a)*np.cos(th)
        s = self._scale(Upsilond, Upsilonb)
        out = a*sum(s[c["tag"]]*c["phi"](R, z) for c in self.components)
        return float(np.squeeze(out)) if scalar else out
 
    def potential_function(self, Upsilond=1.0, Upsilonb=1.0, D=None):
        """Return a callable phi(r, th) with M/Ls (and optional distance) baked in."""
        def phi(r, th):
            return self.potential(r, th, Upsilond, Upsilonb, D)
        return phi
    
    def tabulated_potential_function(self, Upsilond=1.0, Upsilonb=1.0, D=None,
                                 rmin=0.05, rmax=None, nr=240, nth=64, method="cubic"):
        """
        Like potential_function, but returns a fast interpolant: the exact MGE
        potential is evaluated once on a (log-r, theta) grid, then interpolated.
        Rebuild per likelihood call (Upsilon/D change the potential). ~200-300x
        faster per evaluation at ~1e-4 accuracy.
        """
        if rmax is None:
            rmax = 3.0*self.R.max()                      # cover halo integration range
        phi_exact = self.potential_function(Upsilond, Upsilonb, D)
        r  = np.geomspace(rmin, rmax, nr)
        th = np.linspace(0.0, np.pi, nth)
        grid = phi_exact(*np.meshgrid(r, th, indexing="ij"))
        interp = RegularGridInterpolator((r, th), grid, method=method,
                                        bounds_error=False, fill_value=None)
        def phi(r, th):
            scalar = np.isscalar(r) and np.isscalar(th)
            r = np.atleast_1d(np.asarray(r, float))
            th = np.atleast_1d(np.asarray(th, float))
            r, th = np.broadcast_arrays(r, th)
            out = interp(np.stack([r, th], axis=-1))
            return float(out.reshape(())) if scalar else out
        return phi
 
    def vcirc(self, R=None, Upsilond=1.0, Upsilonb=1.0, which="total", D=None):
        """
        In-plane baryonic circular velocity [km/s].
        which : 'total' | 'disk' | 'bulge' | 'gas'
        D     : optional true distance [Mpc];  V_c(R;D)=sqrt(D/D0)*V_c_fid(R*D0/D).
        """
        if R is None:
            R = self.R
        a = 1.0 if D is None else float(D)/self.dist
        R = np.asarray(R, float); Rq = R/a
        s = self._scale(Upsilond, Upsilonb)
        tags = {"disk": "d", "bulge": "b", "gas": "g"}
        if which == "total":
            v2 = sum(s[c["tag"]]*c["vcirc"](Rq)**2 for c in self.components)
        else:
            comp = [c for c in self.components if c["tag"] == tags[which]]
            v2 = s[tags[which]]*comp[0]["vcirc"](Rq)**2 if comp else np.zeros_like(np.atleast_1d(R), float)
        return np.sqrt(np.clip(a*v2, 0, None))
 
    def model_curves(self, Upsilond=1.0, Upsilonb=1.0, R=None, D=None):
        """Dict of model V_c curves for comparison with the SPARC data columns."""
        if R is None:
            R = self.R
        return dict(R=R,
                    Vdisk=self.vcirc(R, Upsilond, Upsilonb, "disk", D),
                    Vbul=self.vcirc(R, Upsilond, Upsilonb, "bulge", D),
                    Vgas=self.vcirc(R, Upsilond, Upsilonb, "gas", D),
                    Vbar=self.vcirc(R, Upsilond, Upsilonb, "total", D))
 
    def vobs_at_inclination(self, incl):
        """
        Inclination-corrected Vobs for the likelihood (inclination does NOT enter Phi).
        Vobs scales as sin(i0)/sin(i); returns (R_phys, Vobs_corr, errV_corr) at the
        sampled inclination.  R_phys is unchanged by inclination.
        """
        f = np.sin(np.radians(self.incl))/np.sin(np.radians(incl))
        return (self.R,
                self.data["Vobs"].values*f,
                self.data["errV"].values*f)
 
    def data_radii(self, D=None):
        """Physical data radii at true distance D:  R_phys = R_fid * (D/D0)."""
        a = 1.0 if D is None else float(D)/self.dist
        return self.R*a
 
    def component_masses(self, Upsilond=1.0, Upsilonb=1.0):
        """Total mass [Msun] of each component at the given M/Ls."""
        s = self._scale(Upsilond, Upsilonb)
        names = {"d": "disk", "b": "bulge", "g": "gas"}
        return {names[c["tag"]]: s[c["tag"]]*self._masses(c["sigma_k"], c["Sigma0_k"]).sum()
                for c in self.components}
 
    def __repr__(self):
        parts = ", ".join(f"{c['tag']}:{c['route']}" for c in self.components)
        return (f"BaryonicModel({self.galaxy}, qdisk={self.qdisk}, "
                f"qbul={self.qbul}, qgas={self.qgas} | {parts})")
        
    def __getstate__(self):
        state = self.__dict__.copy()
        state["components"] = [{k: c[k] for k in ("tag","sigma_k","Sigma0_k","q","route")}
                            for c in self.components]      # drop unpicklable closures
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)
        rebuilt = []
        for c in self.components:
            phi, vcirc = self._make_component_potential(c["sigma_k"], c["Sigma0_k"], c["q"])
            c = dict(c); c["phi"], c["vcirc"] = phi, vcirc
            rebuilt.append(c)
        self.components = rebuilt
        
class TabulatedBaryons:
    """
    Fast, picklable wrapper around a BaryonicModel.

    The expensive MGE quadrature is evaluated ONCE per component (at Upsilon=1,
    fiducial distance) onto a (log-r, theta) grid and stored as interpolants.
    Per MCMC step, potential_function() only does a linear Upsilon-combination
    plus the analytic distance rescale -- no grid rebuild.

    Build once in main(), pass to the sampler, call potential_function() in model().
    """

    def __init__(self, bar, rmin=0.05, rmax=None, nr=240, nth=64, method="cubic"):
        if rmax is None:
            rmax = 3.0*bar.R.max()          # cover Jeans/halo range + distance rescale
        self.D0 = bar.dist
        r  = np.geomspace(rmin, rmax, nr)
        th = np.linspace(0.0, np.pi, nth)
        RR, TT = np.meshgrid(r, th, indexing="ij")
        R, z = RR*np.sin(TT), RR*np.cos(TT)
        self.interp = {}                    # tag -> interpolant at Upsilon=1, fiducial D
        for c in bar.components:
            grid = c["phi"](R, z)           # expensive MGE quadrature, once per component
            self.interp[c["tag"]] = RegularGridInterpolator(
                (r, th), grid, method=method, bounds_error=False, fill_value=None)

    def potential_function(self, Upsilond=1.0, Upsilonb=1.0, D=None):
        """Return phi(r, th) with M/Ls and distance applied -- drop-in for the Jeans model."""
        a = 1.0 if D is None else float(D)/self.D0
        scale = {"d": Upsilond, "b": Upsilonb, "g": 1.0}
        itp = self.interp

        def phi(r, th):
            scalar = np.isscalar(r) and np.isscalar(th)
            r  = np.atleast_1d(np.asarray(r, float))
            th = np.atleast_1d(np.asarray(th, float))
            r, th = np.broadcast_arrays(r, th)
            pts = np.stack([r/a, th], axis=-1)          # r/a is the distance rescale
            out = a*sum(scale[t]*itp[t](pts) for t in itp)
            return float(out.reshape(())) if scalar else out

        return phi