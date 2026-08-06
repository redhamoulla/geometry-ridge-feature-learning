import numpy as np
from geometric_register.core import orbit_signature, register_diagnostics

def test_orbit_signature_rotation_invariant():
    B=np.diag([.2,1.,2.]); b=np.array([1.,2.,3.]); q,_=np.linalg.qr(np.random.default_rng(0).normal(size=(3,3)))
    a=orbit_signature(B,b); c=orbit_signature(q@B@q.T,q@b)
    assert np.allclose([(x.eigenvalue,x.effort_energy) for x in a],[(x.eigenvalue,x.effort_energy) for x in c])

def test_scalarization_is_noninjective():
    d1=register_diagnostics(np.diag([1.,0.]),np.array([1.,0.])); d2=register_diagnostics(np.eye(2)/3,np.array([1.,0.]))
    assert np.isclose(d1['trace_contraction'],d2['trace_contraction'])
