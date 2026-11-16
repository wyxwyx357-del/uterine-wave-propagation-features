"""Anatomically ordered seeds in processed-image xy coordinates."""
import json
from pathlib import Path
import numpy as np

CONFIG = dict(window_px=21, spacing_px=10.5, radial_baseline_px=21.,
              endpoint_margin_px=21., band_fractions=[.25, .5, .75],
              coordinate_scale=.5)


def read_labelme(path, *, coordinate_scale=.5):
    """Read only first-frame geometry; imageData and other labels are ignored."""
    if not np.isfinite(coordinate_scale) or coordinate_scale <= 0:
        raise ValueError("coordinate_scale must be positive")
    document = json.loads(Path(path).read_text(encoding="utf-8-sig"))
    names = ("ant_wall", "post_wall", "cervix_point", "fundus_point")
    shapes = {}
    for name in names:
        entries = [s for s in document["shapes"] if s["label"] == name]
        if len(entries) != 1:
            raise ValueError(f"Expected one {name} annotation")
        shapes[name] = np.asarray(entries[0]["points"], float) * coordinate_scale
    validate_shapes(shapes)
    return shapes


def validate_shapes(shapes):
    for name in ("ant_wall", "post_wall", "cervix_point", "fundus_point"):
        p = np.asarray(shapes[name], float)
        required = 2 if name.endswith("wall") else 1
        if p.ndim != 2 or p.shape[1] != 2 or len(p) < required or not np.isfinite(p).all():
            raise ValueError(f"Invalid {name} geometry")
        if not name.endswith("wall") and len(p) != 1:
            raise ValueError(f"{name} must contain one point")
    if np.linalg.norm(shapes["cervix_point"][0] - shapes["fundus_point"][0]) < 1e-8:
        raise ValueError("Cervix and fundus must be distinct")


def orient(rail,cervix,fundus):
 rail=np.asarray(rail,float)
 if rail.ndim!=2 or rail.shape[1]!=2 or len(rail)<2:raise ValueError('Expected 2D wall polyline')
 if np.linalg.norm(rail[0]-cervix)+np.linalg.norm(rail[-1]-fundus)>np.linalg.norm(rail[-1]-cervix)+np.linalg.norm(rail[0]-fundus):rail=rail[::-1]
 if np.any(np.linalg.norm(np.diff(rail,axis=0),axis=1)<=1e-8):raise ValueError('Repeated wall vertices')
 return rail
def sample(rail,u):
 distances=np.r_[0.,np.cumsum(np.linalg.norm(np.diff(rail,axis=0),axis=1))]
 s=u*distances[-1]
 return np.column_stack([np.interp(s,distances,rail[:,dim]) for dim in range(2)]),distances[-1]
def layout(shapes,config=CONFIG):
 validate_shapes(shapes)
 if min(config['spacing_px'],config['radial_baseline_px'])<=0 or config['endpoint_margin_px']<0:raise ValueError('Invalid sampling lengths')
 if not config['band_fractions'] or any(not 0<f<1 for f in config['band_fractions']):raise ValueError('Band fractions must lie inside (0,1)')
 cervix=shapes['cervix_point'][0];fundus=shapes['fundus_point'][0]
 ra=orient(shapes['ant_wall'],cervix,fundus);rp=orient(shapes['post_wall'],cervix,fundus)
 lengths=[float(np.linalg.norm(np.diff(r,axis=0),axis=1).sum()) for r in [ra,rp]]
 trim=config['endpoint_margin_px']/min(lengths)
 if trim>=.5:raise ValueError('Annotated path shorter than the proposed endpoint margins')
 count=int(np.ceil(max(lengths)*(1-2*trim)/config['spacing_px']))+1
 u=np.linspace(trim,1-trim,count)
 aa,la=sample(ra,u);pp,lp=sample(rp,u)
 rails=np.stack((aa,pp));center=(aa+pp)/2
 normals=[];radial=[];normal_tangent_error=[]
 for wi,boundary in enumerate((aa,pp)):
  tangent=np.gradient(boundary,axis=0)
  tangent/=np.linalg.norm(tangent,axis=1)[:,None]
  normal=np.column_stack((-tangent[:,1],tangent[:,0]))
  # Orient the normal away from the annotated endometrial cavity.
  normal*=np.where(np.einsum('ij,ij->i',normal,boundary-center)>=0,1.,-1.)[:,None]
  outer=boundary+config['radial_baseline_px']*normal
  radial.append(np.stack((boundary,outer),axis=1));normals.append(normal)
  normal_tangent_error.append(float(np.max(np.abs(np.einsum('ij,ij->i',normal,tangent)))))
 band=np.stack([(1-frac)*aa+frac*pp for frac in config['band_fractions']])
 # All IDs are stable; pair topology is explicit instead of relying on adjacency.
 a_points=np.concatenate([pairs.reshape(-1,2) for pairs in radial])
 a_pairs=np.arange(len(a_points)).reshape(2,count,2)
 b_points=band.reshape(-1,2);b_curves=np.arange(len(b_points)).reshape(len(config['band_fractions']),count)
 arc=[u*la,u*lp]
 # Geometry is resampled only at initialization; measured trajectories are not interpolated.
 return dict(u=u,radial=np.array(radial),normals=np.array(normals),band=band,wall=rails,
  A_points=a_points,A_pair_ids=a_pairs,B_points=b_points,B_curve_ids=b_curves,
  arc_positions=np.array(arc),B_arc_positions=np.array([np.r_[0.,np.cumsum(np.linalg.norm(np.diff(curve,axis=0),axis=1))] for curve in band]),wall_lengths=lengths,tangent_normal_error=max(normal_tangent_error))

