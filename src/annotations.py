"""Minimal dependency-free conversion from binary detections to GeoJSON polygons.

Bounding rectangles are deliberately used as an illustrative fallback. They do not
preserve biological boundaries and must not be used for quantitative morphology.
"""
from __future__ import annotations
import json
import numpy as np

def component_boxes(mask: np.ndarray, min_pixels: int = 1) -> list[dict]:
    """4-connected components with bounding boxes (x0,y0,x1,y1), exclusive maxima."""
    a = np.asarray(mask)
    if a.ndim != 2 or min_pixels < 1:
        raise ValueError("Expected 2D mask and positive min_pixels")
    active = a.astype(bool)
    visited = np.zeros(active.shape, dtype=bool)
    h,w = active.shape
    results = []
    for y in range(h):
        for x in range(w):
            if not active[y,x] or visited[y,x]:
                continue
            stack=[(y,x)]; visited[y,x]=True; count=0
            x0=x1=x; y0=y1=y
            while stack:
                cy,cx=stack.pop(); count+=1
                x0=min(x0,cx); x1=max(x1,cx); y0=min(y0,cy); y1=max(y1,cy)
                for ny,nx in ((cy-1,cx),(cy+1,cx),(cy,cx-1),(cy,cx+1)):
                    if 0<=ny<h and 0<=nx<w and active[ny,nx] and not visited[ny,nx]:
                        visited[ny,nx]=True; stack.append((ny,nx))
            if count>=min_pixels:
                results.append(dict(box=(x0,y0,x1+1,y1+1), pixels=count))
    return results

def boxes_geojson(mask: np.ndarray, min_pixels: int = 1) -> dict:
    features=[]
    for i, obj in enumerate(component_boxes(mask, min_pixels)):
        x0,y0,x1,y1=obj['box']
        coords=[[x0,y0],[x1,y0],[x1,y1],[x0,y1],[x0,y0]]
        features.append({'type':'Feature','properties':{'component_id':i,'pixel_count':obj['pixels'],
              'geometry_approximation':'bounding_box'},
              'geometry':{'type':'Polygon','coordinates':[coords]}})
    return {'type':'FeatureCollection','features':features}
