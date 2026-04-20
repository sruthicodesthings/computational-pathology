import tempfile
import unittest
from pathlib import Path
import numpy as np
from src.inference import tile_grid, predict_tiled
from src.registration import root_anchored_translation, atlas_bounds_mask
from src.annotations import component_boxes, boxes_geojson
from src.batch_processing import process_one_brain

class TestV4(unittest.TestCase):
    def test_tiling_covers_edges(self):
        tiles=tile_grid(35,47,16,5)
        coverage=np.zeros((35,47), dtype=int)
        for t in tiles: coverage[t.y:t.y+t.height,t.x:t.x+t.width]+=1
        self.assertTrue((coverage>0).all())
    def test_tiling_invalid(self):
        with self.assertRaises(ValueError): tile_grid(10,10,8,8)
    def test_identity_predictor(self):
        arr=np.arange(35*47,dtype=float).reshape(35,47)/(35*47)
        np.testing.assert_allclose(predict_tiled(arr, lambda x:x,16,5),arr,rtol=1e-6)
    def test_bad_prediction(self):
        with self.assertRaises(ValueError): predict_tiled(np.ones((8,8)),lambda x:np.ones((3,3)),8,0)
    def test_reference_anchor(self):
        pts=np.array([[3.,4.,5.],[4.,5.,6.]])
        out=root_anchored_translation(pts,np.array([3.,4.,5.]),np.array([1.,2.,3.]))
        np.testing.assert_array_equal(out,[[1,2,3],[2,3,4]])
    def test_atlas_bounds(self):
        mask=atlas_bounds_mask(np.array([[1,1,1],[5,1,1],[np.nan,1,1]]),np.array([[0,2],[0,2],[0,2]]))
        self.assertEqual(mask.tolist(),[True,False,False])
    def test_component_boxes(self):
        a=np.zeros((5,6),dtype=int);a[0:2,0:2]=1;a[4,5]=1
        self.assertEqual(len(component_boxes(a,min_pixels=2)),1)
        self.assertEqual(component_boxes(a,min_pixels=2)[0]['box'],(0,0,2,2))
        self.assertEqual(len(boxes_geojson(a)['features']),2)
    def test_safe_batch(self):
        with tempfile.TemporaryDirectory() as root:
            r=Path(root); src=r/'input'/'brain1';src.mkdir(parents=True)
            (src/'scan.txt').write_text('image placeholder')
            def processor(stage,out):
                (out/'result.txt').write_text((stage/'scan.txt').read_text())
            dest=process_one_brain(src,r/'stage',r/'results',processor,cleanup=True)
            self.assertEqual((dest/'result.txt').read_text(),'image placeholder')
            self.assertTrue((src/'scan.txt').exists())
            self.assertFalse((r/'stage'/'brain1').exists())
    def test_batch_no_output_retains_stage(self):
        with tempfile.TemporaryDirectory() as root:
            r=Path(root);src=r/'input'/'brain1';src.mkdir(parents=True)
            (src/'data').write_text('data')
            with self.assertRaises(RuntimeError):
                process_one_brain(src,r/'stage',r/'out',lambda s,d:None,cleanup=True)
            self.assertTrue((r/'stage'/'brain1'/'data').exists())

if __name__=='__main__': unittest.main()
