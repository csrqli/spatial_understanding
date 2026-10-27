import unittest
import torch
from .fusion import Alignment
from .prepare import Union


class Contracts(unittest.TestCase):
    def test_connected_views(self):
        u=Union();u.join('a','b');u.join('c','b');u.join('d','e')
        self.assertEqual(u.find('a'),u.find('c'));self.assertNotEqual(u.find('a'),u.find('d'))

    def test_identity_dtype_and_padding(self):
        torch.manual_seed(7)
        m=Alignment(scene=True).eval()
        v=torch.randn(2,9,1152).bfloat16();local=torch.randn(2,197,2048);scene=torch.randn(2,17,2048)
        valid=torch.ones(2,9,dtype=torch.bool);valid[0,5:]=False
        with torch.autocast('cpu',dtype=torch.bfloat16):out=m(v,local,scene,visual_valid=valid)
        self.assertEqual(out.dtype,v.dtype);torch.testing.assert_close(out,v,atol=0,rtol=0)

    def test_per_image_isolation_after_update(self):
        torch.manual_seed(9)
        m=Alignment(scene=True).eval()
        with torch.no_grad():m.local_output.weight.normal_(std=.01);m.scene_output.weight.normal_(std=.01)
        v=torch.randn(2,9,1152);local=torch.randn(2,197,2048);scene=torch.randn(2,17,2048)
        first=m(v,local,scene)
        changed=local.clone();changed[1]*=20
        second=m(v,changed,scene)
        torch.testing.assert_close(first[0],second[0],atol=0,rtol=0)


if __name__=='__main__':unittest.main()
