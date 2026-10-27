import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

from PIL import Image
import torch

from research.qwen35_omega.data import canonical_rgb, letterbox_square, split_groups
from research.qwen35_omega.fusion import SpatialAlignment, masked_grid_pool


class DataContracts(unittest.TestCase):
    def test_split_is_deterministic_and_keeps_duplicate_images_together(self):
        rows = [{"id": str(i), "image_sha256": str(i // 2), "task_type": "A", "sub_task_type": "B"} for i in range(60)]
        splits = split_groups(rows)
        self.assertEqual(splits, split_groups(rows))
        ownership = {}
        for name, group in splits.items():
            for row in group:
                self.assertEqual(ownership.setdefault(row["image_sha256"], name), name)
        self.assertEqual(sum(map(len, splits.values())), len(rows))

    def test_padding_preserves_all_pixels_and_tracks_valid_region(self):
        image = Image.new("RGB", (20, 3), (12, 34, 56))
        padded, mask, transform = letterbox_square(image)
        box = tuple(transform["content_box_xyxy"])
        self.assertEqual(padded.crop(box).tobytes(), image.tobytes())
        self.assertEqual(sum(mask.getdata()), 20 * 3 * 255)
        self.assertFalse(transform["crop_applied"])
        self.assertEqual(canonical_rgb(Image.new("RGBA", (1, 1), (0, 0, 0, 0))).getpixel((0, 0)), (255, 255, 255))

    def test_media_audit_excludes_test_overlap_without_reading_test_annotations(self):
        with tempfile.TemporaryDirectory() as folder:
            base = Path(folder)
            train, test, out = base / "train", base / "test", base / "out"
            (train / "A").mkdir(parents=True)
            (test / "A").mkdir(parents=True)
            rows = []
            for i in range(20):
                Image.new("RGB", (4, 4), (i, i, i)).save(train / "A" / f"{i}.png")
                rows.append({"id": f"{i}_0", "task_type": "A", "sub_task_type": "B", "question": "Which?", "options": ["yes", "no", "", ""], "answer": 0})
            (train / "data.json").write_text(json.dumps(rows))
            (test / "data.json").write_text("INVALID JSON: test labels must never be read")
            Image.new("RGB", (4, 4), (0, 0, 0)).save(test / "A" / "overlap.png")
            subprocess.run([sys.executable, "-m", "research.qwen35_omega.data", "--train-root", str(train),
                            "--test-image-root", str(test), "--output", str(out), "--workers", "1"],
                           check=True, capture_output=True)
            manifest = json.loads((out / "manifest.json").read_text())
            self.assertEqual(manifest["test_overlap_qa_excluded"], 1)
            self.assertFalse(manifest["test_labels_read"])
            self.assertEqual(manifest["counts"]["full_train"], 19)


class FusionContracts(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(42)
        self.model = SpatialAlignment(visual_width=24, spatial_width=32, width=24, heads=4, scene_branch=True, dropout=0).eval()
        self.visual = torch.randn(2, 9, 24)
        self.local = torch.randn(2, 197, 32)
        self.scene = torch.randn(2, 17, 32)

    def test_batch_isolation_and_fixed_token_shape(self):
        batch = self.model(self.visual, self.local, self.scene)
        alone = self.model(self.visual[:1], self.local[:1], self.scene[:1])
        self.assertEqual(batch.shape, self.visual.shape)
        torch.testing.assert_close(batch[:1], alone, atol=1e-6, rtol=1e-5)

    def test_masked_tokens_cannot_change_valid_outputs(self):
        lv, sv = torch.ones(2, 197, dtype=torch.bool), torch.ones(2, 17, dtype=torch.bool)
        lv[:, -1], sv[:, 0] = False, False
        before = self.model(self.visual, self.local, self.scene, local_valid=lv, scene_valid=sv)
        local, scene = self.local.clone(), self.scene.clone()
        local[:, -1], scene[:, 0] = torch.randn(2, 32) * 1000, torch.randn(2, 32) * 1000
        after = self.model(self.visual, local, scene, local_valid=lv, scene_valid=sv)
        torch.testing.assert_close(before, after, atol=1e-6, rtol=1e-5)

    def test_gradients_reach_alignment_through_frozen_downstream_model(self):
        downstream = torch.nn.Linear(24, 4).requires_grad_(False)
        logits = downstream(self.model(self.visual, self.local, self.scene).mean(1))
        loss = torch.nn.functional.cross_entropy(logits, torch.tensor([0, 1]))
        loss.backward()
        for name in ("local_gate", "local_input.weight", "scene_gate", "scene_input.weight"):
            grad = dict(self.model.named_parameters())[name].grad
            self.assertIsNotNone(grad, name)
            self.assertTrue(torch.isfinite(grad).all(), name)
            self.assertGreater(grad.abs().sum().item(), 0, name)
        self.assertTrue(all(p.grad is None for p in downstream.parameters()))

    def test_masked_pooling_ignores_padding_and_all_padding_bins_are_masked(self):
        patch, valid = torch.randn(1, 4, 4, 3), torch.zeros(1, 4, 4)
        valid[:, :2, :2] = 1
        before, mask = masked_grid_pool(patch, valid, 2)
        patch[:, 2:] = 1e6
        patch[:, :, 2:] = -1e6
        after, mask2 = masked_grid_pool(patch, valid, 2)
        torch.testing.assert_close(before, after)
        self.assertEqual(mask.tolist(), [[True, False, False, False]])
        self.assertTrue(torch.equal(mask, mask2))

    def test_zero_gates_restore_native_tokens_exactly(self):
        with torch.no_grad():
            self.model.local_gate.zero_()
            self.model.scene_gate.zero_()
        result = self.model(self.visual,self.local,self.scene)
        torch.testing.assert_close(result,self.visual,atol=0,rtol=0)


class ScoringContracts(unittest.TestCase):
    def test_patch_projection_preserves_order_and_native_operator(self):
        from types import SimpleNamespace
        from research.qwen35_omega.runtime import bound_patch_convolution
        class Patch(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.proj=torch.nn.Conv3d(1,3,(1,2,2),stride=(1,2,2))
            def forward(self,x):
                return self.proj(x.reshape(-1,1,1,2,2)).flatten(1)
        patch=Patch().eval()
        x=torch.randn(1401,4)
        before=patch(x)
        bound_patch_convolution(SimpleNamespace(patch_embed=patch))
        after=patch(x)
        torch.testing.assert_close(before,after)

    def test_invalid_and_ambiguous_answers_are_not_defaulted(self):
        from research.qwen35_omega.evaluate import parse_answer
        self.assertEqual(parse_answer("Answer: C"),"C")
        self.assertEqual(parse_answer("d."),"D")
        self.assertIsNone(parse_answer("I cannot determine this."))
        self.assertIsNone(parse_answer("Answer: A or Answer: B"))

    def test_selected_logits_preserve_assistant_only_loss_and_gradients(self):
        from torch.nn import functional as F
        logits = torch.randn(3,8,11,requires_grad=True)
        labels = torch.full((3,8),-100)
        labels[0,-2:] = torch.tensor([3,4])
        labels[1,-3:] = torch.tensor([5,2,6])
        labels[2,-1:] = 7
        shifted = F.pad(labels,(0,1),value=-100)[:,1:]
        indices = torch.where((shifted!=-100).any(0))[0]
        full = F.cross_entropy(logits.reshape(-1,11),shifted.reshape(-1))
        selected = F.cross_entropy(logits[:,indices].reshape(-1,11),shifted[:,indices].reshape(-1))
        torch.testing.assert_close(full,selected)
        a, = torch.autograd.grad(full,logits,retain_graph=True)
        b, = torch.autograd.grad(selected,logits)
        torch.testing.assert_close(a,b)


if __name__ == "__main__":
    unittest.main()
