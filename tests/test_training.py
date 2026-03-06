import unittest
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from src.training import (LossConfig, targets_and_weights, WeightedBinaryDiceBCELoss,
                          group_disjoint_split, train_one_epoch, evaluate)

class TrainingTests(unittest.TestCase):
    def test_mapping(self):
        t, w = targets_and_weights(torch.tensor([[[0,1,2,3]]]), LossConfig())
        self.assertEqual(t.tolist(), [[[0.,1.,0.,1.]]])
        self.assertEqual(w.tolist(), [[[1.,3.,5.,0.5]]])

    def test_hard_negative_weight_increases_bce_penalty(self):
        labels = torch.tensor([[[2]]]); logits = torch.tensor([[[[4.]]]])
        base = WeightedBinaryDiceBCELoss(LossConfig(dice_coefficient=0))
        # Per-image weight normalization makes uniform single-pixel weights cancel.
        # Compare mixed hard-negative/background pixels instead.
        labels = torch.tensor([[[2,0]]]); logits = torch.tensor([[[[4.,-4.]]]])
        high = base(logits, labels)
        low = WeightedBinaryDiceBCELoss(LossConfig(hard_negative=1, dice_coefficient=0))(logits,labels)
        self.assertGreater(float(high), float(low))

    def test_perfect_prediction_better(self):
        labels = torch.tensor([[[1,2],[0,3]]])
        good = torch.tensor([[[[9.,-9.],[-9.,9.]]]])
        bad = -good
        fn = WeightedBinaryDiceBCELoss()
        self.assertLess(float(fn(good,labels)),float(fn(bad,labels)))

    def test_gradients(self):
        logits = torch.zeros((2,1,4,4), requires_grad=True)
        labels = torch.randint(0,4,(2,4,4))
        WeightedBinaryDiceBCELoss()(logits,labels).backward()
        self.assertTrue(torch.isfinite(logits.grad).all())

    def test_invalid_label(self):
        with self.assertRaises(ValueError):
            targets_and_weights(torch.tensor([[[4]]]),LossConfig())

    def test_animal_disjoint(self):
        ids = ['a','a','b','b','c','c','d','d','e','e','f','f']
        s = group_disjoint_split(ids, seed=4)
        sets = [set(ids[i] for i in s[k]) for k in ('train','val','test')]
        self.assertTrue(all(sets))
        self.assertFalse(sets[0]&sets[1] or sets[0]&sets[2] or sets[1]&sets[2])
        self.assertEqual(s,group_disjoint_split(ids,seed=4))

    def test_train_evaluate_smoke(self):
        images = torch.randn(4,1,8,8)
        labels = torch.randint(0,4,(4,8,8))
        loader = DataLoader(TensorDataset(images,labels),batch_size=2)
        model = nn.Conv2d(1,1,1)
        opt = torch.optim.AdamW(model.parameters(),lr=1e-3)
        loss = WeightedBinaryDiceBCELoss()
        self.assertGreaterEqual(train_one_epoch(model,loader,opt,loss,'cpu'),0)
        metrics = evaluate(model,loader,loss,'cpu')
        self.assertIn('hard_negative_fpr',metrics)
        self.assertIsNotNone(metrics['manual_dice'])

if __name__ == '__main__': unittest.main()
