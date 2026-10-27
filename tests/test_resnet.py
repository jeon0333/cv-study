"""데이터셋 없이 실행하는 모델 구조 테스트."""
import unittest

import torch
from torch import nn

from cv_models.resnet import BasicBlock, TinyResNet


class BasicBlockTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(0)

    def test_identity_shortcut_shape(self):
        block = BasicBlock(16, 16)
        x = torch.randn(2, 16, 32, 32)
        self.assertIsInstance(block.shortcut, nn.Identity)
        self.assertEqual(tuple(block(x).shape), (2, 16, 32, 32))

    def test_projection_shortcut_shape(self):
        block = BasicBlock(16, 32, stride=2)
        x = torch.randn(2, 16, 32, 32)
        self.assertIsInstance(block.shortcut, nn.Sequential)
        self.assertEqual(tuple(block.shortcut(x).shape), (2, 32, 16, 16))
        self.assertEqual(tuple(block(x).shape), (2, 32, 16, 16))

    def test_backward(self):
        block = BasicBlock(8, 8)
        x = torch.randn(2, 8, 16, 16, requires_grad=True)
        block(x).square().mean().backward()
        self.assertIsNotNone(x.grad)
        self.assertTrue(bool(torch.isfinite(x.grad).all()))
        self.assertIsNotNone(block.conv1.weight.grad)

    def test_classifier_output_shape(self):
        model = TinyResNet(num_classes=10)
        model.eval()
        with torch.no_grad():
            self.assertEqual(tuple(model(torch.randn(2, 3, 32, 32)).shape), (2, 10))

    def test_invalid_arguments(self):
        with self.assertRaises(ValueError):
            BasicBlock(0, 16)
        with self.assertRaises(ValueError):
            TinyResNet(blocks_per_stage=(2, 2, 2))


if __name__ == '__main__':
    unittest.main()
