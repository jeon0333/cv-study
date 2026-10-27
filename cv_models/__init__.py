"""교육용 컴퓨터 비전 모델. 논문 원본 구현과 구분해서 사용합니다."""
from .resnet import BasicBlock, TinyResNet

__all__ = ['BasicBlock', 'TinyResNet']
