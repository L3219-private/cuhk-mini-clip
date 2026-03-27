# tests/test_models.py

import torch
from src.models.image_encoder_smallcnn import ImageEncoder_SmallCNN
from src.models.image_encoder_resnet18 import ImageEncoder_ResNet18
from src.models.image_encoder_resnet50 import ImageEncoder_ResNet50
from src.models.text_encoder_textcnn import TextEncoder_TextCNN

IMAGE_CONST = torch.randn(2, 3, 224, 224)
TEXT_CONST  = torch.randint(0, 5000, (2, 15))

# SmallCNN
def test_smallcnn_shape():
    assert ImageEncoder_SmallCNN(embed_dim=128)(IMAGE_CONST).shape == (2, 128)

def test_smallcnn_no_nan():  # Not a number
    assert not torch.isnan(ImageEncoder_SmallCNN(embed_dim=128)(IMAGE_CONST)).any()  # all false?

def test_smallcnn_backward():
    ImageEncoder_SmallCNN(embed_dim=128)(IMAGE_CONST).sum().backward()  # no error -> pass test

# ResNet18
def test_resnet18_shape():
    assert ImageEncoder_ResNet18(embed_dim=128)(IMAGE_CONST).shape == (2, 128)

def test_resnet18_no_nan():
    assert not torch.isnan(ImageEncoder_ResNet18(embed_dim=128)(IMAGE_CONST)).any()

def test_resnet18_backward():
    ImageEncoder_ResNet18(embed_dim=128)(IMAGE_CONST).sum().backward()

# ResNet50
def test_resnet50_shape():
    assert ImageEncoder_ResNet50(embed_dim=128)(IMAGE_CONST).shape == (2, 128)

def test_resnet50_no_nan():
    assert not torch.isnan(ImageEncoder_ResNet50(embed_dim=128)(IMAGE_CONST)).any()

def test_resnet50_backward():
    ImageEncoder_ResNet50(embed_dim=128)(IMAGE_CONST).sum().backward()

# TextCNN
def test_textcnn_shape():
    assert TextEncoder_TextCNN(vocab_size=5000, embed_dim=128)(TEXT_CONST).shape == (2, 128)

def test_textcnn_no_nan():
    assert not torch.isnan(TextEncoder_TextCNN(vocab_size=5000, embed_dim=128)(TEXT_CONST)).any()

def test_textcnn_backward():
    TextEncoder_TextCNN(vocab_size=5000, embed_dim=128)(TEXT_CONST).sum().backward()

def test_clip_forward_compatible():
    """SmallCNN + TextCNN can produce a valid (B, B) similarity matrix."""
    import torch.nn.functional as F
    img_out  = F.normalize(ImageEncoder_SmallCNN(embed_dim=128)(IMAGE_CONST),  dim=-1)
    text_out = F.normalize(TextEncoder_TextCNN(vocab_size=5000, embed_dim=128)(TEXT_CONST), dim=-1)
    similarity_matrix = img_out @ text_out.T
    assert similarity_matrix.shape == (2, 2)
    assert not torch.isnan(similarity_matrix).any()
