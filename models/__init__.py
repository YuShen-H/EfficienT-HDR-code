from .efficient_hdr import HDRTransformer as EfficienTHDR
from .efficient_hdr_lite import HDRTransformer as EfficienTHDRLite
from .hdr_transformer_v9 import HDRTransformer as HDRTransformerV9


MODEL_VARIANTS = {
    'main': EfficienTHDR,
    'lite': EfficienTHDRLite,
    'v9': HDRTransformerV9,
}
