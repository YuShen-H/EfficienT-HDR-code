from .efficient_hdr import HDRTransformer as EfficienTHDR
from .efficient_hdr_lite import HDRTransformer as EfficienTHDRLite


MODEL_VARIANTS = {
    'main': EfficienTHDR,
    'lite': EfficienTHDRLite,
}
