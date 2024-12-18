from alphagen.data.expression import Feature, Ref
from alphagen_qlib.stock_data import FeatureType


high = Feature(FeatureType.HIGH)
low = Feature(FeatureType.LOW)
volume = Feature(FeatureType.VOLUME)
open_ = Feature(FeatureType.OPEN)
close = Feature(FeatureType.CLOSE)
vwap = Feature(FeatureType.VWAP)
opt_put_oi = Feature(FeatureType.OPT_PUT_OI)
opt_call_oi = Feature(FeatureType.OPT_CALL_OI)
estimate_target = Feature(FeatureType.ESTIMATE_TARGET)
shortint = Feature(FeatureType.SHORTINT)
target = Ref(close, -6) / Ref(close, -1)  - 1