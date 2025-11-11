"""ML training module for dataset generation, model training, and inference"""

try:
    from .generate_dataset import SyntheticDatasetGenerator
    from .train_large_model import LargeScaleMLTrainer
    from .inference import HLDQualityPredictor
except ImportError as e:
    import warnings
    warnings.warn(f"Could not import all training modules: {e}")

__all__ = ['SyntheticDatasetGenerator', 'LargeScaleMLTrainer', 'HLDQualityPredictor']
# Expose training and inference modules
from .train_large_model import LargeScaleMLTrainer
from .inference import HLDQualityPredictor

__all__ = ['train_large_model', 'inference', 'LargeScaleMLTrainer', 'HLDQualityPredictor']
