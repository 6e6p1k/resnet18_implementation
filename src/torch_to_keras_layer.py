import keras
import torch
import torch.nn as nn
from typing import Optional, Tuple


# wraps a torch module as a keras layer, output shape comes from a dummy forward pass
class TorchLayerWrapper(keras.layers.Layer):
    def __init__(
        self,
        module: nn.Module,
        name: Optional[str] = None,
        trainable: bool = True,
        **kwargs
    ):
        super().__init__(name=name, trainable=trainable, **kwargs)

        if not isinstance(module, nn.Module):
            raise TypeError(f"Expected nn.Module, got {type(module)}")
        if keras.backend.backend() != "torch":
            raise ValueError(
                f"Keras backend must be 'torch', got '{keras.backend.backend()}'."
            )

        # keep keras from wrapping the module itself
        object.__setattr__(self, '_module', module)
        self._output_shape_cache: Optional[Tuple] = None

    def call(self, x):
        if x.device.type != 'meta':
            self._module.to(x.device)
        output = self._module(x)
        if isinstance(output, tuple):
            output = output[0]
        return output


    def compute_output_shape(self, input_shape):
        if self._output_shape_cache is not None:
            return self._output_shape_cache

        concrete_shape = tuple(1 if d is None else d for d in input_shape)
        dummy_input = torch.zeros(concrete_shape)

        with torch.no_grad():
            if not any(p.is_meta for p in self._module.parameters()):
                self._module.to(dummy_input.device)
            output = self._module(dummy_input)
        if isinstance(output, tuple):
            output = output[0]

        output_shape = (input_shape[0],) + tuple(output.shape[1:])
        self._output_shape_cache = output_shape
        return output_shape


def torch_to_keras_layer(
    module: nn.Module,
    name: Optional[str] = None,
    trainable: bool = True,
) -> TorchLayerWrapper:
    return TorchLayerWrapper(module, name=name, trainable=trainable)