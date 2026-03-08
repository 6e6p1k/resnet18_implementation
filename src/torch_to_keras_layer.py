import keras
import torch
import torch.nn as nn
from typing import Optional, Tuple


class TorchLayerWrapper(keras.layers.Layer):
    """
    Wraps any PyTorch nn.Module as a Keras layer.
    Output shape is inferred automatically via a dummy forward pass.

    Args:
        module:      A PyTorch nn.Module instance.
        name:        Optional name for the Keras layer.
        trainable:   Whether the layer's weights should be trainable.
    """

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

        # bypass Keras attribute tracking to prevent auto-wrapping in TorchModuleWrapper
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
    """
    Wraps a PyTorch nn.Module into a Keras layer.

    Args:
        module:    A PyTorch nn.Module instance to wrap.
        name:      Optional name for the Keras layer.
        trainable: Whether the layer's weights should be trainable (default: True).

    Returns:
        A TorchLayerWrapper — a Keras layer that runs the torch module.

    Raises:
        TypeError:  If module is not an nn.Module.
        ValueError: If Keras backend is not set to 'torch'.
    """
    return TorchLayerWrapper(module, name=name, trainable=trainable)