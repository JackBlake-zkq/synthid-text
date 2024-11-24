# Copyright 2024 DeepMind Technologies Limited
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""Code for Mean and Weighted Mean scoring functions."""

from typing import Optional
import jax.numpy as jnp
from scipy.stats import binom, norm


def frequentist_score(
    g_values: jnp.ndarray,
    mask: jnp.ndarray,
) -> jnp.ndarray:
  """Computes the Frequentist score.

  Args:
    g_values: g-values of shape [batch_size, seq_len, watermarking_depth].
    mask: A binary array shape [batch_size, seq_len] indicating which g-values
      should be used. g-values with mask value 0 are discarded.

  Returns:
    Mean scores, of shape [batch_size]. This is the mean of the unmasked
      g-values.
  """
  watermarking_depth = g_values.shape[-1]
  num_unmasked = jnp.sum(mask, axis=1)  # shape [batch_size]

  sum = jnp.sum(g_values * jnp.expand_dims(mask, 2), axis=(1, 2))
  max_sum = watermarking_depth * num_unmasked

  p_value = 1 - binom.cdf(sum - 1, n=max_sum, p=0.5)

  return -p_value


def weighted_frequentist_score(
    g_values: jnp.ndarray,
    mask: jnp.ndarray,
    weights: Optional[jnp.ndarray] = None,
) -> jnp.ndarray:
  """Computes the Weighted Mean score.

  Args:
    g_values: g-values of shape [batch_size, seq_len, watermarking_depth].
    mask: A binary array shape [batch_size, seq_len] indicating which g-values
      should be used. g-values with mask value 0 are discarded.
    weights: array of non-negative floats, shape [watermarking_depth]. The
      weights to be applied to the g-values. If not supplied, defaults to
      linearly decreasing weights from 10 to 1.

  Returns:
    Weighted Mean scores, of shape [batch_size]. This is the mean of the
      unmasked g-values, re-weighted using weights.
  """
  watermarking_depth = g_values.shape[-1]

  if weights is None:
    weights = jnp.linspace(start=10, stop=1, num=watermarking_depth)

  # Normalise weights so they sum to watermarking_depth.
  weights *= watermarking_depth / jnp.sum(weights)

  # Apply weights to g-values.
  g_values *= jnp.expand_dims(weights, axis=(0, 1))

  T = jnp.size(g_values, axis=2)

  sum = jnp.sum(g_values * jnp.expand_dims(mask, 2), axis=(1, 2))

  k = sum / T

  mean = watermarking_depth / 2
  variance = jnp.sum(jnp.square(weights)) / 4

  p_value = 1 - norm.cdf(k, loc=mean, scale=variance / T)

  return -p_value
