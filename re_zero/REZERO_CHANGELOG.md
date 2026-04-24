## Re-Zero Work Summary

This document summarizes the changes made in the `re_zero` branch during the recent refactor and loss/classifier redesign work.

### 1. Module split

Some logic that had previously lived inside `object_representation_model.py` was moved into separate files.

- `loss_function.py`
  - Stores the standalone `ObjectRepresentationLoss` helper.
  - Keeps supervised loss selection logic.
  - Now defines the unsupervised object loss from classifier outputs.
- `top_down_feedback.py`
  - Stores the standalone `TopDownFeedback` helper.
  - Builds pairwise feedback tensors from current spike activity.

As a result, `object_representation_model.py` no longer owns those implementations directly and instead uses helper objects.

### 2. Classifier redesign

`classifier.py` was updated to reflect the current object-centric workflow.

#### MeanSpikeClassifier

- Uses late-time mean spike values.
- For each pixel, computes a scalar mean spike value.
- Builds a pixel-pixel similarity matrix from scalar differences.
- Converts the similarity graph into connected-component object masks.
- Returns:
  - `masks`
  - `mean_spike_grid`

#### SpikeFeatureClassifier

- Uses pixel-wise learned feature vectors from the spike trace.
- Encodes each pixel's spike-time pattern with `PixelPatternEncoder`.
- Normalizes feature vectors and computes cosine-style similarity.
- Converts the similarity graph into connected-component object masks.
- Returns:
  - `masks`
  - `pixel_feature_grid`

#### Current assumption

- The classifier logic currently assumes `B = 1` in the final classification step.
- The overall model still keeps the batch dimension in earlier parts of the pipeline.

### 3. Unsupervised loss redesign

`loss_function.py` was rewritten so that unsupervised loss is no longer computed directly from `spike_trace` and external mask tensors.

Instead, it now uses the classifier output dictionary:

- `masks`
- `mean_spike_grid` or `pixel_feature_grid`

#### New unsupervised flow

1. Read classifier output.
2. Extract object masks.
3. Extract per-pixel grid information.
4. For each object mask, gather the pixels inside that object.
5. Compute reusable components.
6. Combine selected components into the final loss.

#### Components retained

The component-based structure was preserved.

- `within_similarity`
- `between_difference`
- `object_density`
- `between_distance`
- `background_suppression`

#### Variant losses retained

The variant-selection structure was also preserved.

- `unsupervised_object_loss_1234`
- `unsupervised_object_loss_124`
- `unsupervised_object_loss_123`
- `unsupervised_object_loss`

### 4. ObjectRepresentationSNN integration

`object_representation_model.py` was updated to align with the new helper modules and classifier/loss interfaces.

#### Current model behavior

- The model builds:
  - `ReadoutLayer`
  - `KuramotoLayer`
  - `SinusoidalGate`
  - `SNNLayer`
  - `TopDownFeedback`
  - `ObjectRepresentationLoss`
- During `forward(...)`, it still:
  - validates input
  - initializes oscillator state
  - rolls the recurrent dynamics over time
  - updates spikes and top-down feedback
  - builds `spike_trace`
  - runs the classifier

#### Output change

The final classifier result is no longer treated as supervised class logits in the current object-centric workflow.

It is now the classifier output dictionary, such as:

- `{"masks", "mean_spike_grid"}`
- or `{"masks", "pixel_feature_grid"}`

#### New model wrappers

`ObjectRepresentationSNN` now exposes wrappers that delegate to `ObjectRepresentationLoss`:

- `object_spike_loss_components(...)`
- `unsupervised_object_loss_1234(...)`
- `unsupervised_object_loss_124(...)`
- `unsupervised_object_loss_123(...)`
- `unsupervised_object_loss(...)`

### 5. Hyperparameter updates

`hyperparameters.py` was updated to support the separated loss helper and current loss selection flow.

Added/kept:

- `loss_function`
- `object_loss_function`
- component weights
- object density target
- object distance scale

No additional cleanup was required during the latest integration step because the currently defined fields are still referenced.

### 6. Current verified behavior

The following checks were run during development:

- `classifier.py` syntax check
- `loss_function.py` syntax check
- `object_representation_model.py` syntax check
- `MeanSpikeClassifier` output check
- `SpikeFeatureClassifier` output check
- `ObjectRepresentationSNN.forward(...)` run with `B = 1`
- unsupervised loss computation from classifier output

Observed examples:

- `MeanSpikeClassifier` output keys:
  - `masks`
  - `mean_spike_grid`
- `SpikeFeatureClassifier` output keys:
  - `masks`
  - `pixel_feature_grid`
- `history["spikes"]` example shape:
  - `(1, 12, 256)`

### 7. Current caveat

The current `re_zero` pipeline is now aligned around object-mask generation and unsupervised object-centric losses.

This means:

- the classifier is currently object-centric, not class-logit-centric
- supervised classification with `cross_entropy` is not the main active path unless a separate classification head is introduced again

### 8. Files touched in this work

- `re_zero/classifier.py`
- `re_zero/loss_function.py`
- `re_zero/top_down_feedback.py`
- `re_zero/object_representation_model.py`
- `re_zero/hyperparameters.py`

