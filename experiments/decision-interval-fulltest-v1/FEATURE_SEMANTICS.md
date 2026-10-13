# Preserved controller semantics

The controller wrapper runs the exact frozen training/evaluation module.
For interval variants it supplies the previous raw statistics from **one
queue removal ago**, then lets the original code perform subtraction and
float32 casting. It does not substitute a block difference or use NumPy to
replace Chainer inference. A single FF removal still precedes the first
learned decision in every condition.


The planner patch, controller adapter and resource supervisor are identical to the locally validated pilot version. See VALIDATION.md.
