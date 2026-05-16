const LOG_2 = log(2)
const SQRT_PI_OVER_2 = sqrt(pi / 2)

_batch_param_axis1(x::Number) = x
_batch_param_axis1(x::AbstractVector) = reshape(x, 1, length(x))
_batch_param_axis1(x::AbstractArray) = x
