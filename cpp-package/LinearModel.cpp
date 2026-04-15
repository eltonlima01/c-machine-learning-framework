#include <math_kernels.h>
#include <ml.hpp>

namespace ML
{
void LinearModel::train(const float *paramX, const float *paramY, const int samples, const float trainingRate,
                        const int epochs)
{
    ::train(&param0, &param1, paramX, paramY, samples, trainingRate, epochs);
}
} // namespace ML