#include <ml.hpp>
#include <math_kernels.h>

namespace ML
{
void LinearModel::train(const Dataset &dataset, const float trainingRate, const int epochs)
{
    ::train(&this->param0, &this->param1, dataset.getParamXData(), dataset.getParamYData(), dataset.getDatasetSize(),
            trainingRate, epochs);
}
} // namespace ML