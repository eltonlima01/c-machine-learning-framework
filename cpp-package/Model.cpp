#include <ml.hpp>

#include <cmath>
#include <stdexcept>

extern "C"
{
    #include <math_kernels.h>
}

namespace ML
{
    Model::Model(REGRESSION_TYPE regressionType, const float param0, const float param1) :
    regressionType{regressionType}, param0{param0}, param1{param1}
    {
        if ((regressionType != ML_LINEAR) && (regressionType != ML_LOGISTIC))
        {
            throw std::invalid_argument("");
        }
    }

    float Model::predict(const float param) const
    {
        return (regressionType == ML_LINEAR) ?
        param0 + (param1 * param) : 1.0f / (1.0f + expf(-(param0 + (param1 * param))));
    }

void Model::train(const Dataset &dataset, const float trainingRate, const int epochs)
{
    if (regressionType == ML_LINEAR)
    {
        ::trainLinear(&this->param0, &this->param1, dataset.getParamXData(), dataset.getParamYData(), dataset.getSize(), trainingRate, epochs);
    }
    else
    {
        ::trainLogistic(&this->param0, &this->param1, dataset.getParamXData(), dataset.getParamYData(), dataset.getSize(), trainingRate, epochs);

    }
}
} // namespace ML