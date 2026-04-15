#pragma once

// ****************************************************************
// * Linear Model
// ****************************************************************

namespace ML
{
class LinearModel
{
  public:
    LinearModel(const float param0, const float param1) : param0{param0}, param1{param1}
    {
    }

    ~LinearModel() = default;

    void train(const float *paramX, const float *paramY, const int samples, const float trainingRate, const int epochs);

    float predict(const float x) const
    {
        return (param0 + (param1 * x));
    }

    void setParam0(const float param0)
    {
        this->param0 = param0;
    }

    float getParam0() const
    {
        return param0;
    }

    void setParam1(const float param1)
    {
        this->param1 = param1;
    }

    float getParam1() const
    {
        return param1;
    }

  private:
    float param0, param1;
};
} // namespace ML