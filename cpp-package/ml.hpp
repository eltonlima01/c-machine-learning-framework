#pragma once

namespace ML
{

// ****************************************************************
// * Dataset
// ****************************************************************

class Dataset
{
  public:
    Dataset(const char *datasetPath, const char *paramX, const char *paramY);
    ~Dataset();

    int getDatasetSize() const
    {
        return datasetSize;
    }

    const float *getParamXData() const
    {
        return paramXData;
    }

    const float *getParamYData() const
    {
        return paramYData;
    }

  private:
    int datasetSize{0};
    float *paramXData{nullptr};
    float *paramYData{nullptr};
};

// ****************************************************************
// * Linear Model
// ****************************************************************

class LinearModel
{
  public:
    LinearModel(const float param0, const float param1) : param0{param0}, param1{param1}
    {
    }

    ~LinearModel() = default;

    void train(const Dataset &dataset, const float trainingRate, const int epochs);

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