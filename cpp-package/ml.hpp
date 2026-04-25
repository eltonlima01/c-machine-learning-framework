#pragma once

typedef enum REGRESSION_TYPE
{
    ML_LINEAR = 0,
    ML_LOGISTIC = 1
} REGRESSION_TYPE;

namespace ML
{

/* **************************************************************** */
/*    Dataset class definition for basic CSV loading & management   */
/* **************************************************************** */

class Dataset
{
  public:
    Dataset(const char *datasetPath, const char *paramX, const char *paramY);
    ~Dataset();

    int getSize() const
    {
      return size;
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
    int size{0};
    float *paramXData{nullptr};
    float *paramYData{nullptr};
};

/* **************************************************************** */
/*                       Model class definition                     */
/* **************************************************************** */

class Model
{
  public:
    Model(REGRESSION_TYPE regressionType, const float param0, const float param1);
    ~Model() = default;

    void train(const Dataset &dataset, const float trainingRate, const int epochs);
    float predict(const float x) const;

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
    REGRESSION_TYPE regressionType;
};
} // namespace ML