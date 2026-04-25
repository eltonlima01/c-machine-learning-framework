#pragma once

typedef enum RegressionType
{
    ML_LINEAR = 0,
    ML_LOGISTIC = 1
} RegressionType;

/* **************************************************************** */
/*    Dataset struct definition for CSV loading & basic functions   */
/* **************************************************************** */

typedef struct MLDataset MLDataset;

MLDataset *mlNewDataset(const char *datasetPath, const char *param_x, const char *param_y, const int samples);
void mlDeleteDataset(MLDataset **dataset);

/* *************** */
/* Basic functions */
/* *************** */

int mlGetDatasetSize(const MLDataset *dataset);
const float *mlGetDatasetParamXData(const MLDataset *dataset);
const float *mlGetDatasetParamYData(const MLDataset *dataset);

/* **************************************************************** */
/*             Model struct definition & basic functions            */
/* **************************************************************** */

typedef struct MLModel MLModel;

MLModel *mlNewModel(RegressionType regressionType, const float param0, const float param1);
void mlDeleteModel(MLModel **model);

void mlGetModelParams(const MLModel *model, float *param0, float *param1);
void mlSetModelParams(MLModel *model, const float param0, const float param1);

/* *************** */
/* Basic functions */
/* *************** */

float mlPredict(const MLModel *model, const float param);
float mlMSE(const MLModel *restrict model, const MLDataset *restrict dataset);
void mlTrainModel(MLModel *restrict model, const MLDataset *restrict dataset, const float trainingRate,
                  const int epochs);