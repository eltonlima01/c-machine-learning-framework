#pragma once

/* Dataset */

typedef struct MLDataset MLDataset;

MLDataset *mlNewDataset(const char *datasetPath, const char *param_x, const char *param_y, const int samples);
void mlDeleteDataset(MLDataset **dataset);

int mlGetDatasetSize(const MLDataset *dataset);
const float *mlGetDatasetParamX(const MLDataset *dataset);
const float *mlGetDatasetParamY(const MLDataset *dataset);

/* Linear Model */

typedef struct MLLinearModel MLLinearModel;

MLLinearModel *mlNewLinearModel(const float param0, const float param1);
void mlDeleteLinearModel(MLLinearModel **linearModel);

void mlSetLinearModelParam0(MLLinearModel *linearModel, const float param0);
float mlGetLinearModelParam0(const MLLinearModel *linearModel);

void mlSetLinearModelParam1(MLLinearModel *linearModel, const float param1);
float mlGetLinearModelParam1(const MLLinearModel *linearModel);

float mlPredict(const MLLinearModel *linearModel, const float x);
float mlMSE(const MLDataset *dataset, const MLLinearModel *linearModel);
void mlTrain(const MLDataset *dataset, MLLinearModel *linearModel, const float trainingRate, const int epochs);