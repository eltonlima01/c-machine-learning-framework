#pragma once

typedef struct LinearModel LinearModel;

LinearModel *newLM(const float param_0, const float param_1);
void deleteLM(LinearModel **linearModel);

float predict(const LinearModel *linearModel, const float x);

float getParam_0(const LinearModel *linearModel);
float getParam_1(const LinearModel *linearModel);

typedef struct Dataset Dataset;

Dataset *newDataset(const char *datasetPath, const char *param_x, const char *param_y, const int samples);
void deleteDataset(Dataset **dataset);

float getParam_x(const Dataset *dataset, int index);
float getParam_y(const Dataset *dataset, int index);