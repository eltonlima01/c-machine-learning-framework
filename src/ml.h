#pragma once

// ================================================================ //

typedef struct Dataset Dataset;

Dataset *newDataset(const char *datasetPath, const char *param_x, const char *param_y, const int samples);
void deleteDataset(Dataset **dataset);

float dataset_getParam_x(const Dataset *dataset, int index);
float dataset_getParam_y(const Dataset *dataset, int index);
int dataset_getSamples(const Dataset *dataset);

// ================================================================ //

typedef struct LinearModel LinearModel;

LinearModel *newLM(const float param_0, const float param_1);
void deleteLM(LinearModel **linearModel);

float lm_getParam_0(const LinearModel *linearModel);
float lm_getParam_1(const LinearModel *linearModel);

float predict(const LinearModel *linearModel, const float x);
float MSE(const Dataset *dataset, const LinearModel *linearModel);
void train(const Dataset *dataset, LinearModel *linearModel, float trainingRate, int epochs);