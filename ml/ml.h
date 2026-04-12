#pragma once

// **************************************************************** //
// * Dataset * //
// **************************************************************** //

typedef struct Dataset Dataset;

Dataset *newDataset(const char *datasetPath, const char *param_x, const char *param_y, const int samples);
void deleteDataset(Dataset **dataset);

int dataset_getSamples(const Dataset *dataset);
const float *dataset_getParam_x(const Dataset *dataset);
const float *dataset_getParam_y(const Dataset *dataset);

// **************************************************************** //
// * Linear Model * //
// **************************************************************** //

typedef struct LinearModel LinearModel;

LinearModel *newLM(const float param_0, const float param_1);
void deleteLM(LinearModel **linear_model);

void lm_setParam_0(LinearModel *linear_model, float param_0);
float lm_getParam_0(const LinearModel *linear_model);

void lm_setParam_1(LinearModel *linear_model, const float param_1);
float lm_getParam_1(const LinearModel *linear_model);

float predict(const LinearModel *linear_model, const float x);
float mse(const Dataset *dataset, const LinearModel *linear_model);
void train(const Dataset *dataset, LinearModel *linear_model, const float trainingRate, const int epochs);