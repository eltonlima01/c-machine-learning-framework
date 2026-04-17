#pragma once

/* **************************************************************** */
/*    Dataset struct definition for CSV loading & basic functions   */
/* **************************************************************** */

typedef struct MLDataset MLDataset;

MLDataset *mlNewDataset(const char *datasetPath, const char *param_x, const char *param_y, const int samples);

/* *************** */
/* Basic functions */
/* *************** */

void mlDeleteDataset(MLDataset **dataset);

int mlGetDatasetSize(const MLDataset *dataset);
const float *mlGetDatasetParamXData(const MLDataset *dataset);
const float *mlGetDatasetParamYData(const MLDataset *dataset);

/* **************************************************************** */
/*              Model struct creation & basic functions             */
/* **************************************************************** */

typedef struct MLModel MLModel;

/* ********************* */
/* Linear Model creation */
/* ********************* */

MLModel *mlNewLinear(const float param0, const float param1);

/* *********************** */
/* Logistic Model creation */
/* *********************** */

MLModel *mlNewLogistic(const float param0, const float param1);

/* *************** */
/* Basic functions */
/* *************** */

void mlDeleteModel(MLModel **model);

float mlSigmoid(const float param);
float mlPredict(const MLModel *model, const float param);
float mlMSE(const MLDataset *dataset, const MLModel *model);
int mlSigmoidClassification(const MLModel *model, const float param);
void mlTrain(const MLDataset *dataset, MLModel *model, const float trainingRate, const int epochs);

void mlGetModelParams(const MLModel *model, float *param0, float *param1);
void mlSetModelParams(MLModel *model, const float param0, const float param1);