#pragma once

typedef struct LinearModel LinearModel;

LinearModel *newLM(const float param_0, const float param_1);
void deleteLM(LinearModel **linearModel);

float predict(const LinearModel *linearModel, float x);

float getParam_0(const LinearModel *linearModel);
float getParam_1(const LinearModel *linearModel);