#include <ml.hpp>

#include <fstream>
#include <sstream>
#include <vector>

using namespace std;

namespace ML
{
Dataset::Dataset(const char *datasetPath, const char *paramX, const char *paramY)
{
    ifstream file(datasetPath);

    if (file.is_open() == false)
    {
        return;
    }

    string line, column;
    vector<string> headers;

    if (getline(file, line))
    {
        stringstream ss(line);

        while (getline(ss, column, ','))
        {
            if ((column.empty() == false) && (column.back() == '\r'))
            {
                column.pop_back();
            }

            headers.push_back(column);
        }
    }

    int xIndex = -1;
    int yIndex = -1;

    for (int i = 0; i < headers.size(); ++i)
    {
        if (headers[i] == paramX)
        {
            xIndex = i;
        }

        if (headers[i] == paramY)
        {
            yIndex = i;
        }
    }

    if ((xIndex == -1) || (yIndex == -1))
    {
        return;
    }

    vector<float> tmprrX, tmprrY;

    while (getline(file, line))
    {
        if (line.empty() == true)
        {
            continue;
        }

        string value;
        stringstream ss(line);

        int cIndex = 0;

        float pX = 0.0f;
        float pY = 0.0f;

        bool foundX = false;
        bool foundY = false;

        while (getline(ss, value, ','))
        {
            if (cIndex == xIndex)
            {
                pX = stof(value);
                foundX = true;
            }
            else if (cIndex == yIndex)
            {
                pY = stof(value);
                foundY = true;
            }

            cIndex++;
        }

        if ((foundX == true) && (foundY))
        {
            tmprrX.push_back(pX);
            tmprrY.push_back(pY);
        }
    }

    size = tmprrX.size();

    if (size > 0)
    {
        paramXData = new float[size];
        paramYData = new float[size];

        for (int i = 0; i < size; i++)
        {
            paramXData[i] = tmprrX[i];
            paramYData[i] = tmprrY[i];
        }
    }
}

Dataset::~Dataset()
{
    delete[] paramXData;
    delete[] paramYData;
}
}; // namespace ML