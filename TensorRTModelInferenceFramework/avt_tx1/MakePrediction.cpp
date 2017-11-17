// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License. See LICENSE in the project root for license information.
// Based on code provided by NVIDIA's JetPack Sample

#include <Python.h>
#include <opencv.hpp>
#include <assert.h>
#include <fstream>
#include <sstream>
#include <iostream>
#include <cmath>
#include <time.h>
#include <algorithm>
#include <chrono>
#include <string.h>
#include <map>
#include <iomanip>
#include <random>
#include <iterator>
#include <math.h>
#include <cuda_runtime.h>

#include <cstring>
#include <cstdlib>
#include <vector>

#include <string>
#include <stdio.h>

#include <cstdlib>
#include <cuda_runtime_api.h>
#include <string>
#include <sys/stat.h>
#include <unordered_map>
#include <cassert>
#include <vector>
#include "NvInfer.h"
#include "NvUffParser.h"

#include "NvUtils.h"

using namespace nvuffparser;
using namespace nvinfer1;
using namespace std;
//using namespace cv;

#include "common.h"

static Logger gLogger;

#define MAX_WORKSPACE (1 << 30)

#define RETURN_AND_LOG(ret, severity, message)                                              \
    do {                                                                                    \
        std::string error_message = "sample_uff_cars: " + std::string(message);            \
        gLogger.log(ILogger::Severity::k ## severity, error_message.c_str());               \
        return (ret);                                                                       \
    } while(0)

inline int64_t volume(const Dims& d)
{
    int64_t v = 1;
    for (int64_t i = 0; i < d.nbDims; i++)
        v *= d.d[i];
    return v;
}


inline unsigned int elementSize(DataType t)
{
    switch (t)
    {
        case DataType::kFLOAT: return 4;
        case DataType::kHALF: return 2;
        case DataType::kINT8: return 1;
    }
    assert(0);
    return 0;
}

std::vector<std::pair<int64_t, DataType>>
calculateBindingBufferSizes(const ICudaEngine& engine, int nbBindings, int batchSize)
{
    std::vector<std::pair<int64_t, DataType>> sizes;
    for (int i = 0; i < nbBindings; ++i)
    {
        Dims dims = engine.getBindingDimensions(i);
        DataType dtype = engine.getBindingDataType(i);

        int64_t eltCount = volume(dims) * batchSize;
        sizes.push_back(std::make_pair(eltCount, dtype));
    }

    return sizes;
}

static const int INPUT_H = 224;
static const int INPUT_W = 224;
static const int NUM_CHANNELS = 3;
static const int OUTPUT_SIZE = 10;

//define Prediction type
typedef std::pair<string, float> Prediction;

int main(int argc, char** argv){
    cout << "C++ File Is Called" << endl;
    return 0;
}

void* safeCudaMalloc(size_t memSize)
{
    void* deviceMem;
    CHECK(cudaMalloc(&deviceMem, memSize));
    if (deviceMem == nullptr)
    {
        std::cerr << "Out of memory" << std::endl;
        exit(1);
    }
    return deviceMem;
}

//Evaluator class based on model files
class Evaluator {
    public:
        Evaluator();
        int evaluate();
        void getEval();
        void getProb();
        const char * setImg(const char * img);
        void* createPngCudaBuffer(int64_t eltCount, DataType dtype, int run);
        ICudaEngine* loadModelAndCreateEngine(const char* uffFile, int maxBatchSize,
                                                      IUffParser* parser);
        void execute(ICudaEngine& engine);
        void printOutput(int64_t eltCount, DataType dtype, void* buffer);
        void readTXTFile(const char * img_file, float buffer[INPUT_H*INPUT_W*NUM_CHANNELS]);
	std::vector<std::pair<int64_t, DataType>> calculateBindingBufferSizes(const ICudaEngine& engine, int nbBindings, int batchSize);
	void* safeCudaMalloc(size_t memSize);

    private:
        Prediction my_pred;
        ofstream outputeval;
        ofstream outputprob;
        const char * image_file = "";
	    vector<float> uni_buffer;
};

//this makes the Evaluator class invokable by Python
extern "C" {
    Evaluator * Evaluator_new() { return new Evaluator(); }
    void Evaluator_geteval(Evaluator * Evaluator) { Evaluator->getEval(); }
    void Evaluator_getprob(Evaluator * Evaluator) { Evaluator->getProb(); }
    void Evaluator_setimg(Evaluator * Evaluator, const char * img) {
        Evaluator->setImg(img);
    }
    void Evaluator_evaluate(Evaluator * Evaluator) { Evaluator->evaluate(); }
}

//default constructor
Evaluator::Evaluator(void){
    cout << "Evaluator Is Instantiated" << endl;
}

//returns the classification of the image
void Evaluator::getEval() {
    outputeval.open("outputeval.txt");
    if (my_pred.first.compare("car")){
        outputeval << "Car";
    }
    else{
        outputeval << "No Car";
    }
    outputeval.close();
}

//returns the probability of the given classification
void Evaluator::getProb(){
    stringstream ss;
    ss << fixed << setprecision(4) << my_pred.second << endl;
    const char * mypred_str = ss.str().c_str();
    
    outputprob.open("outputprob.txt");
    outputprob << mypred_str;
    outputprob.close();
}

const char * Evaluator::setImg(const char * img_file){
    image_file = img_file;
    return image_file;

}

void* Evaluator::safeCudaMalloc(size_t memSize)
{
    void* deviceMem;
    CHECK(cudaMalloc(&deviceMem, memSize));
    if (deviceMem == nullptr)
    {
        std::cerr << "Out of memory" << std::endl;
        exit(1);
    }
    return deviceMem;
}

std::vector<std::pair<int64_t, DataType>>
Evaluator::calculateBindingBufferSizes(const ICudaEngine& engine, int nbBindings, int batchSize)
{
    std::vector<std::pair<int64_t, DataType>> sizes;
    for (int i = 0; i < nbBindings; ++i)
    {
        Dims dims = engine.getBindingDimensions(i);
        DataType dtype = engine.getBindingDataType(i);

        int64_t eltCount = volume(dims) * batchSize;
        sizes.push_back(std::make_pair(eltCount, dtype));
    }
    return sizes;
}

void Evaluator::readTXTFile(const char * img_file, float buffer[INPUT_H*INPUT_W*NUM_CHANNELS]){
    string line;
    ifstream txtfile;
    int sz = INPUT_H*INPUT_W*NUM_CHANNELS;

    int i = 0;

    uni_buffer.clear();
    img_file = "/home/nvidia/avt/avt_tx1/ifile.txt";
    txtfile.open(img_file,ios::in);

    if(!txtfile.is_open()){
	cout << "Unable to open file " << image_file << endl;
    }
    else{
	//adding values from python input to universal buffer
	while ((txtfile >> line || !txtfile.eof()) && i < sz){
	    uni_buffer.push_back(stof(line));
            i++;
	}
	txtfile.close();
    }
}

int Evaluator::evaluate(){
    //get the UFF file
    auto fileName = "/home/nvidia/avt/avt_tx1/frozen_alexnet.uff";

    int maxBatchSize = 1;
    auto parser = createUffParser();

    //Register tensorflow input; different depending on the UFF file
    //parser->registerInput("x", DimsCHW(3, 135, 240));
    parser->registerInput("images_placeholder", DimsCHW(3,224,224));
    //parser->registerInput("Input_0", DimsCHW(1, 28, 28));

    //load model and create engine
    ICudaEngine* engine = loadModelAndCreateEngine(fileName, maxBatchSize, parser);
    if (!engine)
        RETURN_AND_LOG(EXIT_FAILURE, ERROR, "Model load failed");

    //free parser memory
    parser->destroy();

    //execute engine and print prediction
    execute(*engine);

    return 1;
}

ICudaEngine* Evaluator::loadModelAndCreateEngine(const char* uffFile, int maxBatchSize,
                                      IUffParser* parser)
{
    //create builder and network needed to make engine
    IBuilder* builder = createInferBuilder(gLogger);
    INetworkDefinition* network = builder->createNetwork();

    cudaError_t error = cudaGetLastError();
    if(error != cudaSuccess)
    {
        // print the CUDA error message and exit
        cout << "CUDA error: " << cudaGetErrorString(error) << endl;
        exit(-1);
    }

    if (!parser->parse(uffFile, *network, nvinfer1::DataType::kFLOAT))
        RETURN_AND_LOG(nullptr, ERROR, "Fail to parse");

    //create the engine
    builder->setMaxBatchSize(maxBatchSize);
    builder->setMaxWorkspaceSize(MAX_WORKSPACE);

    ICudaEngine* engine = builder->buildCudaEngine(*network);
    if (!engine)
        RETURN_AND_LOG(nullptr, ERROR, "Unable to create engine");

    //free memory from network and parser
    network->destroy();
    builder->destroy();

    return engine;
}

void Evaluator::execute(ICudaEngine& engine)
{
    IExecutionContext* context = engine.createExecutionContext();
    int batchSize = 1;

    //gets input and output buffer pointers that we pass to the engine
    int nbBindings = engine.getNbBindings();

    std::vector<void*> buffers(nbBindings);
    auto buffersSizes = calculateBindingBufferSizes(engine, nbBindings, batchSize);

    //allocating memory for input
    int bindingIdxInput = 0;
    for (int i = 0; i < nbBindings; ++i)
    {
        if (engine.bindingIsInput(i))
            bindingIdxInput = i;
        else
        {
            auto bufferSizesOutput = buffersSizes[i];
            buffers[i] = safeCudaMalloc(bufferSizesOutput.first *
                                        elementSize(bufferSizesOutput.second));
        }
    }

    auto bufferSizesInput = buffersSizes[bindingIdxInput];

    int iterations = 1;
    int numberRun = 10;
    for (int i = 0; i < iterations; i++)
    {
        float total = 0, ms;
        for (int run = 0; run < numberRun; run++)
        {
            //Create input buffer here and get inputs
            buffers[bindingIdxInput] = createPngCudaBuffer(bufferSizesInput.first,
                                               bufferSizesInput.second, run);

            auto t_start = std::chrono::high_resolution_clock::now();
            context->execute(batchSize, &buffers[0]);
            auto t_end = std::chrono::high_resolution_clock::now();
            ms = std::chrono::duration<float, std::milli>(t_end - t_start).count();
            total += ms;

            for (int bindingIdx = 0; bindingIdx < nbBindings; ++bindingIdx)
            {
                if (engine.bindingIsInput(bindingIdx))
                    continue;

                auto bufferSizesOutput = buffersSizes[bindingIdx];
                printOutput(bufferSizesOutput.first, bufferSizesOutput.second,
                            buffers[bindingIdx]);
            }
            CHECK(cudaFree(buffers[bindingIdxInput]));
        }

        total /= numberRun;
        std::cout << "Average over " << numberRun << " runs is " << total << " ms." << std::endl;
    }

    for (int bindingIdx = 0; bindingIdx < nbBindings; ++bindingIdx)
        if (!engine.bindingIsInput(bindingIdx))
            CHECK(cudaFree(buffers[bindingIdx]));
    context->destroy();
}

void* Evaluator::createPngCudaBuffer(int64_t eltCount, DataType dtype, int run)
{
    size_t memSize = eltCount * elementSize(dtype);
    float* inputs = new float[eltCount];

    float fileData[INPUT_H * INPUT_W * NUM_CHANNELS];
    //read float array from txt file and put it in buffer
    readTXTFile(image_file, fileData);

    //initialize the inputs buffer
    for (int i = 0; i < uni_buffer.size(); i++) {
        //normalizing values
        inputs[i] = uni_buffer.at(i);
    }

    //allocating memory for input to engine
    void* deviceMem = safeCudaMalloc(memSize);
    CHECK(cudaMemcpy(deviceMem, inputs, memSize, cudaMemcpyHostToDevice));

    delete[] inputs;
    return deviceMem;
}

void Evaluator::printOutput(int64_t eltCount, DataType dtype, void* buffer)
{
    std::cout << eltCount << " eltCount" << std::endl;
    std::cout << "--- OUTPUT ---" << std::endl;

    size_t memSize = eltCount * elementSize(dtype);
    float* outputs = new float[eltCount];
    CHECK(cudaMemcpy(outputs, buffer, memSize, cudaMemcpyDeviceToHost));

    int maxIdx = 0;
    for (int i = 0; i < eltCount; ++i){
        if (outputs[i] > outputs[maxIdx])
            maxIdx = i;
    }

    for (int64_t eltIdx = 0; eltIdx < eltCount; ++eltIdx)
    {
        std::cout << eltIdx << " => " << outputs[eltIdx] << "\t : ";
        if (eltIdx == maxIdx){
            std::cout << "***";
	    my_pred.second = outputs[eltIdx];
	    if (eltIdx == 0)
		my_pred.first = "car";
	    else
		my_pred.first = "no car";
	    }
        cout << "\n";
    }

    cout << endl;
    delete[] outputs;
}
