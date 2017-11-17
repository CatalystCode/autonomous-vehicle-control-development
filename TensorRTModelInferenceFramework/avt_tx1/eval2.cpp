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

static const int INPUT_H = 256;
static const int INPUT_W = 256;
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
    cout << "Calling safeCudaMalloc" << endl;
    CHECK(cudaMalloc(&deviceMem, memSize));
    if (deviceMem == nullptr)
    {
        std::cerr << "Out of memory" << std::endl;
        exit(1);
    }
    cout << "Did check" << endl;
    return deviceMem;
}

//Classifier class based on model files
class Classifier {
    public:
        Classifier();
        int classify();
        void getClass();
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
        ofstream outputclass;
        ofstream outputprob;
        const char * image_file = "";
	vector<float> uni_buffer;
        //vector<Prediction> predictions;
};

//this makes class invokable by Python
extern "C" {
    Classifier * Classifier_new() { return new Classifier(); }
    void Classifier_getclass(Classifier * classifier) { classifier->getClass(); }
    void Classifier_getprob(Classifier * classifier) { classifier->getProb(); }
    void Classifier_setimg(Classifier * classifier, const char * img) {
        classifier->setImg(img);
    }
    void Classifier_classify(Classifier * classifier) { classifier->classify(); }
}

//default constructor
Classifier::Classifier(void){
    //my_pred = *prediction_one;
    //predictions.push_back(*prediction_one);
    cout << "Classifier Default Constructor" << endl;
}

//returning the classification of the image
void Classifier::getClass() {
    outputclass.open("outputclass.txt");
    if (my_pred.first.compare("car")){
        outputclass << "Car";
    }
    else{
        outputclass << "No Car";
    }
    outputclass.close();
}

//returns the probability of the given classification
void Classifier::getProb(){
    cout << "C++ Prob: " << my_pred.second << endl;
    stringstream ss;
    ss << fixed << setprecision(4) << my_pred.second << endl;
    const char * mypred_str = ss.str().c_str();
    //cout << mypred_str << endl;
    
    outputprob.open("outputprob.txt");
    outputprob << mypred_str;
    outputprob.close();
}

const char * Classifier::setImg(const char * img_file){
    image_file = img_file;
    //cout << "image file: " << image_file << endl;
    return image_file;

}

void* Classifier::safeCudaMalloc(size_t memSize)
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
Classifier::calculateBindingBufferSizes(const ICudaEngine& engine, int nbBindings, int batchSize)
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

void Classifier::readTXTFile(const char * img_file, float buffer[INPUT_H*INPUT_W*NUM_CHANNELS]){
    string line;
    ifstream txtfile;
    int sz = INPUT_H*INPUT_W*NUM_CHANNELS;

    int i = 0;
    //cout << "readtxtfile called " << img_file << endl;

    uni_buffer.clear();
    txtfile.open(img_file,ios::in);

    if(!txtfile.is_open()){
	cout << "Unable to open file " << image_file << endl;
    }
    else{
	//adding values from python to universal buffer
	//cout << "Was able to open file" << endl;
	while ((txtfile >> line || !txtfile.eof()) && i < sz){
	    uni_buffer.push_back(stof(line));
            i++;
	}
	txtfile.close();
    }
}

int Classifier::classify(){
    //get the UFF file
    auto fileName = "/home/liz/upload/frozen.uff";

    int maxBatchSize = 1;
    auto parser = createUffParser();

    //Register tensorflow input
    parser->registerInput("x", DimsCHW(3, 135, 240));

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

ICudaEngine* Classifier::loadModelAndCreateEngine(const char* uffFile, int maxBatchSize,
                                      IUffParser* parser)
{
    //create builder and network needed to make engine
    IBuilder* builder = createInferBuilder(gLogger);
    INetworkDefinition* network = builder->createNetwork();

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

void Classifier::execute(ICudaEngine& engine)
{
    IExecutionContext* context = engine.createExecutionContext();
    int batchSize = 1;

    //gets input and output buffer pointers that we pass to the engine
    int nbBindings = engine.getNbBindings();
    cout << "nbBindings, should be 2?: " << endl;
    //assert(nbBindings == 2);

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

void* Classifier::createPngCudaBuffer(int64_t eltCount, DataType dtype, int run)
{
    size_t memSize = eltCount * elementSize(dtype);
    float* inputs = new float[eltCount];

    float fileData[INPUT_H * INPUT_W * NUM_CHANNELS];
    //read float array from txt file and put it in buffer
    readTXTFile(image_file, fileData);

    /* display the number in an ascii representation */
    //std::cout << "\n\n\n---------------------------" << "\n\n\n" << std::endl;
    //for (int i = 0; i < eltCount; i++)
        //td::cout << (" .:-=+*#%@"[fileData[i] / 26]) << (((i + 1) % INPUT_W) ? "" : "\n");

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

void Classifier::printOutput(int64_t eltCount, DataType dtype, void* buffer)
{
    std::cout << eltCount << " eltCount" << std::endl;
    //assert(elementSize(dtype) == sizeof(float));
    std::cout << "--- OUTPUT ---" << std::endl;

    size_t memSize = eltCount * elementSize(dtype);
    float* outputs = new float[eltCount];
    CHECK(cudaMemcpy(outputs, buffer, memSize, cudaMemcpyDeviceToHost));

    int maxIdx = 0;
    for (int i = 0; i < eltCount; ++i){
        if (outputs[i] > outputs[maxIdx])
            maxIdx = i;
	//cout << "output element " << i << ": "  << setprecision(5) <<  outputs[i] << endl;
    }

    for (int64_t eltIdx = 0; eltIdx < eltCount; ++eltIdx)
    {
        std::cout << eltIdx << " => " << outputs[eltIdx] << "\t : ";
        if (eltIdx == maxIdx){
            std::cout << "***";
            //cout << "output: " << outputs[eltIdx] << endl;
            //cout << "index: " << eltIdx << endl;
	    my_pred.second = outputs[eltIdx];
	    if (eltIdx == 0)
		my_pred.first = "car";
	    else
		my_pred.first = "no car";
	}
        std::cout << "\n";
    }

    std::cout << std::endl;
    delete[] outputs;
}
