/**
 * Copyright (c) 2018, NVIDIA CORPORATION. All rights reserved.
 * Full license terms provided in LICENSE.md file.
 */

#include <fstream>
#include <iostream>
#include <sstream>
#include <string>

#include <NvInfer.h>
#include <NvUffParser.h>

using namespace std;
using namespace nvinfer1;
using namespace nvuffparser;

class Logger : public ILogger {
  void log(Severity severity, const char *msg) override {
    switch (severity) {
    case Severity::kINTERNAL_ERROR:
      cout << "kINTERNAL_ERROR: " << msg << endl;
      break;
    case Severity::kERROR:
      cout << "kERROR: " << msg << endl;
      break;
    case Severity::kWARNING:
      cout << "kWARNING: " << msg << endl;
      break;
    case Severity::kINFO:
      cout << "kINFO: " << msg << endl;
      break;
    default:
      cout << "kUNKNOWN: " << msg << endl;
    }
  }
} gLogger;

int toInteger(string value) {
  int valueInteger;
  stringstream ss;
  ss << value;
  ss >> valueInteger;
  return valueInteger;
}

DataType toDataType(string value) {
  if (value == "float")
    return DataType::kFLOAT;
  else if (value == "half")
    return DataType::kHALF;
  else
    throw runtime_error("Unsupported data type");
}

int main(int argc, char *argv[]) {
  if (argc != 11) {
    cout << "Need 11 arguments, but get " << argc << " only!\n";
    cout << "Usage: <uff_filename> <plan_filename> <input_name> <output_name> "
            "<input channel>  <input_height> <input_width> <max_batch_size> "
            "<max_workspace_size> <data_type>\n";
    return 1;
  }

  /* parse command line arguments */
  string uffFilename = argv[1];
  string planFilename = argv[2];
  string inputName = argv[3];
  string outputName = argv[4];
  int inputChannel = toInteger(argv[5]);
  int inputHeight = toInteger(argv[6]);
  int inputWidth = toInteger(argv[7]);
  int maxBatchSize = toInteger(argv[8]);
  int maxWorkspaceSize = toInteger(argv[9]);
  DataType dataType = toDataType(argv[10]);

  /* parse uff */
  IBuilder *builder = createInferBuilder(gLogger);
  INetworkDefinition *network = builder->createNetwork();
  IUffParser *parser = createUffParser();
  parser->registerInput(inputName.c_str(),
                        DimsCHW(inputChannel, inputHeight, inputWidth));
  parser->registerOutput(outputName.c_str());
  if (!parser->parse(uffFilename.c_str(), *network, dataType)) {
    cout << "Failed to parse UFF\n";
    builder->destroy();
    parser->destroy();
    network->destroy();
    return 1;
  }

  /* build engine */
  if (dataType == DataType::kHALF)
    builder->setHalf2Mode(true);

  builder->setMaxBatchSize(maxBatchSize);
  builder->setMaxWorkspaceSize(maxWorkspaceSize);
  ICudaEngine *engine = builder->buildCudaEngine(*network);

  /* serialize engine and write to file */
  cout << "Write PLAN file: " << planFilename << endl;
  ofstream planFile;
  planFile.open(planFilename);
  IHostMemory *serializedEngine = engine->serialize();
  planFile.write((char *)serializedEngine->data(), serializedEngine->size());
  planFile.close();

  /* break down */
  builder->destroy();
  parser->destroy();
  network->destroy();
  engine->destroy();
  serializedEngine->destroy();

  return 0;
}
