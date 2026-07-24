import { readFile } from "node:fs/promises";
import { deserialize } from "node:v8";

const source = process.argv[2];
if (!source) throw new Error("Usage: node scripts/inspect-saved-model.mjs <v8-blob>");

const blob = await readFile(source);
const payload = blob[0] === 0xff && blob[2] === 0xfe ? blob.subarray(15) : blob;
const record = deserialize(payload);

const summary = {
  id: record.id,
  name: record.name,
  createdAt: record.createdAt,
  source: record.source,
  trainingMode: record.trainingMode,
  hiddenLayers: record.hiddenLayers,
  convolutionConfigs: record.convolutionConfigs,
  progress: record.progress,
  calibrated: record.model?.calibrated,
  trained: record.model?.trained,
  denseLayers: record.model?.layers?.map((layer) => ({
    inputSize: layer.inputSize,
    outputSize: layer.outputSize,
    activation: layer.activation,
    weights: layer.weights?.length,
    biases: layer.biases?.length,
  })),
  convolutions: record.model?.convolutions?.map((layer) => ({
    id: layer.id,
    position: layer.position,
    input: [layer.inputWidth, layer.inputHeight, layer.inputChannels],
    output: [layer.outputWidth, layer.outputHeight, layer.filters],
    kernelSize: layer.kernelSize,
    stride: layer.stride,
    padding: layer.padding,
    activation: layer.activation,
    weights: layer.weights?.length,
    biases: layer.biases?.length,
  })),
};

console.log(JSON.stringify(summary, null, 2));
