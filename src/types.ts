export type ActivationKind =
  | "relu"
  | "leakyRelu"
  | "elu"
  | "selu"
  | "relu6"
  | "gelu"
  | "swish"
  | "mish"
  | "sigmoid"
  | "tanh"
  | "hardSigmoid"
  | "hardTanh"
  | "softplus"
  | "softsign"
  | "softmax"
  | "linear";

export interface HiddenLayer {
  id: string;
  units: number;
  activation: ActivationKind;
  dropout: number;
}

export interface ConvolutionConfig {
  id: string;
  enabled: boolean;
  position: number;
  filters: number;
  kernelSize: number;
  stride: number;
  padding: number;
  activation: ActivationKind;
  kernels: number[][];
}

export type DatasetSplit = "training" | "test";

export interface CustomDatasetSample {
  id: string;
  label: number;
  split: DatasetSplit;
  indices: number[];
  values: number[];
}

export interface DenseLayerData {
  inputSize: number;
  outputSize: number;
  activation: ActivationKind | "linear";
  weights: Float32Array;
  biases: Float32Array;
}

export interface ConvolutionLayerData {
  id: string;
  position: number;
  inputWidth: number;
  inputHeight: number;
  inputChannels: number;
  outputWidth: number;
  outputHeight: number;
  filters: number;
  kernelSize: number;
  stride: number;
  padding: number;
  activation: ActivationKind;
  weights: Float32Array;
  biases: Float32Array;
}

export interface NeuralModel {
  convolutions: ConvolutionLayerData[];
  /** Legacy single-layer field kept while IndexedDB records are migrated. */
  convolution?: ConvolutionLayerData | null;
  layers: DenseLayerData[];
  calibrated: boolean;
  trained?: boolean;
}

export interface InferenceResult {
  probabilities: number[];
  activations: number[][];
  latencyMs: number;
  backend: "Wasm SIMD" | "Wasm" | "JavaScript";
}

export interface SerializedModel {
  architecture: number[];
  weights: number[][];
  biases: number[][];
  samples: number[][];
  accuracy: number;
}

export interface TrainingSettings {
  epochs: number;
  learningRate: number;
  mathMode: MathMode;
  optimizer: OptimizerConfig;
}

export interface TrainingProfileSettings {
  epochs: number;
  learningRate: number;
  optimizer: OptimizerConfig;
}

export interface TrainingProfiles {
  mathMode: MathMode;
  scratch: TrainingProfileSettings;
  finetune: TrainingProfileSettings;
}

export type MathMode = "fast" | "full";

export type OptimizerKind = "sgd" | "momentum" | "adam" | "rmsprop" | "adagrad";

export interface OptimizerConfig {
  kind: OptimizerKind;
  momentum: number;
  beta1: number;
  beta2: number;
  decay: number;
  epsilon: number;
}

export type PropagationDirection = "forward" | "backward";

export type TrainingMode = "scratch" | "finetune";

export interface TrainingProgress {
  phase: "idle" | "loading" | "training" | "paused" | "complete" | "cancelled" | "error";
  epoch: number;
  epochs: number;
  accuracy: number;
  loss: number;
  elapsedMs: number;
  message?: string;
}

export interface TrainingTrace {
  activations: number[][];
  gradients: number[][];
  epoch: number;
  sample: number;
  samples: number;
  label: number;
  prediction: number;
  loss: number;
}

export type SavedModelSource = "complete" | "paused";

export interface SavedModel {
  id: string;
  name: string;
  createdAt: number;
  source: SavedModelSource;
  trainingMode?: TrainingMode;
  hiddenLayers: HiddenLayer[];
  convolutionConfigs?: ConvolutionConfig[];
  /** Legacy single-layer field kept while IndexedDB records are migrated. */
  convolutionConfig?: ConvolutionConfig | null;
  model: NeuralModel;
  progress: {
    epoch: number;
    epochs: number;
    accuracy: number;
    loss: number;
    elapsedMs: number;
  };
}

export type ModelStatus = "calibrated" | "trained" | "experimental";
