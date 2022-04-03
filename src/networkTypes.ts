import { Matrix } from './matrix';

export interface NeuralNetModel {}

export interface Layer {
    size: number;
    activation: (x: number, derivative?: boolean) => number;
    aName: ActivationFunctionName;

    weights: Matrix;
    bias: Matrix;

    output: Matrix;

    error: Matrix;
    gradient: Matrix;
}

export type ActivationFunctionName =
    | 'sigmoid'
    | 'tanh'
    | 'relu'
    | 'leakyRelu'
    | 'binaryStep';
