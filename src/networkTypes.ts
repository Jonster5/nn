import { Matrix } from './matrix';

export interface NeuralNetModel {}

export interface Layer {
    size: number;
    act: (x: number) => number;
    der: (x: number) => number;

    weights: Matrix;
    bias: Matrix;

    output: Matrix;

    error: Matrix;
    gradient: Matrix;
}
