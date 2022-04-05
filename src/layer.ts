import { Activation } from './activation';
import { Matrix } from './matrix';
import { ActivationFunctionName } from './networkTypes';

export class Layer {
    size: number;
    activation: (x: number, derivative?: boolean) => number;
    aName: ActivationFunctionName;

    weights: Matrix;
    bias: Matrix;

    output: Matrix;

    error: Matrix;
    gradient: Matrix;

    pGradient: Matrix;

    constructor(
        size: number,
        pSize: number,
        activation: ActivationFunctionName
    ) {
        this.size = size;
        this.activation = Activation[activation];
        this.aName = activation;

        this.weights = new Matrix(size, pSize).randomize();
        this.bias = new Matrix(size, 1).randomize();

        this.output = new Matrix(size, 1);

        this.error = new Matrix(size, 1);
        this.gradient = new Matrix(size, 1);

        this.pGradient = new Matrix(size, 1);
    }
}
