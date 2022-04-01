import * as math from 'mathjs';
import { matrix, Matrix } from 'mathjs';

export class NeuralNetwork {
    inputNodes: Matrix;
    hiddenNodes: Matrix;
    outputNodes: Matrix;
    weightsIH: Matrix;
    weightsHO: Matrix;
    biasH: Matrix;
    biasO: Matrix;
    activation: ((x: number) => number)[];
    lRate: number;

    constructor(
        model: [number, number, number],
        activation: Function,
        learningRate?: number
    ) {
        this.inputNodes = math.map(math.zeros(model[0]), Math.random) as Matrix;
        this.hiddenNodes = math.map(
            math.zeros(model[0]),
            Math.random
        ) as Matrix;
        this.outputNodes = math.map(
            math.zeros(model[0]),
            Math.random
        ) as Matrix;

        this.weightsIH = math.map(
            math.zeros([model[1], model[0]]),
            Math.random
        ) as Matrix;
        this.weightsHO = math.map(
            math.zeros([model[2], model[1]]),
            Math.random
        ) as Matrix;
        this.biasH = math.map(math.zeros(model[1]), Math.random) as Matrix;
        this.biasO = math.map(math.zeros(model[2]), Math.random) as Matrix;

        this.activation = activation();
        this.lRate = learningRate ?? 0.1;
    }

    predict(inputArr: number[]) {
        let input = matrix(inputArr);

        let hidden = math.multiply(this.weightsIH, input);
        hidden = math.add(hidden, this.biasH);
        hidden = math.map(hidden, this.activation[0]);

        let output = math.multiply(this.weightsHO, hidden);
        output = math.add(output, this.biasO);
        output = math.map(output, this.activation[0]);

        return output.toArray();
    }

    train(inputArr: number[], targetArr: number[]) {
        let input = matrix(inputArr);
        let target = matrix(targetArr);

        // feed forward
        let hidden = math.multiply(this.weightsIH, input);
        hidden = math.add(hidden, this.biasH);
        hidden = math.map(hidden, this.activation[0]);

        let output = math.multiply(this.weightsHO, hidden);
        output = math.add(output, this.biasO);
        output = math.map(output, this.activation[0]);

        // Calculate the error
        let outputError = math.subtract(target, output);

        // Calculate gradient
        let outputGradient = math.map(output, this.activation[1]);
        outputGradient = math.multiply(outputGradient, outputError);
        outputGradient = math.multiply(outputGradient, this.lRate);

        // Calculate deltas
        let weightHODelta = math.multiply(outputGradient, math.transpose(hidden));
        this.weightsHO = math.add(this.weightsHO, weightHODelta);
        this.biasO = math.add(this.biasO, outputGradient);

        // Calculate the hidden layer errors
        let hiddenError = math.multiply(math.transpose(this.weightsHO), outputError);

        // Calculate hidden gradient
        let hiddenGradient = math.map(hidden, this.activation[1]);
        hiddenGradient = math.multiply(hiddenGradient, hiddenError);
        hiddenGradient = math.multiply(hiddenGradient, this.lRate);

        // Calcuate input->hidden deltas
        let weightIHDelta = math.multiply(hiddenGradient, math.transpose(input));
        this.weightsIH = math.add(this.weightsIH, weightIHDelta);
        this.biasH = math.add(this.biasH, hiddenGradient);
    }

    // serialize() {
    //     return JSON.stringify(this);
    // }

    // static deserialize(data) {
    //     if (typeof data == 'string') {
    //         data = JSON.parse(data);
    //     }
    //     let nn = new NeuralNetwork(
    //         data.input_nodes,
    //         data.hidden_nodes,
    //         data.output_nodes
    //     );
    //     nn.weights_ih = Matrix.deserialize(data.weights_ih);
    //     nn.weights_ho = Matrix.deserialize(data.weights_ho);
    //     nn.bias_h = Matrix.deserialize(data.bias_h);
    //     nn.bias_o = Matrix.deserialize(data.bias_o);
    //     nn.learning_rate = data.learning_rate;
    //     return nn;
    // }
}
