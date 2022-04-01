import { Matrix } from "./matrix";

export class NeuralNetwork {
    inputNodes: Matrix;
    hiddenNodes: Matrix;
    outputNodes: Matrix;
    weightsIH: Matrix;
    weightsHO: Matrix;
    biasH: Matrix;
    biasO: Matrix;
    a: (x: number) => number;
    aPrime: (x: number) => number;
    lRate: number;

    constructor(
        model: [number, number, number],
        activation: Function,
        learningRate?: number
    ) {
        this.inputNodes = new Matrix(model[0], 1).randomize();
        this.hiddenNodes = new Matrix(model[1], 1).randomize();
        this.outputNodes = new Matrix(model[2], 1).randomize();

        this.weightsIH = new Matrix(model[1], model[0]).randomize();
        this.weightsHO = new Matrix(model[2], model[1]).randomize();

        this.biasH = new Matrix(model[1], 1).randomize();
        this.biasO = new Matrix(model[2], 1).randomize();

        this.a = activation()[0];
        this.aPrime = activation()[1];
        this.lRate = learningRate ?? 0.1;
    }

    predict(inputArr: number[]) {
        let input = Matrix.fromArray(inputArr);

        let hidden = this.weightsIH
            .clone()
            .multiply(input)
            .add(this.biasH)
            .map(this.a);

        let output = this.weightsHO
            .clone()
            .multiply(hidden)
            .add(this.biasO)
            .map(this.a);

        return output.toArray();
    }

    train(inputArr: number[], targetArr: number[]) {
        const input = Matrix.fromArray(inputArr);
        const target = Matrix.fromArray(targetArr);

        // feed forward
        const hidden = this.weightsIH
            .clone()
            .multiply(input)
            .add(this.biasH)
            .map(this.a);

        const output = this.weightsHO
            .clone()
            .multiply(hidden)
            .add(this.biasO)
            .map(this.a);

        // Calculate the error
        const outputError = target.clone().subtract(output);

        // Calculate gradient
        const outputGradient = output
            .clone()
            .map(this.aPrime)
            .multiply(outputError)
            .multiply(this.lRate);

        // Calculate deltas
        const weightsHODelta = outputGradient
            .clone()
            .multiply(hidden)

        // Apply deltas
        this.weightsHO.add(weightsHODelta);
        this.biasO.add(outputGradient);

        // Calculate the hidden layer errors
        const hiddenError = this.weightsHO.clone().multiply(outputError);

        // Calculate hidden gradient
        const hiddenGradient = hidden.clone().map(this.aPrime).multiply(hiddenError).multiply(this.lRate);

        // Calcuate deltas
        const weightsIHDelta = hiddenGradient.clone().multiply(input);

        // Apply deltas
        this.weightsIH.add(weightsIHDelta);
        this.biasH.add(hiddenGradient);
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
