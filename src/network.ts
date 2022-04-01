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

        let hidden = math.multiply(this.weightsIH, input);
        hidden = math.add(hidden, this.biasH);
        hidden = math.map(hidden, this.activation[0]);

        let output = math.multiply(this.weightsHO, hidden);
        output = math.add(output, this.biasO);
        output = math.map(output, this.activation[0]);

        //     // Convert array to matrix object
        //     let targets = .Matrix.fromArray(target);
        //     // Calculate the error
        //     // ERROR = TARGETS - OUTPUTS
        //     let output_errors = .Matrix.subtract(targets, outputs);
        //     // let gradient = outputs * (1 - outputs);
        //     // Calculate gradient
        //     let gradients = .Matrix.map(
        //         outputs,
        //         this.activation_function.dfunc
        //     );
        //     gradients.multiply(output_errors);
        //     gradients.multiply(this.learning_rate);
        //     // Calculate deltas
        //     let hidden_T = .Matrix.transpose(hidden);
        //     let weight_ho_deltas = .Matrix.multiply(gradients, hidden_T);
        //     // Adjust the weights by deltas
        //     this.weights_ho.add(weight_ho_deltas);
        //     // Adjust the bias by its deltas (which is just the gradients)
        //     this.bias_o.add(gradients);
        //     // Calculate the hidden layer errors
        //     let who_t = .Matrix.transpose(this.weights_ho);
        //     let hidden_errors = .Matrix.multiply(who_t, output_errors);
        //     // Calculate hidden gradient
        //     let hidden_gradient = .Matrix.map(
        //         hidden,
        //         this.activation_function.dfunc
        //     );
        //     hidden_gradient.multiply(hidden_errors);
        //     hidden_gradient.multiply(this.learning_rate);
        //     // Calcuate input->hidden deltas
        //     let inputs_T = .Matrix.transpose(inputs);
        //     let weight_ih_deltas = .Matrix.multiply(
        //         hidden_gradient,
        //         inputs_T
        //     );
        //     this.weights_ih.add(weight_ih_deltas);
        //     this.bias_h.add(hidden_gradient);
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
