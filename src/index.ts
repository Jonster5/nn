import fs from 'fs';
import { KnightsTour } from './KnightsTour';
import { Matrix } from './matrix';
import { NeuralNetwork } from './network';
import { Trainer } from './trainer';

// const tour = new KnightsTour();

// for (let i = 0; i < 63; i++) {
//     tour.nextMove();
// }

// const net = NeuralNetwork.import('output/knight.net.json');

// neural network large enough to handle the knight's tour problem
// 64 inputs, 8 outputs
// has to be able to correctly predict the next move based on the state of the board
// const net = new NeuralNetwork([
//     { size: 64 },
//     { size: 48, activation: 'leakyRelu' },
//     { size: 48, activation: 'leakyRelu' },
//     { size: 48, activation: 'leakyRelu' },
//     { size: 8, activation: 'sigmoid' },
// ]);

// const net = new NeuralNetwork([
//     { size: 2 },
//     { size: 3, activation: 'relu' },
//     { size: 1, activation: 'relu' },
// ]);

// const net = NeuralNetwork.import('output/xor.net.json');

// // training set for xor
// const set = [
//     {
//         input: [0, 0],
//         target: [0],
//     },
//     {
//         input: [0, 1],
//         target: [1],
//     },
//     {
//         input: [1, 0],
//         target: [1],
//     },
//     {
//         input: [1, 1],
//         target: [0],
//     },
// ];

// set.forEach((s) => {
//     const output = net.predict(s.input);
//     console.log(`${s.input} -> ${output}`);
// });

const net = NeuralNetwork.import('output/evenOrOdd.nn.json');
// const net = new NeuralNetwork([
//     { size: 1 },
//     { size: 3, activation: 'sigmoid' },
//     { size: 3, activation: 'sigmoid' },
//     { size: 1, activation: 'sigmoid' },
// ]);

net.export('output/evenOrOdd.nn.json');
const trainer = new Trainer(net, 'input/evenOrOdd.ds.json', 0.01, 10000);

trainer.demonstrate(4);

trainer.start({
    save: 'output/evenOrOdd.nn.json',
    verbose: true,
    maxCycles: 100000,
});

// trainer.demonstrate(4);

/* Naming Conventions:
 * - all files storing neural networks end in .nn.json
 * - all files storing datasets end in .ds.json
 * - all files storing training sessions end in .ts.json
 */
