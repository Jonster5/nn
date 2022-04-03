import { Matrix } from './matrix';
import { NeuralNetwork } from './network';
import { Trainer } from './trainer';

const net = NeuralNetwork.import('output/knight.net.json');
// const net = new NeuralNetwork(
//     [
//         { size: 64, activation: 'sigmoid' },
//         { size: 48, activation: 'sigmoid' },
//         { size: 48, activation: 'sigmoid' },
//         { size: 16, activation: 'sigmoid' },
//         { size: 8, activation: 'sigmoid' },
//     ],
//     0.1
// );

const trainer = new Trainer(net, 'training.json', 0.001, 5000);

trainer.start({ verbose: true, save: 'output/knight.net.json' });
