import { NeuralNetwork } from './network';

const net = new NeuralNetwork([2, 3, 1]);




console.log(net.predict([0, 1]));
console.log(net.predict([1, 1]));

net.export('output/export.json');
