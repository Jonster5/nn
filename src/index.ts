import { NeuralNetwork } from './network';

const net = new NeuralNetwork([2, 3, 1]);

net.export('dsfsdf');

for (let i = 0; i < 100000; i++) {
    net.train([0, 0], [0]);
    net.train([1, 1], [0]);
}

console.log('trained');

net.export('dsfsdf');

console.log(net.predict([0]));
console.log(net.predict([1]));
// console.log(net.predict([0, 1]));
// console.log(net.predict([1, 1]));
