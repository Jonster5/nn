import * as math from 'mathjs';
import { matrix, Matrix } from 'mathjs';
import { sigmoid } from './activation';
import { NeuralNetwork } from './network';

const net = new NeuralNetwork([2, 3, 1], sigmoid);
