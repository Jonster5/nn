import { Connection } from './connection';

export class Node {
    connections: Connection[];

    value: number;
    pDelta: number;

    output: number;
    delta: number;
    gradient: number;
    error: number;

    constructor() {
        this.connections = [];

        this.value = Math.random();
        this.pDelta = 0;

        this.output = 0;
        this.delta = 0;
        this.gradient = 0;
        this.error = 0;
    }

    reset() {
        this.output = 0;
        this.delta = 0;
        this.gradient = 0;
    }
}
