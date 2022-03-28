import { Node } from './node';

export class Connection {
    left: Node;
    right: Node;
    value: number;

    error: number;
    pDelta: number;

    constructor(left: Node, right: Node) {
        this.left = left;
        this.right = right;

        this.error = 0;

        this.value = Math.random();
        this.pDelta = 0;
    }
}
