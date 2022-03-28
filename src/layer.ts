import { Connection } from './connection';
import { Node } from './node';

export class Layer {
    nodes: Node[];

    constructor(nodes: number) {
        this.nodes = [];

        for (let i = 0; i < nodes; i++) {
            this.nodes.push(new Node());
        }
    }

    connect(left: Layer) {
        this.nodes.forEach((n1) => {
            left.nodes.forEach((n2) => {
                n1.connections.push(new Connection(n2, n1));
            });
        });
    }

    reset() {
        this.nodes.forEach((node) => node.reset());
    }
}
