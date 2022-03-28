export class Tour {
    constructor() {
        const board = [
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
        ];

        this.startBoard(board);

        console.log('Move: 1');
        this.showBoard(board);

        for (let i = 0; i < 64; i++) this.nextMove(board);
    }

    startBoard(brd: number[][]) {
        brd[0][0] = 1;
    }

    showBoard(brd: number[][]) {
        for (let row of brd) {
            const t = row
                .map((x) => (x < 10 ? `0${x}` : `${x}`))
                .reduce((a, b) => a + b + ' ', '');
            console.log(t);
        }
        console.log();
    }

    checkVisit(brd: number[][], r: number, c: number) {
        return (
            r >= 0 ||
            c >= 0 ||
            r <= brd.length ||
            c <= brd[0].length ||
            brd[r][c] !== 0
        );
    }

    nextMove(brd: number[][]) {
        const n = brd.reduce(
            (a, b) =>
                a < b.reduce((c, d) => (c < d ? d : c))
                    ? b.reduce((c, d) => (c < d ? d : c))
                    : a,
            0
        );

        let r = 0,
            c = 0;

        const xm = [2, 1, -1, -2, -2, -1, 1, 2];
        const ym = [1, 2, 2, 1, -1, -2, -2, -1];

        brd.forEach((row, i) => {
            row.forEach((col, j) => {
                if (col !== n) return;
                r = i;
                c = j;
            });
        });
    }
}
