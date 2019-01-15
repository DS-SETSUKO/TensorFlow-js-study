console.warn("Polynomial regression example");

function generateData(numPoints, coeff, sigma = 0.04) {
    return tf.tidy(() => {
        const [a, b, c, d] = [
            tf.scalar(coeff.a), tf.scalar(coeff.b), tf.scalar(coeff.c),
            tf.scalar(coeff.d)
        ];

        const xs = tf.randomUniform([numPoints], -1, 1);

        // Generate polynomial data
        const three = tf.scalar(3, 'int32');
        const ys = a.mul(xs.pow(three))
            .add(b.mul(xs.square()))
            .add(c.mul(xs))
            .add(d)
            // Add random noise to the generated data
            // to make the problem a bit more interesting
            .add(tf.randomNormal([numPoints], 0, sigma));

        // Normalize the y values to the range 0 to 1.
        const ymin = ys.min();
        const ymax = ys.max();
        const yrange = ymax.sub(ymin);
        const ysNormalized = ys.sub(ymin).div(yrange);

        return {
            xs,
            ys: ysNormalized
        };
    })
}


const trueCoefficients = {a: -.8, b: -.2, c: .9, d: .5};
let data = generateData(100, trueCoefficients);

data["xs"].print();
data["ys"].print();

const a = tf.variable(tf.scalar(Math.random()));
const b = tf.variable(tf.scalar(Math.random()));
const c = tf.variable(tf.scalar(Math.random()));
const d = tf.variable(tf.scalar(Math.random()));

function predict(x) {
    // y = a * x^3 + b * x^2 + c * x + d
    return tf.tidy(() => {
        return a.mul(x.pow(tf.scalar(3)))
            .add(b.mul(x.square()))
            .add(c.mul(x))
            .add(c)
    })
}

function loss(predictions, labels) {
    const meanSquareError = predictions.sub(labels).square().mean();
    console.log("Loss %: ", meanSquareError.print())
    return meanSquareError;
}

function train(xs, ys, numInterations) {
    const optimizer = tf.train.sgd(0.1);
    
    for (let iter = 0; iter < numInterations; iter++) {
        optimizer.minimize(() => {
            const predsYs = predict(xs);
            return loss(predsYs, ys);
        })
    }
}

// train(data["xs"], data["ys"], 100);