console.warn("MODELS: ");
// Models

/*
* First approach:
* 
* You can create a model using ops directly to represent the work of the model does.
*
*/

function predict(input) {
    // y = a * x ^ 2 + b * x + c
    
    return tf.tidy(() => {
        const x = tf.scalar(input);

        const ax2 = a.mul(x.square());
        const bx = b.mul(x);
        const y = ax2.add(bx).add(c)

        return y;
    });
}

// Define constants: y = 2x^2 + 4x +8
const a = tf.scalar(2);
const b = tf.scalar(4);
const c = tf.scalar(8);

const result = predict(2);
result.print();


/*
* Second approach:
* 
* You can use the high-level API tf.model() to construct a model out of layers.
*
*/

const data = [[2,3,54, 23], [1,3,4,5]];
const labels = [[1,0,1,0]];

// Init a sequential model.
const model = tf.sequential();

// Create a simple rnn layer.
const layer1 = tf.layers.simpleRNN({
    units: 1,
    recurrentInitializer: "GlorotNormal",
    inputShape: [2, 4]
});

model.add(layer1);

const optimizer = tf.train.sgd(0.001);

// model.compile({optimizer, loss: "categoricalCrossentropy"})
// model.fit({x: data, y: labels})


