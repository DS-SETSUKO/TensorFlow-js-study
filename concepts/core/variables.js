console.warn("VARIABLES: ");
// Variables

/**
 * Variables are initialized with a tensor of values.
 * The are mutables.
 * You can assign a new tensor to a existing variable.
 */

const initialValues = tf.zeros([5]);
const biases = tf.variable(initialValues);
biases.print();

const updatedValues = tf.tensor1d([0, 1, 0, 1, 1])
biases.assign(updatedValues)
biases.print();