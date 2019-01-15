console.warn("MEMORY MANAGEMENT: ");
/**
 * Tensorflow uses GPU to accelerate math operation, 
 * for that reason we need to manage the gpu memory
 * when working with tensors and variables.
 */

/**
 * Dispose:
 * you can use it on a tensor or variable to purge it
 * and free up its gpu memory
 */

 const tensor_x = tf.tensor2d([[0.0, 2.0], [2.0, 3.0]]);
 const tensor_x_squared = tensor_x.square();

 tensor_x.dispose();
 tensor_x_squared.dispose();


 /**
  * tf.tidy:
  * you usually have a lot of tensor operations it would be tedious
  * to dispose each one. Tidy executes a function and purges any intermediate
  * tensors created. It does not purge the return value of the inner function.
  * 
  * - Varibles will not be affected, use dispose.
  */

  const average = tf.tidy(() => {

    const y = tf.tensor1d([1.0, 2.0, 3.0, 4.0]);
    const z = tf.ones([4]);

    return y.sub(z).square().mean();
  })

  console.log("tf.tidy: ")
  average.print();

