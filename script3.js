// Generate input numbers from 1 to 20 inclusive.

const INPUTS = [];

for (let n = 1; n <= 20; n++) {

    INPUTS.push(n);

}


// Generate Outputs that are simply each input multiplied by itself,

// to generate some non linear data.

const OUTPUTS = [];

for (let n = 0; n < INPUTS.length; n++) {

    OUTPUTS.push(INPUTS[n] * INPUTS[n]);

}


// Input feature Array is 1 dimensional.

const INPUTS_TENSOR = tf.tensor1d(INPUTS);


// Output can stay 1 dimensional.

const OUTPUTS_TENSOR = tf.tensor1d(OUTPUTS);

const LEARNING_RATE = 0.0001;
  
const OPTIMIZER = tf.train.sgd(LEARNING_RATE);


// Function to take a Tensor and normalize values
// with respect to each column of values contained in that Tensor.
function normalize(tensor, min, max) {
    const result = tf.tidy(function () {
        // Find the minimum value contained in the Tensor.
        const MIN_VALUES = min || tf.min(tensor, 0);

        // Find the maximum value contained in the Tensor.
        const MAX_VALUES = max || tf.max(tensor, 0);

        // Now calculate subtract the MIN_VALUE from every value in the Tensor
        // And store the results in a new Tensor.
        const TENSOR_SUBTRACT_MIN_VALUE = tf.sub(tensor, MIN_VALUES);

        // Calculate the range size of possible values.
        const RANGE_SIZE = tf.sub(MAX_VALUES, MIN_VALUES);

        // Calculate the adjusted values divided by the range size as a new Tensor.
        const NORMALIZED_VALUES = tf.div(TENSOR_SUBTRACT_MIN_VALUE, RANGE_SIZE);

        // Return the important tensors.
        return { NORMALIZED_VALUES, MIN_VALUES, MAX_VALUES };
    });
    return result;
}


// Normalize all input feature arrays and then dispose of the original non normalized Tensors.
const FEATURE_RESULTS = normalize(INPUTS_TENSOR);
console.log('Normalized Values:');
FEATURE_RESULTS.NORMALIZED_VALUES.print();

console.log('Min Values:');
FEATURE_RESULTS.MIN_VALUES.print();

console.log('Max Values:');
FEATURE_RESULTS.MAX_VALUES.print();

INPUTS_TENSOR.dispose();


// Now actually create and define model architecture.
const model = tf.sequential();

// We will use one dense layer with 1 neuron (units) and an input of 
// 2 input feaature values (representing house size and number of rooms).
model.add(tf.layers.dense({inputShape: [1], units: 100 , activation: 'relu'}));

model.add(tf.layers.dense({units: 100, activation: 'relu'}));

// Add another dense layer with 1 neuron that will be connected

// to the first input layer above.

model.add(tf.layers.dense({units: 1}));


model.summary();
train();


async function train() {

    // Choose a learning rate that is suitable for the data we are using.
  

  
  
    // Compile the model with the defined learning rate and specify
  
    // our loss function to use.
  
    model.compile({
  
      optimizer: OPTIMIZER,
  
      loss: 'meanSquaredError'
  
    });
  
  
    // Finally do the training itself 
  
    let results = await model.fit(FEATURE_RESULTS.NORMALIZED_VALUES, OUTPUTS_TENSOR, {
  
      callbacks: {onEpochEnd: logProgress},
  
      shuffle: true,  // Ensure data is shuffled in case it was in an order
  
      batchSize: 2,
  
      epochs: 200     // Go over the data 200 times!
  
    });
  
    
  
    OUTPUTS_TENSOR.dispose();
  
    FEATURE_RESULTS.NORMALIZED_VALUES.dispose();
  
  
    console.log("Average error loss: " + Math.sqrt(results.history.loss[results.history.loss.length - 1]));

    // Once trained we can evaluate the model.
    evaluate();
}



function logProgress(epoch, logs) {

    console.log('Data for epoch ' + epoch, Math.sqrt(logs.loss));

    if (epoch == 70) {

        OPTIMIZER.setLearningRate(LEARNING_RATE / 2);
    
      }
  
  }


function evaluate() {

    // Predict answer for a single piece of data.

    tf.tidy(function () {

        let newInput = normalize(tf.tensor1d([7]), FEATURE_RESULTS.MIN_VALUES, FEATURE_RESULTS.MAX_VALUES);


        let output = model.predict(newInput.NORMALIZED_VALUES);

        output.print();

    });



    // Finally when you no longer need to make any more predictions,

    // clean up the remaining Tensors. 

    FEATURE_RESULTS.MIN_VALUES.dispose();

    FEATURE_RESULTS.MAX_VALUES.dispose();

    model.dispose();



    console.log(tf.memory().numTensors);

}