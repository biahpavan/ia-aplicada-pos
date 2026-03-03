import tf from '@tensorflow/tfjs-node'


async function trainModel(inputXs, outputYs) {
    // create a new sequential model by *calling* the factory function
    const model = tf.sequential()

    // InputShape
    // First: Hidden Layer
    // Input with 7 features (normalized age + 3 colors + 3 locations)
    // Units
    // 80 neurons because the training dataset is small. 
    // The more neurons you add, the more complex the network becomes and the more it can learn. 
    // But that also means it will need more processing power
    // Activation: relu 
    // RELU works like a filter, allowing only the 'interesting' data to pass through the network. 
    // If the information that reaches this neuron is positive, it moves forward; 
    // if it is zero or negative, the information is discarded.
    model.add(tf.layers.dense({ inputShape: [7], units: 80, activation: 'relu' }))

    // Output: 3 neurons
    // One for each category(premium, medium and basic)
    // Activation: softmax
    // Normalizes the output into a probability.
    model.add(tf.layers.dense({ units: 3, activation: 'softmax' }))

    // Compiling the Model
    // Optimizer: Adam (Adaptive Moment Estimation)
    // Adam as a personal trainer for the network: 
    // it adjusts the weights step by step, learning from mistakes and successes to improve performance

    // Loss: Categorical Crossentropy
    // It compares what the model predicts (the scores for each category) with the correct answer.
    // Example: the "premium" category will always be [1, 0, 0]

    // Metrics: Accuracy
    // The farther the model’s prediction is from the correct answer, the greater the error (loss).
    // Classical examples: image classification, recommendations, user categorization
    // Any task where the correct answer is only one among several possibilities.
    model.compile({
        optimizer: 'adam',
        loss: 'categoricalCrossentropy',
        metrics: ['accuracy']
    })

    // Model Training
    // verbose: disables the internal log (and uses only the callback)
    // epochs: number of times the dataset will be run
    // shuffle: shuffles the data to avoid bias
    await model.fit(
        inputXs,
        outputYs,
        {
            verbose: 0,
            epochs: 100,
            shuffle: true,
            callbacks: {
                onEpochEnd: (epoch, log) => console.log(`Epoch: ${epoch}: loss = ${log.loss}`)
            }
        }
    )

    return model
}

async function predict(model, pessoa) {
    // Convert the JavaScript array into a tensor (tfjs)
    const tfInput = tf.tensor2d(pessoa)

    // Make the prediction (the output will be a vector of 3 probabilities)
    const pred = model.predict(tfInput)
    const predArray = await pred.array()
    return predArray[0].map((prob, index) => ({ prob, index }))
}

// Example of people for training (each person with age, color, and location)
// const people = [
//     { name: "Erick", age: 30, color: "blue", location: "São Paulo" },
//     { name: "Ana", age: 25, color: "red", location: "Rio" },
//     { name: "Carlos", age: 40, color: "green", location: "Curitiba" }
// ];

// Input vectors with values already normalized and one-hot encoded
// Order: [normalized_age, blue, red, green, São Paulo, Rio, Curitiba]
// const tensorPeople = [
//     [0.33, 1, 0, 0, 1, 0, 0], // Erick
//     [0, 0, 1, 0, 0, 1, 0],    // Ana
//     [1, 0, 0, 1, 0, 0, 1]     // Carlos
// ]

// We only use numerical data, since the neural network only understands numbers.
// tensorPeopleNormalized corresponds to the input dataset for the model.
const tensorPeopleNormalized = [
    [0.33, 1, 0, 0, 1, 0, 0], // Erick
    [0, 0, 1, 0, 0, 1, 0],    // Ana
    [1, 0, 0, 1, 0, 0, 1]     // Carlos
]

// Labels of the categories to be predicted (one-hot encoded)
// [premium, medium, basic]
const labelsNames = ["premium", "medium", "basic"]; // Order of labels
const tensorLabels = [
    [1, 0, 0], // premium - Erick
    [0, 1, 0], // medium - Ana
    [0, 0, 1]  // basic - Carlos
];

// We create input tensors (xs) and output tensors (ys) to train the model
const inputXs = tf.tensor2d(tensorPeopleNormalized)
const outputYs = tf.tensor2d(tensorLabels)

const model = await trainModel(inputXs, outputYs);

// Example prediction
// const person = { name: "Beatriz", age: 28, color: "green", location: "Curitiba" }

// Normalizing the age of the new person using the same training pattern
// Example: min_age = 25, max_age = 40, so (28 - 25) / (40 - 25) = 0.2

const personTensorNormalized = [
    [
        0.2, // normalized age
        1,   // color blue
        0,   // color red
        0,   // color green
        0,   // location São Paulo
        1,   // location Rio
        0    // location Curitiba
    ]
]

const predictions = await predict(model, personTensorNormalized)
const results = predictions
    .sort((a, b) => b.prob - a.prob)
    .map(p => `${labelsNames[p.index]} (${(p.prob * 100).toFixed(2)}%)`)
    .join('\n')
console.log(results)