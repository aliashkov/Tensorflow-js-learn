
const STATUS = document.getElementById('status');

const VIDEO = document.getElementById('webcam');

const ENABLE_CAM_BUTTON = document.getElementById('enableCam');

const RESET_BUTTON = document.getElementById('reset');

const TRAIN_BUTTON = document.getElementById('train');

const MOBILE_NET_INPUT_WIDTH = 224;

const MOBILE_NET_INPUT_HEIGHT = 224;

const STOP_DATA_GATHER = -1;

const CLASS_NAMES = [];


ENABLE_CAM_BUTTON.addEventListener('click', enableCam);

//TRAIN_BUTTON.addEventListener('click', trainAndPredict);

RESET_BUTTON.addEventListener('click', reset);



let dataCollectorButtons = document.querySelectorAll('button.dataCollector');

/* for (let i = 0; i < dataCollectorButtons.length; i++) {

  dataCollectorButtons[i].addEventListener('mousedown', gatherDataForClass);

  dataCollectorButtons[i].addEventListener('mouseup', gatherDataForClass);

  // Populate the human readable names for classes.

  CLASS_NAMES.push(dataCollectorButtons[i].getAttribute('data-name'));

} */


let mobilenet = undefined;

let gatherDataState = STOP_DATA_GATHER;

let videoPlaying = false;

let trainingDataInputs = [];

let trainingDataOutputs = [];

let examplesCount = [];

let predict = false;


/**

 * Loads the MobileNet model and warms it up so ready for use.

 **/

async function loadMobileNetFeatureModel() {

    const URL =

        'https://tfhub.dev/google/tfjs-model/imagenet/mobilenet_v3_small_100_224/feature_vector/5/default/1';



    mobilenet = await tf.loadGraphModel(URL, { fromTFHub: true });

    STATUS.innerText = 'MobileNet v3 loaded successfully!';



    // Warm up the model by passing zeros through it once.

    tf.tidy(function () {

        let answer = mobilenet.predict(tf.zeros([1, MOBILE_NET_INPUT_HEIGHT, MOBILE_NET_INPUT_WIDTH, 3]));

        console.log(answer.shape);

    });

    let model = tf.sequential();

    model.add(tf.layers.dense({ inputShape: [1024], units: 128, activation: 'relu' }));

    model.add(tf.layers.dense({ units: CLASS_NAMES.length, activation: 'softmax' }));

    model.summary();


    // Compile the model with the defined optimizer and specify a loss function to use.

    model.compile({

        // Adam changes the learning rate over time which is useful.

        optimizer: 'adam',

        // Use the correct loss function. If 2 classes of data, must use binaryCrossentropy.

        // Else categoricalCrossentropy is used if more than 2 classes.

        loss: (CLASS_NAMES.length === 2) ? 'binaryCrossentropy' : 'categoricalCrossentropy',

        // As this is a classification problem you can record accuracy in the logs too!

        metrics: ['accuracy']

    });

}



loadMobileNetFeatureModel();


function hasGetUserMedia() {

    return !!(navigator.mediaDevices && navigator.mediaDevices.getUserMedia);

}



function enableCam() {

    if (hasGetUserMedia()) {

        // getUsermedia parameters.

        const constraints = {

            video: true,

            width: 640,

            height: 480

        };



        // Activate the webcam stream.

        navigator.mediaDevices.getUserMedia(constraints).then(function (stream) {

            VIDEO.srcObject = stream;

            VIDEO.addEventListener('loadeddata', function () {

                videoPlaying = true;

                ENABLE_CAM_BUTTON.classList.add('removed');

            });

        });

    } else {

        console.warn('getUserMedia() is not supported by your browser');

    }

}

/**

/**

* Handle Data Gather for button mouseup/mousedown.

**/

function gatherDataForClass() {

    let classNumber = parseInt(this.getAttribute('data-1hot'));

    gatherDataState = (gatherDataState === STOP_DATA_GATHER) ? classNumber : STOP_DATA_GATHER;

    dataGatherLoop();

}