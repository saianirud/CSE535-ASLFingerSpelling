const tf = require('@tensorflow/tfjs-node-gpu');
const posenet = require('@tensorflow-models/posenet');

const {
    createCanvas, Image
} = require('canvas');

let fs = require('fs');
let path = require('path');

const readline = require('readline').createInterface({
    input: process.stdin,
    output: process.stdout
});

// Function Params to decide Pose Estimation
const imageScaleFactor = 0.5;
const outputStride = 16;
const flipHorizontal = false;
const defaultQuantBytes = 4;

const defaultMobileNetMultiplier = 0.75;
const defaultMobileNetStride = 16;
const defaultMobileNetInputResolution = 801;

// Posenet instance params used
let guiState = {
    net: null,
    model: {
        architecture: 'MobileNetV1',
        outputStride: defaultMobileNetStride,
        inputResolution: defaultMobileNetInputResolution,
        multiplier: defaultMobileNetMultiplier,
        quantBytes: defaultQuantBytes,
    },
    image: 'tennis_in_crowd.jpg',
    multiPoseDetection: {
        minPartConfidence: 0.1,
        minPoseConfidence: 0.2,
        nmsRadius: 20.0,
        maxDetections: 15,
    },
    showKeypoints: true,
    showSkeleton: true,
    showBoundingBox: false,
};

// Path to the location where frames of all the video files are stored
getArgumentValue = process.argv[2];
if (getArgumentValue === undefined) {
    photo_path_to_frames = path.join(__dirname, "../Posenet_Frames");
} else {
    photo_path_to_frames = getArgumentValue;
}


/**
 * Asynchronous Function to decide poses for a set of images and storing it in a single json file
 *
 * @returns {Promise<void>}
 * @param photo_path
 * @param choice
 */
async function cascading_images_pose_estimation(photo_path, choice) {
    let config;
    if (choice === '1') {
        console.log("Loading model : MobileNetV1\n");
        config = {
            architecture: guiState.model.architecture,
            outputStride: guiState.model.outputStride,
            inputResolution: guiState.model.inputResolution,
            multiplier: guiState.model.multiplier,
            quantBytes: guiState.model.quantBytes
        };
    } else {
        console.log("Loading Model : ResNet50\n");
        config = {
            architecture: "ResNet50",
            outputStride: guiState.model.outputStride,
            inputResolution: guiState.model.inputResolution,
            quantBytes: guiState.model.quantBytes
        };
    }

    const single_net = await posenet.load(config);

    let canvas;
    let input;
    let ctx;

    fs.readdir(photo_path, async function (err, items) {
        let func_path = "";
        for (let i = 0; i < items.length; i++) {
            let pose_list = [];
            if (path.extname(items[i]) === "") {
                func_path = photo_path + "/" + items[i] + "/";
                let length = fs.readdirSync(func_path).length;
                for (let i = 0; i < length - 1; i++) {
                    let image = await loadImage(func_path + i + ".png");
                    // image.src = func_path + i + ".png";
                    canvas = createCanvas(image.width, image.height);
                    ctx = canvas.getContext('2d');
                    ctx.drawImage(image, 0, 0);
                    input = tf.browser.fromPixels(canvas);
                    let pose = await single_net.estimateSinglePose(input, imageScaleFactor, flipHorizontal, outputStride);
                    input.dispose();
                    pose_list.push(pose);
                }

                fs.writeFileSync(func_path + "key_points.json", JSON.stringify(pose_list));
                console.log("Key Points generated for file: " + items[i]);
            }
        }
    });
}

async function loadImage(path) {
    let image = new Image();
    const promise = new Promise((resolve, reject) => {
        image.onload = () => {
            resolve(image);
        };
    });
    image.src = path;
    return promise;
}

readline.question(`\nInput your Architecture.\n 1. MobileNetV1\n 2. ResNet50\n`, (name) => {
    cascading_images_pose_estimation(photo_path_to_frames, name);
    readline.close();
    process.stdin.destroy();
});