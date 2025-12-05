// ---------------------------
// Canvas Drawing Logic
// ---------------------------
const canvas = document.getElementById("drawCanvas");
const ctx = canvas.getContext("2d");

let drawing = false;

// Set drawing style
ctx.lineWidth = 20;
ctx.lineCap = "round";
ctx.strokeStyle = "black";

canvas.addEventListener("mousedown", () => (drawing = true));
canvas.addEventListener("mouseup", () => (drawing = false));
canvas.addEventListener("mouseleave", () => (drawing = false));

canvas.addEventListener("mousemove", (e) => {
    if (!drawing) return;
    const rect = canvas.getBoundingClientRect();
    ctx.lineTo(e.clientX - rect.left, e.clientY - rect.top);
    ctx.stroke();
    ctx.beginPath();
    ctx.moveTo(e.clientX - rect.left, e.clientY - rect.top);
});

// Clear button
document.getElementById("clearBtn").addEventListener("click", () => {
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    ctx.beginPath();
});


// ---------------------------
// Upload Image Preview
// ---------------------------
const uploadInput = document.getElementById("uploadInput");
const previewImage = document.getElementById("previewImage");

uploadInput.addEventListener("change", function () {
    const file = uploadInput.files[0];
    if (!file) return;

    const reader = new FileReader();
    reader.onload = function (e) {
        previewImage.src = e.target.result;
        previewImage.classList.remove("d-none");
    };
    reader.readAsDataURL(file);
});


// ---------------------------
// Replace this with trained model
// ---------------------------
function SoftmaxPredict(imageData) {
  // Generate 10 random probabilities

    let probs = Array.from({ length: 10 }, () => Math.random());
    let sum = probs.reduce((a, b) => a + b, 0);
    probs = probs.map((p) => p / sum);

    return probs;
}


// ---------------------------
// Predict from Canvas
// ---------------------------
document.getElementById("predictDrawBtn").addEventListener("click", () => {
    const imageData = ctx.getImageData(0, 0, 280, 280);
    const probs = SoftmaxPredict(imageData);

    const predictedDigit = probs.indexOf(Math.max(...probs));
    displayPrediction(predictedDigit, probs);
});


// ---------------------------
// Predict from Uploaded Image
// ---------------------------
document.getElementById("predictUploadBtn").addEventListener("click", () => {
    if (previewImage.classList.contains("d-none")) return;

    // Create temp canvas to extract pixel data
    const tempCanvas = document.createElement("canvas");
    tempCanvas.width = 280;
    tempCanvas.height = 280;
    const tctx = tempCanvas.getContext("2d");

    tctx.drawImage(previewImage, 0, 0, 280, 280);
    const imageData = tctx.getImageData(0, 0, 280, 280);

    const probs = SoftmaxPredict(imageData);
    const predictedDigit = probs.indexOf(Math.max(...probs));

    displayPrediction(predictedDigit, probs);
});


// ---------------------------
// Show Prediction + Chart
// ---------------------------
let chartInstance = null;

function displayPrediction(digit, probabilities) {
    document.getElementById("resultDigit").textContent = digit;

    const ctxChart = document.getElementById("probChart").getContext("2d");

    if (chartInstance) chartInstance.destroy();

    chartInstance = new Chart(ctxChart, {
        type: "bar",
        data: {
        labels: ["0","1","2","3","4","5","6","7","8","9"],
        datasets: [
            {
            label: "Probability",
            data: probabilities,
            },
        ],
        },
        options: {
        scales: {
            y: { beginAtZero: true }
        }
        }
    });
}
