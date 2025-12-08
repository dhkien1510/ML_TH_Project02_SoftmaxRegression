// ---------------------------
// Canvas Drawing Logic
// ---------------------------
const canvas = document.getElementById("drawCanvas");
const ctx = canvas.getContext("2d");

let drawing = false;

ctx.lineWidth = 20;
ctx.lineCap = "round";
ctx.strokeStyle = "black";

canvas.addEventListener("mousedown", () => (drawing = true));
canvas.addEventListener("mouseup", () => {
    drawing = false;
    ctx.beginPath();
});
canvas.addEventListener("mouseleave", () => {
    drawing = false;
    ctx.beginPath();
});

canvas.addEventListener("mousemove", (e) => {
    if (!drawing) return;
    const rect = canvas.getBoundingClientRect();
    ctx.lineTo(e.clientX - rect.left, e.clientY - rect.top);
    ctx.stroke();
    ctx.beginPath();
    ctx.moveTo(e.clientX - rect.left, e.clientY - rect.top);
});

// Clear
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
// Send base64 image to backend
// ---------------------------
async function sendImageToBackend(base64Image) {
    const res = await fetch("http://localhost:5000/predict", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ image: base64Image })
    });

    return await res.json();
}


// ---------------------------
// Predict from Canvas
// ---------------------------
document.getElementById("predictDrawBtn").addEventListener("click", async () => {
    const base64 = canvas.toDataURL("image/png"); // convert to base64

    const result = await sendImageToBackend(base64);

    displayPrediction(result.predicted_class, result.probabilities);
});


// ---------------------------
// Predict from Uploaded Image
// ---------------------------
document.getElementById("predictUploadBtn").addEventListener("click", async () => {
    if (previewImage.classList.contains("d-none")) return;

    const base64 = previewImage.src; // already base64 from reader

    const result = await sendImageToBackend(base64);

    console.log(result);

    displayPrediction(result.predicted_class, result.probabilities);
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
