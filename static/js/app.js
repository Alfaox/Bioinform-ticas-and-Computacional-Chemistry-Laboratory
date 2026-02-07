// app.js - Client-side logic for Flask app

const smilesInput = document.getElementById('smiles-input');
const evaluateBtn = document.getElementById('evaluate-btn');
const resultsContainer = document.getElementById('results-container');
const loader = document.getElementById('loader');
const errorBox = document.getElementById('error-box');

evaluateBtn.addEventListener('click', async () => {
    const smiles = smilesInput.value.trim();

    if (!smiles) {
        showError('Please enter a SMILES structure');
        return;
    }

    // Show loader
    loader.style.display = 'block';
    resultsContainer.style.display = 'none';
    errorBox.style.display = 'none';

    try {
        const response = await fetch('/predict', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({ smiles })
        });

        const data = await response.json();

        if (!data.ok) {
            showError(data.error || 'Prediction failed');
            return;
        }

        displayResults(data);

    } catch (error) {
        showError('Network error: ' + error.message);
    } finally {
        loader.style.display = 'none';
    }
});

function displayResults(data) {
    resultsContainer.style.display = 'block';

    // Molecule image
    if (data.molecule_image) {
        document.getElementById('molecule-img').src = data.molecule_image;
        document.getElementById('molecule-section').style.display = 'block';
    }

    // DoA
    const doa = data.doa;
    const doaCard = document.getElementById('doa-card');
    document.getElementById('doa-status').textContent = doa.in_domain ? 'Within Domain ✅' : 'Out of Domain ⚠️';
    document.getElementById('doa-details').textContent = `Similarity: ${doa.kNN_mean_sim.toFixed(3)} | Threshold: ${doa.threshold.toFixed(3)}`;
    doaCard.className = 'result-card ' + (doa.in_domain ? 'status-success' : 'status-warning');

    // Stage 1: Toxicity
    const s1 = data.stage1_dsstox_like;
    const s1Card = document.getElementById('s1-card');
    document.getElementById('s1-status').textContent = s1.alert ? 'TOXIC ALERT ⛔' : 'Pass Toxicity Filter ✅';
    document.getElementById('s1-details').textContent = `Probability: ${s1.p.toFixed(3)} | Threshold: ${s1.threshold.toFixed(2)}`;
    s1Card.className = 'result-card ' + (s1.alert ? 'status-error' : 'status-success');

    // Stage 2: Bioactivity
    const s2 = data.stage2_clue_like;
    const s2Card = document.getElementById('s2-card');
    if (s2.p !== null) {
        s2Card.style.display = 'block';
        document.getElementById('s2-status').textContent = s2.pass ? 'Bioactivity Detected ✅' : 'No Bioactivity ⚠️';
        document.getElementById('s2-details').textContent = `Probability: ${s2.p.toFixed(3)} | Threshold: ${s2.threshold.toFixed(2)}`;
        s2Card.className = 'result-card ' + (s2.pass ? 'status-success' : 'status-warning');
    } else {
        s2Card.style.display = 'none';
    }

    // Stage 3: Target Specificity
    const s3 = data.stage3_uiref_like;
    const s3Card = document.getElementById('s3-card');
    if (s3.p !== null) {
        s3Card.style.display = 'block';
        document.getElementById('s3-status').textContent = s3.candidate ? 'TARGET CANDIDATE ⭐' : 'Not Specific';
        document.getElementById('s3-details').textContent = `Probability: ${s3.p.toFixed(3)} | Threshold: ${s3.threshold.toFixed(2)}`;
        s3Card.className = 'result-card ' + (s3.candidate ? 'status-success' : 'status-warning');
    } else {
        s3Card.style.display = 'none';
    }

    // Final Label
    const finalLabel = document.getElementById('final-label');
    finalLabel.textContent = data.final_label;
    finalLabel.className = 'conclusion-box ' + getConclusionClass(data.final_label);
}

function getConclusionClass(label) {
    if (label.includes('CANDIDATO') || label.includes('CANDIDATE')) return 'success';
    if (label.includes('ALERTA') || label.includes('ALERT')) return 'error';
    return 'warning';
}

function showError(message) {
    errorBox.textContent = '❌ ' + message;
    errorBox.style.display = 'block';
    resultsContainer.style.display = 'none';
}
