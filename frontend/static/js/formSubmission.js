// Handle form submission via AJAX/fetch
document.getElementById('surveyForm').addEventListener('submit', function (e) {
    e.preventDefault();

    // Validate required fields
    let isValid = true;
    const requiredFields = this.querySelectorAll('[required]');
    requiredFields.forEach(field => {
        if (!field.value && field.type !== 'checkbox' && field.type !== 'radio') {
            isValid = false;
            field.classList.add('invalid');
            if (field.nextElementSibling && field.nextElementSibling.classList.contains('form-error')) {
                field.nextElementSibling.style.display = 'block';
            }
        }
    });

    // Check radio button groups for selection
    const radioGroups = {};
    this.querySelectorAll('input[type="radio"][required]').forEach(radio => {
        if (!radioGroups[radio.name]) {
            radioGroups[radio.name] = false;
        }
        if (radio.checked) {
            radioGroups[radio.name] = true;
        }
    });
    for (const group in radioGroups) {
        if (!radioGroups[group]) {
            this.querySelectorAll(`input[name="${group}"]`).forEach(radio => {
                radio.parentElement.style.borderColor = '#d9534f';
            });
        }
    }

    // If not valid, scroll to first invalid field
    if (!isValid) {
        const firstInvalid = this.querySelector('.invalid, input[style*="border-color: rgb(217, 83, 79)"]');
        if (firstInvalid) {
            firstInvalid.scrollIntoView({ behavior: 'smooth', block: 'center' });
        }
        return;
    }

    // Gather form data into JSON
    const formData = new FormData(this);
    const jsonData = {};
    formData.forEach((value, key) => {
        if (jsonData[key]) {
            if (!Array.isArray(jsonData[key])) jsonData[key] = [jsonData[key]];
            jsonData[key].push(value);
        } else {
            jsonData[key] = value;
        }
    });

    // Convert state name to code and parse numeric fields
    if (jsonData.state && stateNumberDict[jsonData.state]) {
        jsonData.state = stateNumberDict[jsonData.state];
    }
    if (jsonData.nco !== undefined && jsonData.nco !== "") {
        jsonData.nco = parseInt(jsonData.nco, 10);
    }
    if (jsonData.nic !== undefined && jsonData.nic !== "") {
        jsonData.nic = parseInt(jsonData.nic, 10);
    }
    if (jsonData.min_edu !== undefined && jsonData.min_edu !== "") {
        jsonData.min_edu = parseInt(jsonData.min_edu, 10);
    }
    if (jsonData.max_edu !== undefined && jsonData.max_edu !== "") {
        jsonData.max_edu = parseInt(jsonData.max_edu, 10);
    }

    // Submit data to backend /submit endpoint
    const uniqueKey = 'current_record';

    fetch('/submit', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json'
        },
        body: JSON.stringify({
            key: uniqueKey,
            record: jsonData
        })
    })
    .then(res => res.json())
    .then(response => {
        if (response.success) {
            // On success, fetch prediction from /process
            fetch('/process')
                .then(res => res.json())
                .then(data => {
                    if (data.success && data.result) {
                        const result = data.result;
                        let pred = result.predicted_expense;
                        // If pred is an array, extract the first element
                        if (Array.isArray(pred)) pred = pred[0];
                        // If pred is a string, try to convert to number
                        if (typeof pred === "string") pred = Number(pred);
                        // Show styled result card
                        const resultCard = document.getElementById('resultCard');
                        const resultAmount = document.getElementById('resultAmount');
                        resultAmount.innerHTML = `Predicted Expense: <b>₹${Number(pred).toLocaleString(undefined, {maximumFractionDigits:2})}</b>`;
                        resultCard.style.display = 'block';
                        document.getElementById('resultSection').style.display = 'block';
                        document.getElementById('resultSection').scrollIntoView({ behavior: 'smooth' });
                    } else {
                        // Show error if prediction unavailable
                        document.getElementById('resultDetails').textContent = 'Prediction unavailable.';
                        document.getElementById('resultSection').style.display = 'block';
                        document.getElementById('resultCard').style.display = 'none';
                    }
                });
        } else {
            // Show error if submission failed
            alert('Submission failed: ' + response.error);
        }
    })
    .catch(err => {
        // Handle network or other errors
        console.error('Error:', err);
        alert('An error occurred during submission.');
    });
});

// Prefill form with latest record from backend (if available)
window.addEventListener('DOMContentLoaded', function() {
    fetch('/records')
        .then(res => res.json())
        .then(data => {
            if (data && Object.keys(data).length > 0) {
                // Get the first record object
                const record = Object.values(data)[0];
                if (record && typeof record === 'object') {
                    Object.entries(record).forEach(([key, value]) => {
                        const el = document.querySelector(`[name="${key}"]`);
                        if (el) {
                            if (el.type === 'radio' || el.type === 'checkbox') {
                                if (Array.isArray(value)) {
                                    value.forEach(v => {
                                        const box = document.querySelector(`[name="${key}"][value="${v}"]`);
                                        if (box) box.checked = true;
                                    });
                                } else {
                                    const box = document.querySelector(`[name="${key}"][value="${value}"]`);
                                    if (box) box.checked = true;
                                }
                            } else {
                                el.value = value;
                            }
                        }
                    });
                }
            }
        });
});

// Attach to result section display to show data preview
const resultSection = document.getElementById('resultSection');
const observer = new MutationObserver(function(mutations) {
    mutations.forEach(function(mutation) {
        if(mutation.attributeName === "style" && resultSection.style.display === "block") {
            displayProcessedData();
        }
    });
});
observer.observe(resultSection, { attributes: true });
