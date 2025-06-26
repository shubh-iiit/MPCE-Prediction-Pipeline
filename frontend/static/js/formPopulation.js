// Populate State dropdown
const stateSelect = document.getElementById('state');
Object.keys(stateNumberDict).forEach(stateName => {
    const option = document.createElement('option');
    option.value = stateName;
    option.text = stateName;
    stateSelect.appendChild(option);
});

// Populate Religion dropdown
const religionSelect = document.getElementById('religion');
Object.entries(religionCategories).forEach(([key, value]) => {
    if (value) {
        const option = document.createElement('option');
        option.value = key;
        option.text = value;
        religionSelect.appendChild(option);
    }
});

// Populate Caste dropdown
const casteSelect = document.getElementById('caste');
Object.entries(casteCategories).forEach(([key, value]) => {
    const option = document.createElement('option');
    option.value = key;
    option.text = value;
    casteSelect.appendChild(option);
});

// Populate Household Type dropdown based on sector selection (rural/urban)
document.querySelectorAll('input[name="sector"]').forEach(radio => {
    radio.addEventListener('change', function () {
        const selected = this.value === "1" ? "rural" : "urban";
        const householdType = document.getElementById('household_type');
        householdType.innerHTML = `<option value="">Select Household Type</option>`;
        const categories = householdCategories[selected];
        for (const code in categories) {
            const option = document.createElement('option');
            option.value = code;
            option.text = categories[code];
            householdType.appendChild(option);
        }
    });
});

// Remove region dropdown logic

// Add hidden input for region code if not present
let regionInput = document.getElementById('region');
if (!regionInput) {
    regionInput = document.createElement('input');
    regionInput.type = 'hidden';
    regionInput.id = 'region';
    regionInput.name = 'region';
    document.getElementById('state').parentElement.appendChild(regionInput);
}

// Populate District dropdown based on selected State (all districts from all regions)
stateSelect.addEventListener('change', function () {
    const stateName = this.value;
    const stateCode = stateNumberDict[stateName];
    const districtSelect = document.getElementById('district');
    districtSelect.innerHTML = `<option value="">Select District</option>`;
    let allDistricts = [];
    let regionMap = {};
    if (stateCode && stateRegions[stateCode]) {
        stateRegions[stateCode].forEach(([regionCode, regionName]) => {
            if (regionDistricts[regionCode]) {
                regionDistricts[regionCode].forEach(([districtCode, districtName]) => {
                    allDistricts.push({
                        regionCode,
                        districtCode,
                        districtName
                    });
                    regionMap[districtCode] = regionCode;
                });
            }
        });
    }
    // Sort districts alphabetically by districtName
    allDistricts.sort((a, b) => a.districtName.localeCompare(b.districtName));
    allDistricts.forEach(({regionCode, districtCode, districtName}) => {
        const option = document.createElement('option');
        option.value = districtCode;
        option.text = districtName;
        option.setAttribute('data-region', regionCode);
        districtSelect.appendChild(option);
    });
    // Clear region hidden input
    regionInput.value = '';
});

// When a district is selected, set the region hidden input accordingly
document.getElementById('district').addEventListener('change', function () {
    const selectedOption = this.options[this.selectedIndex];
    const regionCode = selectedOption ? selectedOption.getAttribute('data-region') : '';
    document.getElementById('region').value = regionCode || '';
});

// Populate NCO (Occupation) dropdown
const ncoSelect = document.getElementById('nco');
if (ncoSelect && typeof occupationDescriptions !== 'undefined') {
    ncoSelect.innerHTML = '<option value="">Select Occupation</option>';
    Object.entries(occupationDescriptions).forEach(([code, desc]) => {
        const option = document.createElement('option');
        option.value = code;
        option.text = desc;
        ncoSelect.appendChild(option);
    });
}

// Populate NIC (Industry) dropdown
const nicSelect = document.getElementById('nic');
if (nicSelect && typeof industrialClassification !== 'undefined') {
    nicSelect.innerHTML = '<option value="">Select Industry</option>';
    Object.entries(industrialClassification).forEach(([code, desc]) => {
        const option = document.createElement('option');
        option.value = code;
        option.text = desc;
        nicSelect.appendChild(option);
    });
}

// Populate min/max education dropdowns
const minEduSelect = document.getElementById('min_edu');
const maxEduSelect = document.getElementById('max_edu');
if (minEduSelect && maxEduSelect && typeof educationalQulification !== 'undefined') {
    minEduSelect.innerHTML = '<option value="">Select Education Level</option>';
    maxEduSelect.innerHTML = '<option value="">Select Education Level</option>';
    educationalQulification.forEach((label, idx) => {
        const minOpt = document.createElement('option');
        minOpt.value = idx;
        minOpt.text = label;
        minEduSelect.appendChild(minOpt);

        const maxOpt = document.createElement('option');
        maxOpt.value = idx;
        maxOpt.text = label;
        maxEduSelect.appendChild(maxOpt);
    });
}

// Enhance UX: Make checkbox/radio items clickable and initialize Select2
document.addEventListener('DOMContentLoaded', function() {
    // Make checkbox-item and radio-item clickable
    document.querySelectorAll('.checkbox-item').forEach(function(item) {
        item.addEventListener('click', function(e) {
            if (e.target.tagName === 'INPUT') return;
            const input = item.querySelector('input[type="checkbox"]');
            if (input) {
                input.checked = !input.checked;
                input.dispatchEvent(new Event('change', { bubbles: true }));
            }
        });
    });
    document.querySelectorAll('.radio-item').forEach(function(item) {
        item.addEventListener('click', function(e) {
            if (e.target.tagName === 'INPUT') return;
            const input = item.querySelector('input[type="radio"]');
            if (input && !input.checked) {
                input.checked = true;
                input.dispatchEvent(new Event('change', { bubbles: true }));
            }
        });
    });

    // Wait for jQuery and Select2 to be loaded, then initialize
    function initSelect2WhenReady() {
        if (window.jQuery && window.$ && $.fn.select2) {
            $('#nic').select2({
                placeholder: "Select Industry",
                allowClear: true,
                width: '100%'
            });
            $('#nco').select2({
                placeholder: "Select Occupation",
                allowClear: true,
                width: '100%'
            });
        } else {
            setTimeout(initSelect2WhenReady, 100);
        }
    }
    initSelect2WhenReady();
});
