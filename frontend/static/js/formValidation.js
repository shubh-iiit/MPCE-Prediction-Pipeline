// Gender sum validation: Ensure sum of gender fields equals household size
function checkGenderSum() {
    const male = parseInt(document.getElementById('gender_male').value) || 0;
    const female = parseInt(document.getElementById('gender_female').value) || 0;
    const others = parseInt(document.getElementById('gender_others').value) || 0;
    const hhSize = parseInt(document.getElementById('hh_size').value) || 0;
    const totalGender = male + female + others;

    const genderFemaleInput = document.getElementById('gender_female');
    let group = genderFemaleInput ? genderFemaleInput.closest('.form-group') : null;
    if (!group && genderFemaleInput) group = genderFemaleInput.parentNode;

    let errorDiv = document.getElementById('gender_sum_error');
    if (errorDiv && errorDiv.parentNode) {
        errorDiv.parentNode.removeChild(errorDiv);
    }

    if (totalGender !== hhSize && group) {
        errorDiv = document.createElement('div');
        errorDiv.id = 'gender_sum_error';
        errorDiv.className = 'form-error';
        errorDiv.style.color = 'red';
        errorDiv.style.display = 'block';
        errorDiv.textContent = 'Total gender count does not match the total household count.';
        group.appendChild(errorDiv);
    }
}
['gender_male', 'gender_female', 'gender_others', 'hh_size'].forEach(id => {
    // Attach input event to gender and hh_size fields
    const el = document.getElementById(id);
    if (el) {
        el.addEventListener('input', checkGenderSum);
    }
});

// Min/Max Age and Education validation: Ensure min <= max for age and education
function checkMinMaxValidation() {
    const minAge = parseInt(document.getElementById('min_age').value) || 0;
    const maxAge = parseInt(document.getElementById('max_age').value) || 0;

    let ageErrorDiv = document.getElementById('minmax_age_error');
    const minAgeInput = document.getElementById('min_age');
    let ageGroup = minAgeInput ? minAgeInput.closest('.form-group') : null;
    if (!ageGroup && minAgeInput) ageGroup = minAgeInput.parentNode;
    if (ageErrorDiv && ageErrorDiv.parentNode) {
        ageErrorDiv.parentNode.removeChild(ageErrorDiv);
    }
    if (minAge > maxAge && ageGroup) {
        ageErrorDiv = document.createElement('div');
        ageErrorDiv.id = 'minmax_age_error';
        ageErrorDiv.className = 'form-error';
        ageErrorDiv.style.color = 'red';
        ageErrorDiv.style.display = 'block';
        ageErrorDiv.textContent = 'Min Age should be less than or equal to Max Age.';
        ageGroup.appendChild(ageErrorDiv);
    }

    let eduErrorDiv = document.getElementById('minmax_edu_error');
    const minEduInput = document.getElementById('min_edu');
    let eduGroup = minEduInput ? minEduInput.closest('.form-group') : null;
    if (!eduGroup && minEduInput) eduGroup = minEduInput.parentNode;
    if (eduErrorDiv && eduErrorDiv.parentNode) {
        eduErrorDiv.parentNode.removeChild(eduErrorDiv);
    }
    const minEduSelect = document.getElementById('min_edu');
    const maxEduSelect = document.getElementById('max_edu');
    const minEduIdx = minEduSelect && minEduSelect.value !== "" ? parseInt(minEduSelect.value) : null;
    const maxEduIdx = maxEduSelect && maxEduSelect.value !== "" ? parseInt(maxEduSelect.value) : null;
    if (
        minEduIdx !== null && maxEduIdx !== null &&
        minEduIdx > maxEduIdx && eduGroup
    ) {
        eduErrorDiv = document.createElement('div');
        eduErrorDiv.id = 'minmax_edu_error';
        eduErrorDiv.className = 'form-error';
        eduErrorDiv.style.color = 'red';
        eduErrorDiv.style.display = 'block';
        eduErrorDiv.textContent = 'Min Education should be less than or equal to Max Education.';
        eduGroup.appendChild(eduErrorDiv);
    }
}
['min_age', 'max_age', 'min_edu', 'max_edu'].forEach(id => {
    // Attach input/change event to min/max fields
    const el = document.getElementById(id);
    if (el) {
        el.addEventListener('input', checkMinMaxValidation);
        el.addEventListener('change', checkMinMaxValidation);
    }
});

// Internet users count validation: Ensure internet users <= household size
function checkInternetUsersVsHHSize() {
    const internetUsers = parseInt(document.getElementById('internet_users').value) || 0;
    const hhSize = parseInt(document.getElementById('hh_size').value) || 0;

    const internetInput = document.getElementById('internet_users');
    let group = internetInput ? internetInput.closest('.form-group') : null;
    if (!group && internetInput) group = internetInput.parentNode;

    let errorDiv = document.getElementById('internet_users_error');
    if (errorDiv && errorDiv.parentNode) {
        errorDiv.parentNode.removeChild(errorDiv);
    }

    if (internetUsers > hhSize && group) {
        errorDiv = document.createElement('div');
        errorDiv.id = 'internet_users_error';
        errorDiv.className = 'form-error';
        errorDiv.style.color = 'red';
        errorDiv.style.display = 'block';
        errorDiv.textContent = 'Internet users should not exceed household size.';
        group.appendChild(errorDiv);
    }
}
['internet_users', 'hh_size'].forEach(id => {
    // Attach input event to internet_users and hh_size fields
    const el = document.getElementById(id);
    if (el) {
        el.addEventListener('input', checkInternetUsersVsHHSize);
    }
});

// Reset validation styling on focus for all form controls
document.querySelectorAll('.form-control, input[type="radio"], input[type="checkbox"]').forEach(field => {
    field.addEventListener('focus', function() {
        this.classList.remove('invalid');
        if (this.parentElement) {
            this.parentElement.style.borderColor = '';
        }
        const errorDiv = this.nextElementSibling;
        if (errorDiv && errorDiv.classList.contains('form-error')) {
            errorDiv.style.display = 'none';
        }
    });
});

