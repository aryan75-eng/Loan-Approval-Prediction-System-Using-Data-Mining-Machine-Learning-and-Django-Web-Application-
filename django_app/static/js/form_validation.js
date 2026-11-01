/**
 * form_validation.js — Client-side validation helpers.
 * Uses Bootstrap 5 validation classes.
 */

"use strict";

// Bootstrap native validation on all forms with .needs-validation
document.addEventListener("DOMContentLoaded", () => {
  const forms = document.querySelectorAll(".needs-validation");
  forms.forEach((form) => {
    form.addEventListener("submit", (event) => {
      if (!form.checkValidity()) {
        event.preventDefault();
        event.stopPropagation();
      }
      form.classList.add("was-validated");
    });
  });
});

/**
 * Validate that a numeric input is within [min, max].
 * @param {HTMLInputElement} input
 * @param {number} min
 * @param {number} max
 * @returns {boolean}
 */
function validateNumericRange(input, min, max) {
  const val = parseFloat(input.value);
  if (isNaN(val) || val < min || val > max) {
    input.classList.add("is-invalid");
    return false;
  }
  input.classList.remove("is-invalid");
  input.classList.add("is-valid");
  return true;
}

/**
 * Highlight file input with validation feedback.
 * @param {HTMLInputElement} fileInput
 * @param {string[]} allowedExtensions - e.g. ['.csv', '.xlsx']
 * @param {number} maxSizeBytes
 */
function validateFileInput(fileInput, allowedExtensions, maxSizeBytes) {
  const file = fileInput.files[0];
  if (!file) return false;

  const ext = "." + file.name.split(".").pop().toLowerCase();
  if (!allowedExtensions.includes(ext)) {
    showInputError(fileInput, `File type '${ext}' not allowed. Use: ${allowedExtensions.join(", ")}`);
    return false;
  }
  if (file.size > maxSizeBytes) {
    showInputError(fileInput, `File too large (${(file.size / 1048576).toFixed(1)} MB). Max: ${maxSizeBytes / 1048576} MB.`);
    return false;
  }
  fileInput.classList.remove("is-invalid");
  return true;
}

function showInputError(input, message) {
  input.classList.add("is-invalid");
  let fb = input.nextElementSibling;
  if (!fb || !fb.classList.contains("invalid-feedback")) {
    fb = document.createElement("div");
    fb.className = "invalid-feedback";
    input.parentNode.insertBefore(fb, input.nextSibling);
  }
  fb.textContent = message;
}
