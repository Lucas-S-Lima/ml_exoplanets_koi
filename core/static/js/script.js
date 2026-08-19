const form = document.getElementById("analysisForm");
const loading = document.getElementById("analysisLoading");
const submitButton = document.getElementById("submitButton");

form.addEventListener("submit", function (event) {

    if (!form.checkValidity()) {
        return;
    }

    event.preventDefault();

    submitButton.disabled = true;

    loading.classList.remove("d-none");

    setTimeout(function () {
        form.submit();
    }, 3000);

});
