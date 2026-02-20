document.addEventListener("DOMContentLoaded", () => {
  const input = document.getElementById("imageInput");
  const previewImg = document.getElementById("previewImg");
  const previewText = document.getElementById("previewText");
  const loadingText = document.getElementById("loadingText");

  if (input) {
    input.addEventListener("change", () => {
      const file = input.files[0];
      if (file) {
        const reader = new FileReader();
        reader.onload = () => {
          previewImg.src = reader.result;
          previewImg.style.display = "block";
          previewText.style.display = "none";
        };
        reader.readAsDataURL(file);
      }
    });
  }

  const form = document.querySelector(".upload-card");
  if (form) {
    form.addEventListener("submit", () => {
      loadingText.style.display = "block";
    });
  }
});
