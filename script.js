document.getElementById('imageUpload').addEventListener('change', function(event) {
    const file = event.target.files[0];
    if (file) {
        const reader = new FileReader();
        reader.onload = function(e) {
            const originalImagePreview = document.getElementById('originalImagePreview');
            originalImagePreview.innerHTML = `<img src="${e.target.result}" alt="Original Image">`;

            // Simulate background removal (replace with actual API call)
            setTimeout(() => {
                const resultImagePreview = document.getElementById('resultImagePreview');
                resultImagePreview.innerHTML = `<img src="${e.target.result}" alt="Result Image">`;
                document.getElementById('downloadBtn').disabled = false;
            }, 1000);
        };
        reader.readAsDataURL(file);
    }
});

document.getElementById('downloadBtn').addEventListener('click', function() {
    const resultImagePreview = document.getElementById('resultImagePreview').querySelector('img');
    if (resultImagePreview) {
        const link = document.createElement('a');
        link.href = resultImagePreview.src;
        link.download = 'background_removed_image.png';
        document.body.appendChild(link);
        link.click();
        document.body.removeChild(link);
    }
});