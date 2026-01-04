document.addEventListener('DOMContentLoaded', () => {
    const dropZone = document.getElementById('drop-zone');
    const fileInput = document.getElementById('file-input');
    const previewPanel = document.getElementById('preview-panel');
    const imagePreview = document.getElementById('image-preview');
    const cancelBtn = document.getElementById('cancel-btn');
    const analyzeBtn = document.getElementById('analyze-btn');
    const resultsSection = document.getElementById('results-section');
    const resultContent = document.getElementById('result-content');
    const loader = document.getElementById('loader');

    const studentIdDisplay = document.getElementById('student-id');
    const confidenceBadge = document.getElementById('confidence-badge');
    const segmentsCount = document.getElementById('segments-count');

    let selectedFile = null;

    // Trigger file input
    dropZone.addEventListener('click', () => fileInput.click());

    // Drag and Drop
    dropZone.addEventListener('dragover', (e) => {
        e.preventDefault();
        dropZone.classList.add('drag-over');
    });

    ['dragleave', 'drop'].forEach(evt => {
        dropZone.addEventListener(evt, () => dropZone.classList.remove('drag-over'));
    });

    dropZone.addEventListener('drop', (e) => {
        e.preventDefault();
        const files = e.dataTransfer.files;
        if (files.length) handleFile(files[0]);
    });

    fileInput.addEventListener('change', (e) => {
        if (e.target.files.length) handleFile(e.target.files[0]);
    });

    function handleFile(file) {
        if (!file.type.startsWith('image/')) {
            alert('Please upload an image file.');
            return;
        }
        selectedFile = file;
        const reader = new FileReader();
        reader.onload = (e) => {
            imagePreview.src = e.target.result;
            dropZone.classList.add('hidden');
            previewPanel.classList.remove('hidden');
            resultsSection.classList.add('hidden');
        };
        reader.readAsDataURL(file);
    }

    cancelBtn.addEventListener('click', () => {
        selectedFile = null;
        fileInput.value = '';
        previewPanel.classList.add('hidden');
        dropZone.classList.remove('hidden');
        resultsSection.classList.add('hidden');
    });

    analyzeBtn.addEventListener('click', async () => {
        if (!selectedFile) return;

        // UI State: Loading
        analyzeBtn.disabled = true;
        analyzeBtn.textContent = 'Analyzing...';
        resultsSection.classList.remove('hidden');
        resultContent.classList.add('hidden');
        loader.classList.remove('hidden');

        const formData = new FormData();
        formData.append('file', selectedFile);

        try {
            const response = await fetch('/predict', {
                method: 'POST',
                body: formData
            });

            if (!response.ok) throw new Error('Prediction failed');

            const data = await response.json();

            // Update UI with results
            studentIdDisplay.textContent = data.label;
            confidenceBadge.textContent = `${data.confidence}% Confidence`;
            segmentsCount.textContent = data.num_segments;

            // Success Transition
            loader.classList.add('hidden');
            resultContent.classList.remove('hidden');

            // Smooth scroll to results
            resultsSection.scrollIntoView({ behavior: 'smooth' });

        } catch (error) {
            console.error('Error:', error);
            alert('An error occurred during analysis.');
            resultsSection.classList.add('hidden');
        } finally {
            analyzeBtn.disabled = false;
            analyzeBtn.textContent = 'Analyze Handwriting';
        }
    });
});
