// script.js
document.addEventListener('DOMContentLoaded', function() {
    const summarizeButton = document.getElementById('summarize');
    const clearButton = document.getElementById('clear');
    const classifyButton = document.getElementById('classify');
    const documentInput = document.getElementById('document');
    const uploadFileInput = document.getElementById('uploadFile');
    const summaryOutput = document.getElementById('summary');
    const classificationOutput = document.getElementById('classification-output');
    const loading = document.getElementById('loading');
    const wordFreqOutput = document.getElementById('word-freq');
    const entitiesOutput = document.getElementById('entities');
    const lengthInfoOutput = document.getElementById('length-info');
    const errorMessage = document.getElementById('error-message');

    uploadFileInput.addEventListener('change', () => {
        // Update placeholder based on the selected file type
        const file = uploadFileInput.files[0];
        updatePlaceholder(file);
        
        // Read the content of the selected file and populate the textarea
        readFile(file).then(content => {
            documentInput.value = content;
        });
    });

    summarizeButton.addEventListener('click', async () => {
        let documentText = documentInput.value;

        // Show loading spinner
        loading.style.display = 'block';
        errorMessage.textContent = ''; // Clear previous error messages

        try {
            const response = await fetch('/summarize', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify({ document: documentText })
            });

            const data = await response.json();

            // Hide loading spinner
            loading.style.display = 'none';

            summaryOutput.textContent = data.summary;

            // Display word frequency analysis
            wordFreqOutput.innerHTML = '<strong>Word Frequency Analysis:</strong><br>';
            if (data.word_freq) {
                data.word_freq.forEach(([word, count]) => {
                    wordFreqOutput.innerHTML += `${word}: ${count}<br>`;
                });
            }

            // Display named entities
            entitiesOutput.innerHTML = '<strong>Named Entities:</strong><br>';
            if (data.entities) {
                data.entities.forEach(([text, label]) => {
                    entitiesOutput.innerHTML += `${text} (${label})<br>`;
                });
            }

            // Display sentiment analysis
            
            // Display document length information
            lengthInfoOutput.innerHTML = `<strong>Document Length:</strong><br>Word Count: ${data.word_count || 0}<br>Character Count: ${data.char_count || 0}`;
        } catch (error) {
            console.error('Error in summarization:', error);
            loading.style.display = 'none';
            errorMessage.textContent = 'Error in summarization. Please try again.';
        }
    });

    classifyButton.addEventListener('click', async () => {
        let documentText = documentInput.value;

        // Show loading spinner
        loading.style.display = 'block';
        errorMessage.textContent = ''; // Clear previous error messages

        try {
            // Call the classifyDocument function to classify the document
            const data = await classifyDocument(documentText);

            // Hide loading spinner
            loading.style.display = 'none';

            // Display classification results
            classificationOutput.innerHTML = `<strong>Predicted Class:</strong> ${data.predicted_class}<br>`;
            classificationOutput.innerHTML += '<strong>Class Probabilities:</strong><br>';
            data.class_probabilities.forEach((probability, index) => {
                classificationOutput.innerHTML += `Class ${index}: ${probability.toFixed(4)}<br>`;
            });
        } catch (error) {
            console.error('Error in text classification:', error);
            loading.style.display = 'none';
            errorMessage.textContent = 'Error in text classification. Please try again.';
        }
    });

    clearButton.addEventListener('click', () => {
        documentInput.value = '';
        summaryOutput.textContent = '';
        wordFreqOutput.innerHTML = '';
        entitiesOutput.innerHTML = '';
        lengthInfoOutput.innerHTML = '';
        classificationOutput.innerHTML = ''; // Clear classification results
        errorMessage.textContent = '';
        uploadFileInput.value = ''; // Clear the file input
        updatePlaceholder(); // Reset placeholder
    });

    // Function to update placeholder based on the selected file type
    function updatePlaceholder(file) {
        if (file) {
            const fileType = file.name.split('.').pop().toLowerCase();
            documentInput.placeholder = `Enter your ${fileType} document...`;
        } else {
            documentInput.placeholder = 'Enter your document...';
        }
    }

    // Function to read file content
    function readFile(file) {
        return new Promise((resolve, reject) => {
            const reader = new FileReader();
            reader.onload = (event) => {
                resolve(event.target.result);
            };
            reader.onerror = (error) => {
                reject(error);
            };
            reader.readAsText(file);
        });
    }

    // Function to classify the document using the backend API
    async function classifyDocument(documentText) {
        try {
            // Make a POST request to the backend API
            const response = await fetch("/classify", {
                method: "POST",
                headers: {
                    "Content-Type": "application/json",
                },
                body: JSON.stringify({
                    document: documentText,
                }),
            });

            // Check if the request was successful (status code 200)
            if (!response.ok) {
                throw new Error(`HTTP error! Status: ${response.status}`);
            }

            // Parse the JSON response
            const result = await response.json();

            // Return the result
            return result;
        } catch (error) {
            // Throw the error for further handling
            throw error;
        }
    }
});
