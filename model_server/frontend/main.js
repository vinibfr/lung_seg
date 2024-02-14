const inputData = {
    // your input data
};

fetch('http://localhost:5000/predict', {
    method: 'POST',
    headers: {
        'Content-Type': 'application/json',
    },
    body: JSON.stringify(inputData),
})
.then(response => response.json())
.then(result => {
    // Handle the result from the Python model
    console.log(result);
})
.catch(error => {
    console.error('Error:', error);
});
