/*
const fs = require('fs'); // Import the 'fs' module for file system operations

// Function to read JSON file and extract names
const fetchNames = () => {
    // Read the JSON file asynchronously
    fs.readFile('food_items.json', 'utf8', (err, data) => {
        if (err) {
            console.error('Error reading file:', err);
            return;
        }

        try {
            // Parse the JSON data
            const jsonData = JSON.parse(data);
            
            // Extract names into an array
            const namesArray = jsonData.map(item => item.name);

            // Log the names array
            console.log('Names:', namesArray);

            // Store the names array in a new JSON file
            storeNamesInFile(namesArray);
        } catch (error) {
            console.error('Error parsing JSON:', error);
        }
    });
};

// Function to store names array in a JSON file
const storeNamesInFile = (namesArray) => {
    // Convert the names array to JSON format
    const namesJson = JSON.stringify(namesArray, null, 2); // Pretty print with 2 spaces indentation

    // Write the names array to 'names.json'
    fs.writeFile('names.json', namesJson, 'utf8', (err) => {
        if (err) {
            console.error('Error writing file:', err);
            return;
        }
        console.log('Names successfully written to names.json');
    });
};

// Call the function to fetch names
fetchNames();
*/
