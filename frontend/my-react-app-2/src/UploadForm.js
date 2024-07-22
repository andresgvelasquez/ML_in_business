import React, { useState } from 'react';
import axios from 'axios'; // Asegúrate de instalar axios con `npm install axios`

function UploadForm() {
  const [file, setFile] = useState(null);
  const [uploadStatus, setUploadStatus] = useState('');

  const handleFileChange = (e) => {
    setFile(e.target.files[0]);
  };

  const handleSubmit = async (e) => {
    e.preventDefault();

    if (!file) {
      setUploadStatus('Please select a file before submitting.');
      return;
    }

    const formData = new FormData();
    formData.append('file', file);

    try {
      const response = await axios.post('http://127.0.0.1:8000/api/upload/', formData, {
        headers: {
          'Content-Type': 'multipart/form-data',
        },
      });

      setUploadStatus(`File uploaded successfully: ${response.data.message}`);
    } catch (error) {
      setUploadStatus(`Error uploading file: ${error.message}`);
    }
  };

  return (
    <div>
      <h1>Upload CSV</h1>
      <form onSubmit={handleSubmit}>
        <input 
          type="file" 
          accept=".csv" 
          onChange={handleFileChange} 
        />
        <button type="submit">Upload</button>
      </form>
      {uploadStatus && <p>{uploadStatus}</p>}
    </div>
  );
}

export default UploadForm;