import React, { useState } from 'react';
import axios from 'axios';
import './UploadForm.css';
import DataVisualization from './DataVisualization';
import ScatterPlot from './ScatterPlot';

const UploadForm = () => {
  const [file, setFile] = useState(null);
  const [uploadedData, setUploadedData] = useState(null);
  const [loading, setLoading] = useState(false);

  const handleFileChange = (e) => {
    setFile(e.target.files[0]);
  };

  const handleUpload = async () => {
    if (!file) {
      alert('Please select a file first!');
      return;
    }

    setLoading(true);

    const formData = new FormData();
    formData.append('file', file);

    try {
      const response = await axios.post('http://localhost:8000/api/upload/', formData, {
        headers: {
          'Content-Type': 'multipart/form-data'
        }
      });
      console.log('File uploaded successfully:', response.data);
      setUploadedData(response.data);
    } catch (error) {
      console.error('Error uploading file:', error);
      alert('Error uploading file.');
    } finally {
      setLoading(false);
    }
  };

  const renderTable = (data) => {
    if (!data) {
      return null;
    }

    const headers = Object.keys(data);
    if (headers.length === 0) {
      return <p>No data available to display.</p>;
    }

    const numRows = Object.keys(data[headers[0]]).length;
    const rows = [];
    for (let i = 0; i < numRows; i++) {
      const cells = headers.map(header => (
        <td key={header}>{data[header][i]}</td>
      ));
      rows.push(<tr key={i}>{cells}</tr>);
    }

    return (
      <table className="styled-table">
        <thead>
          <tr>
            {headers.map(header => (
              <th key={header}>{header}</th>
            ))}
          </tr>
        </thead>
        <tbody>
          {rows}
        </tbody>
      </table>
    );
  };

  const renderGeneralSummary = () => {
    if (!uploadedData || !uploadedData.general_summary) {
      return null;
    }

    const generalSummary = uploadedData.general_summary;

    return (
      <table className="styled-table">
        <thead>
          <tr>
            {Object.keys(generalSummary).map(key => (
              <th key={key}>{key}</th>
            ))}
          </tr>
        </thead>
        <tbody>
          <tr>
            {Object.values(generalSummary).map((value, index) => (
              <td key={index}>{value}</td>
            ))}
          </tr>
        </tbody>
      </table>
    );
  };

  const renderColumnSummary = () => {
    if (!uploadedData || !uploadedData.column_summary) {
      return null;
    }

    const columnSummary = uploadedData.column_summary;

    return (
      <table className="styled-table">
        <thead>
          <tr>
            {Object.keys(columnSummary[0]).map(key => (
              <th key={key}>{key}</th>
            ))}
          </tr>
        </thead>
        <tbody>
          {columnSummary.map((row, index) => (
            <tr key={index}>
              {Object.values(row).map((value, cellIndex) => (
                <td key={cellIndex}>{value}</td>
              ))}
            </tr>
          ))}
        </tbody>
      </table>
    );
  };

  const renderVolumePredictions = () => {
    if (!uploadedData || !uploadedData.volumen_predictions) {
      return null;
    }

    const volumePredictions = uploadedData.volumen_predictions;

    return (
      <table className="styled-table">
        <thead>
          <tr>
            {Object.keys(volumePredictions).map(key => (
              <th key={key}>{key}</th>
            ))}
          </tr>
        </thead>
        <tbody>
          <tr>
            {Object.values(volumePredictions).map((value, index) => (
              <td key={index}>{value}</td>
            ))}
          </tr>
        </tbody>
      </table>
    );
  };

  const renderScatterPlot = () => {
    if (!uploadedData || !uploadedData.predictions_scatter) {
      return null;
    }

    const scatterData = uploadedData.predictions_scatter;

    console.log('Scatter Data:', scatterData); // Verifica el formato

    return (
      <div className="scatter-plot-container">
        <ScatterPlot data={scatterData} />
      </div>
    );
  };

  const renderProfitTable = () => {
    if (!uploadedData || !uploadedData.profit_table) {
      return <p>No profit table data available.</p>;
    }

    let profitTable;
    try {
      // Convertir profit_table a un objeto usando JSON.parse
      profitTable = JSON.parse(uploadedData.profit_table);
    } catch (error) {
      console.error('Error parsing profit table data:', error);
      return <p>Error parsing profit table data.</p>;
    }

    // Asegúrate de que cada valor se maneje correctamente y se convierta en cadena de texto cuando sea necesario
    const averageMean = profitTable.average_mean ? profitTable.average_mean.toFixed(2) : 'N/A';
    const confidenceIntervalInferior = profitTable.intervalo_confianza?.inferior ? profitTable.intervalo_confianza.inferior.toFixed(2) : 'N/A';
    const confidenceIntervalSuperior = profitTable.intervalo_confianza?.superior ? profitTable.intervalo_confianza.superior.toFixed(2) : 'N/A';
    const percentageLosses = typeof profitTable.porcentaje_perdidas === 'string'
    ? profitTable.porcentaje_perdidas.substring(0, 4) // O usa `slice(0, 4)`
    : 'N/A';


    return (
      <table className="styled-table">
        <thead>
          <tr>
            <th>Metric</th>
            <th>Value</th>
          </tr>
        </thead>
        <tbody>
          <tr>
            <td>Average Mean</td>
            <td>${averageMean}</td>
          </tr>
          <tr>
            <td>Confidence Interval Inferior</td>
            <td>${confidenceIntervalInferior}</td>
          </tr>
          <tr>
            <td>Confidence Interval Superior</td>
            <td>${confidenceIntervalSuperior}</td>
          </tr>
          <tr>
            <td>% Risk</td>
            <td>{percentageLosses}</td>
          </tr>
        </tbody>
      </table>
    );
  };

  return (
    <div className="upload-form">
      <div className="banner">
        <div className="banner-left">
          <h1 className="title">OilyGiant: Where to Invest in Wells?</h1>
          <p className="subtitle">Let's upload the data!</p>
        </div>
        <div className="banner-right">
          <p className="description">
            The mining company OilyGiant wants to find the best location for a new oil well. Steps to choose the location:
            <ul>
              <li>Collect parameters of oil wells in the selected region: oil quality and reserve volume.</li>
              <li>Build a model to predict reserve volume in new wells.</li>
              <li>Select oil wells with the highest estimated values.</li>
              <li>Choose the region with the highest total profit for the selected oil wells.</li>
              <li>Analyze potential benefits and risks using the bootstrapping technique.</li>
            </ul>
          </p>
        </div>
      </div>
      <div className="form-container">
        <label className="file-label">Load your Region 1 CSV file:</label>
        <input type="file" onChange={handleFileChange} />
        <button onClick={handleUpload} disabled={loading}>
          {loading ? 'Uploading...' : 'Upload'}
        </button>
      </div>
      
      {loading && <p>Loading data...</p>}
      
      {uploadedData && !loading && (
        <div className="data-analysis-section">
          <h2 className="section-title">Dataset Analysis Overview</h2>
          <div className="tables-container">
            <div className="table-wrapper">
              <h4>Data Head: First 5 Rows of the Dataset</h4>
              {renderTable(uploadedData.data)}
            </div>
            <div className="table-wrapper">
              <h4>General Summary: Overview of Dataset Statistics</h4>
              {renderGeneralSummary()}
            </div>
          </div>
          <h4>Column Summary: Detailed Column-Wise Statistics</h4>
          {renderColumnSummary()}
          <h2 className="section-title">Data Visualization</h2>
          <div className="charts-container">
            <DataVisualization
              histogramData={uploadedData.histogram_data}
              boxplotData={uploadedData.boxplot_data}
              heatmapData={uploadedData.heatmap_data}
            />
          </div>
          <h2 className="section-title">Volume Predictions</h2>
          {renderVolumePredictions()}
          <h2 className="section-title">Predictions Scatter Plot</h2>
          {renderScatterPlot()}
          <h2 className="section-title">Profit Table</h2>
          {renderProfitTable()}
        </div>
      )}
    </div>
  );
};

export default UploadForm;
