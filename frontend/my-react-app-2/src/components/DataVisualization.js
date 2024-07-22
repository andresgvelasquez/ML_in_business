import React from 'react';
import { Bar } from 'react-chartjs-2';
import BoxplotChart from './BoxplotChart';
import Heatmap from './Heatmap';
import './UploadForm.css';

const DataVisualization = ({ histogramData, boxplotData, heatmapData }) => {
  let processedHeatmapData;
  if (typeof heatmapData === 'string') {
    try {
      processedHeatmapData = JSON.parse(heatmapData);
    } catch (e) {
      console.error('Error parsing heatmap data:', e);
      return <p>Error parsing heatmap data.</p>;
    }
  } else {
    processedHeatmapData = heatmapData;
  }

  const features = Object.keys(processedHeatmapData);
  const heatmapDataArray = features.flatMap((feature, x) =>
    features.map((otherFeature, y) => ({
      x: feature,
      y: otherFeature,
      value: processedHeatmapData[feature][otherFeature]
    }))
  );

  if (!histogramData) {
    return <p>No histogram data available.</p>;
  }

  let binCounts;
  if (typeof histogramData === 'string') {
    try {
      binCounts = JSON.parse(histogramData);
    } catch (e) {
      return <p>Error parsing histogram data.</p>;
    }
  } else {
    binCounts = histogramData;
  }

  const labels = binCounts.map(item => `${item.Bin.left.toFixed(2)}-${item.Bin.right.toFixed(2)}`);
  const dataValues = binCounts.map(item => item.Count);

  const histogramChartData = {
    labels: labels,
    datasets: [
      {
        label: 'Frequency',
        data: dataValues,
        backgroundColor: 'rgba(75, 192, 192, 0.2)',
        borderColor: 'rgba(75, 192, 192, 1)',
        borderWidth: 1,
      },
    ],
  };

  return (
    <div className="charts-container">
      <div className="chart">
        <h4 className="section-title">Product Distribution</h4>
        <div className="bar-chart-wrapper">
          <Bar
            data={histogramChartData}
            options={{
              responsive: true,
              maintainAspectRatio: false,
              plugins: {
                legend: {
                  display: false,
                },
                tooltip: {
                  callbacks: {
                    label: function (context) {
                      return `Category: ${context.label}, Frequency: ${context.raw}`;
                    },
                  },
                },
              },
              scales: {
                x: {
                  type: 'category',
                  title: {
                    display: true,
                    text: 'Product [Thousands of barrels]',
                  },
                  ticks: {
                    autoSkip: false,
                    maxRotation: 45,
                    minRotation: 45,
                    callback: function(value) {
                      return value.length > 10 ? value.slice(0, 10) + '...' : value;
                    }
                  },
                  grid: {
                    display: false,
                  },
                },
                y: {
                  type: 'linear',
                  title: {
                    display: true,
                    text: 'Frequency',
                  },
                  beginAtZero: true,
                  ticks: {
                    callback: function(value) {
                      return value.toLocaleString();
                    }
                  },
                  suggestedMax: Math.max(...dataValues) * 1.1,
                  min: 0,
                  maxTicksLimit: 10,
                },
              },
              layout: {
                padding: {
                  bottom: 20,
                },
              },
            }}
          />
        </div>
      </div>

      <div className="chart">
        <h4 className="section-title">Boxplot of Products</h4>
        <BoxplotChart data={boxplotData} />
      </div>

      <div className="chart">
        <h4 className="section-title">Heatmap</h4>
        <div className="heatmap-container">
          <Heatmap data={heatmapDataArray} />
        </div>
      </div>
    </div>
  );
};

export default DataVisualization;