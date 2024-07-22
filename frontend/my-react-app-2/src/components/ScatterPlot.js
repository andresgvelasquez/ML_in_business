import React, { useEffect } from 'react';
import * as d3 from 'd3';

const ScatterPlot = ({ data }) => {
  useEffect(() => {
    console.log('Datos recibidos:', data);

    // Procesar datos JSON si se recibe como cadena
    let processedData;
    if (typeof data === 'string') {
      try {
        processedData = JSON.parse(data);
      } catch (e) {
        console.error('Error al analizar datos:', e);
        return;
      }
    } else {
      processedData = data;
    }

    // Verificar el formato de datos
    if (!processedData || !Array.isArray(processedData.datasets) || processedData.datasets.length === 0 || !Array.isArray(processedData.datasets[0].data)) {
      console.log('Datos no válidos o vacíos');
      return;
    }

    // Procesar el dataset
    const dataset = processedData.datasets[0].data;
    console.log('Datos del dataset:', dataset);

    // Inicializar el SVG
    const svg = d3.select('#scatter-plot');
    svg.selectAll('*').remove(); // Limpia el SVG antes de redibujar

    const margin = { top: 20, right: 30, bottom: 40, left: 40 };
    const width = svg.node().clientWidth - margin.left - margin.right;
    const height = svg.node().clientHeight - margin.top - margin.bottom;

    const g = svg.append('g')
      .attr('transform', `translate(${margin.left},${margin.top})`);

    // Calcular el dominio para los ejes X e Y
    const xExtent = d3.extent(dataset, d => d.x);
    const yExtent = d3.extent(dataset, d => d.y);
    console.log('Dominio X:', xExtent);
    console.log('Dominio Y:', yExtent);

    const x = d3.scaleLinear()
      .domain(xExtent).nice()
      .range([0, width]);

    const y = d3.scaleLinear()
      .domain(yExtent).nice()
      .range([height, 0]);

    // Añadir los ejes
    g.append('g')
      .attr('transform', `translate(0,${height})`)
      .call(d3.axisBottom(x));

    g.append('g')
      .call(d3.axisLeft(y));

    // Dibujar los puntos
    g.append('g')
      .selectAll('circle')
      .data(dataset)
      .enter()
      .append('circle')
      .attr('cx', d => x(d.x))
      .attr('cy', d => y(d.y))
      .attr('r', 3)
      .style('fill', '#69b3a2');

    // Dibujar la línea x = y
    g.append('line')
      .attr('x1', x(0))
      .attr('y1', y(0))
      .attr('x2', x(Math.min(xExtent[1], yExtent[1])))
      .attr('y2', y(Math.min(xExtent[1], yExtent[1])))
      .attr('stroke', 'red')
      .attr('stroke-width', 2)
      .attr('stroke-dasharray', '4 2');

  }, [data]);

  return (
    <svg id="scatter-plot" style={{ width: '100%', height: '400px', border: '1px solid #ddd' }}></svg>
  );
};

export default ScatterPlot;