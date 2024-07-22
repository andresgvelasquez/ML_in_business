import React, { useEffect, useRef } from 'react';
import * as d3 from 'd3';

const Heatmap = ({ data }) => {
  const svgRef = useRef(null);

  useEffect(() => {
    const svg = d3.select(svgRef.current);
    const margin = { top: 60, right: 60, bottom: 80, left: 80 };
    const cellSize = 50;
    const width = 6 * cellSize + margin.left + margin.right;
    const height = 6 * cellSize + margin.top + margin.bottom;

    // Limpia el contenido del SVG
    svg.selectAll('*').remove();

    // Procesa los datos para obtener etiquetas y valores
    const xLabels = Array.from(new Set(data.map(d => d.x)));
    const yLabels = Array.from(new Set(data.map(d => d.y)));

    // Configura escalas y ejes
    const xScale = d3.scaleBand()
      .domain(xLabels)
      .range([0, width - margin.left - margin.right])
      .padding(0.05);

    const yScale = d3.scaleBand()
      .domain(yLabels)
      .range([height - margin.top - margin.bottom, 0])
      .padding(0.05);

    const colorScale = d3.scaleSequential(d3.interpolateViridis)
      .domain([d3.min(data, d => d.value), d3.max(data, d => d.value)]);

    // Dibuja las celdas del heatmap
    svg.selectAll('rect')
      .data(data)
      .enter()
      .append('rect')
      .attr('x', d => xScale(d.x) + margin.left)
      .attr('y', d => yScale(d.y) + margin.top)
      .attr('width', xScale.bandwidth())
      .attr('height', yScale.bandwidth())
      .style('fill', d => colorScale(d.value))
      .style('stroke', '#fff');

    // Agrega etiquetas de texto
    svg.selectAll('text')
      .data(data)
      .enter()
      .append('text')
      .attr('x', d => xScale(d.x) + margin.left + xScale.bandwidth() / 2)
      .attr('y', d => yScale(d.y) + margin.top + yScale.bandwidth() / 2)
      .attr('dy', '.35em')
      .attr('text-anchor', 'middle')
      .text(d => d.value.toFixed(2))
      .style('fill', '#000')
      .style('font-size', '12px');

    // Agrega ejes X e Y
    svg.append('g')
      .attr('transform', `translate(${margin.left},${height - margin.bottom})`)
      .call(d3.axisBottom(xScale))
      .attr('class', 'x-axis');

    svg.append('g')
      .attr('transform', `translate(${margin.left},${margin.top})`)
      .call(d3.axisLeft(yScale))
      .attr('class', 'y-axis');

  }, [data]);

  return (
    <div className="heatmap-container">
      <svg ref={svgRef} className="heatmap-graph" viewBox={`0 0 ${6 * 50 + 80 + 60} ${6 * 50 + 60 + 80}`}></svg>
    </div>
  );
};

export default Heatmap;