import React, { useRef, useEffect } from 'react';
import * as d3 from 'd3';

const BoxplotChart = ({ data }) => {
  const svgRef = useRef(null);

  useEffect(() => {
    if (!data || !data.product) {
      console.error("Data or data.product is undefined");
      return;
    }

    const svg = d3.select(svgRef.current);
    svg.selectAll("*").remove(); // Clear previous content

    const width = svgRef.current.clientWidth;
    const height = svgRef.current.clientHeight;

    // Define margins and dimensions
    const margin = { top: 20, right: 30, bottom: 70, left: 70 }; // Increased bottom margin for xlabel
    const chartWidth = width - margin.left - margin.right;
    const chartHeight = height - margin.top - margin.bottom;

    const g = svg.append("g")
      .attr("transform", `translate(${margin.left},${margin.top})`);

    // Create scales
    const yScale = d3.scaleLinear()
      .domain([data.product.min - 1, data.product.max + 1])
      .nice()
      .range([chartHeight, 0]);

    // Add Y axis
    const yAxis = d3.axisLeft(yScale).tickSize(-chartWidth);
    g.append("g")
      .call(yAxis)
      .append("text")
      .attr("transform", "rotate(-90)")
      .attr("y", -margin.left + 20)
      .attr("x", -chartHeight / 2)
      .attr("dy", "-3em")
      .attr("text-anchor", "middle")
      .text("Thousands of barrels"); // Y-axis label

    // Create a dummy xScale for the X axis (if needed)
    const xScale = d3.scaleBand().domain(["Product"]).range([0, chartWidth]);

    // Add X axis
    const xAxis = d3.axisBottom(xScale);
    g.append("g")
      .attr("transform", `translate(0, ${chartHeight})`)
      .call(xAxis)
      .append("text")
      .attr("y", margin.bottom / 2)
      .attr("x", chartWidth / 2)
      .attr("text-anchor", "middle")
      .text("Product"); // X-axis label

    // Create box plot
    g.append("rect")
      .attr("x", chartWidth / 2 - 15)
      .attr("y", yScale(data.product.Q3))
      .attr("width", 30)
      .attr("height", yScale(data.product.Q1) - yScale(data.product.Q3))
      .attr("fill", "rgba(255, 99, 132, 0.5)")
      .attr("stroke", "rgba(255, 99, 132, 1)");

    g.append("line")
      .attr("x1", chartWidth / 2)
      .attr("x2", chartWidth / 2)
      .attr("y1", yScale(data.product.min))
      .attr("y2", yScale(data.product.max))
      .attr("stroke", "rgba(255, 99, 132, 1)");

    g.append("line")
      .attr("x1", chartWidth / 2 - 15)
      .attr("x2", chartWidth / 2 + 15)
      .attr("y1", yScale(data.product.median))
      .attr("y2", yScale(data.product.median))
      .attr("stroke", "rgba(255, 99, 132, 1)");

    // Add outliers
    g.selectAll("circle")
      .data(data.product.outliers || []) // Ensure outliers is defined
      .enter().append("circle")
      .attr("cx", chartWidth / 2)
      .attr("cy", d => yScale(d))
      .attr("r", 3)
      .attr("fill", "#999999");

  }, [data]);

  return (
    <svg ref={svgRef} width="100%" height="400px"></svg>
  );
};

export default BoxplotChart;