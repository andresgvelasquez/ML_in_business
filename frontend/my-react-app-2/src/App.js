import React from 'react';
import HomePage from './components/Home';

import { Chart as ChartJS, CategoryScale, LinearScale, BarElement, Title, Tooltip, Legend } from 'chart.js';

// Registrar los componentes necesarios
ChartJS.register(CategoryScale, LinearScale, BarElement, Title, Tooltip, Legend);

const App = () => {
  return (
    <div className="App">
      <HomePage />
    </div>
  );
};

export default App;
