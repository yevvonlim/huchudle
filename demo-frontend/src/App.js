// import React, { useState, useRef } from 'react';
// import './App.css';

// import Landmark from './Landmark';
// import MultilineTextFields from './MultilineTextFields';
// import styled from 'styled-components';
// import Button from 'react-bootstrap/Button';
// import 'bootstrap/dist/css/bootstrap.min.css';
// import axios from 'axios';

// const RoundedContainer = styled.div`
//   border-radius: 10px;
//   background-color: #F5F6FA;
//   padding: 20px;
//   margin: 20px;
//   display: flex;
//   flex-direction: column;
//   justify-content: center;
//   align-items: center;
// `;

// function App() {
//   const [textInput, setTextInput] = useState('');
//   const [finalPositions, setFinalPositions] = useState([]);

//   const handleTextInputChange = (text) => {
//     setTextInput(text);
//   };

//   const handleFinalPositionsChange = (positions) => {
//     setFinalPositions(positions);
//   };

//   const handleGenerateImage = async () => {
//     const payload = {
//       caption: textInput,
//       landmark: finalPositions.map(position => [position.x, position.y])
//     };


//     try {
//       const response = await axios.post('http://localhost:8000/postwelltest', payload);
//       console.log('Data sent to server successfully:', response.data);
//     } catch (error) {
//       console.error('Error sending data to server:', error);
//     }
//   };
  

//   return (
//     <div className="container-home">
//       <div className="container-section">
//         <RoundedContainer>
//           <h3>Custom Image Generation</h3>
//           <p>Move facial landmark by drag</p>
//           <Landmark onFinalPositionsChange={handleFinalPositionsChange} />
//           <p>Describe the Face you want to generate</p>
//           <MultilineTextFields onTextInputChange={handleTextInputChange} />
//           <Button variant="dark" onClick={handleGenerateImage}>Generate Image</Button>
          
//         </RoundedContainer>
//       </div>
//       <div className="container-mdA">
//         <RoundedContainer>
//           <h3>Image Generation Result</h3>
//           <img src="https://via.placeholder.com/256" alt="placeholder" />
//         </RoundedContainer>
//       </div>
//     </div>
//   );
// }

// export default App;



// src/App.js

import React from 'react';
import 'bootstrap/dist/css/bootstrap.min.css';
import './App.css';
import MainContent from './MainContent';
import Footer from './Footer';

function App() {
  return (
    <div>
      <header className="site-header sticky-top py-1">
        <nav className="container d-flex flex-column flex-md-row justify-content-between">
          <a className="py-2" href="#" aria-label="Product">
            <svg xmlns="http://www.w3.org/2000/svg" width="24" height="24" fill="none" stroke="currentColor" strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" className="d-block mx-auto" role="img" viewBox="0 0 24 24">
              <title>Product</title>
              <circle cx="12" cy="12" r="10" />
              <path d="M14.31 8l5.74 9.94M9.69 8h11.48M7.38 12l5.74-9.94M9.69 16L3.95 6.06M14.31 16H2.83m13.79-4l-5.74 9.94" />
            </svg>
          </a>
          <a className="py-2 d-none d-md-inline-block" href="#">Tour</a>
          <a className="py-2 d-none d-md-inline-block" href="#">Product</a>
          <a className="py-2 d-none d-md-inline-block" href="#">Features</a>
          <a className="py-2 d-none d-md-inline-block" href="#">Enterprise</a>
          <a className="py-2 d-none d-md-inline-block" href="#">Support</a>
          <a className="py-2 d-none d-md-inline-block" href="#">Pricing</a>
          <a className="py-2 d-none d-md-inline-block" href="#">Cart</a>
        </nav>
      </header>
      <main>
        <MainContent />
      </main>
      <footer className="container py-5">
        <Footer />
      </footer>
    </div>
  );
}

export default App;
