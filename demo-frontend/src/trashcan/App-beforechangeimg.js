import React, { useState, useRef } from 'react';
import './App.css';

import Landmark from './Landmark';
import MultilineTextFields from './MultilineTextFields';
import styled from 'styled-components';
import Button from 'react-bootstrap/Button';
import 'bootstrap/dist/css/bootstrap.min.css';
import axios from 'axios';

const RoundedContainer = styled.div`
  border-radius: 10px;
  background-color: #F5F6FA;
  padding: 20px;
  margin: 20px;
  display: flex;
  flex-direction: column;
  justify-content: center;
  align-items: center;
`;

function App() {
  const [textInput, setTextInput] = useState('');
  const [finalPositions, setFinalPositions] = useState([]);
  const [imageUrl, setImageUrl] = useState('https://via.placeholder.com/256');

  const handleTextInputChange = (text) => {
    setTextInput(text);
  };

  const handleFinalPositionsChange = (positions) => {
    setFinalPositions(positions);
  };

  const handleGenerateImage = async () => {
    const payload = {
      caption: textInput,
      landmark: finalPositions.map(position => [position.x, position.y])
    };

    try {
      const response = await axios.post('http://localhost:8000/test', payload);
      console.log('Data sent to server successfully:', response.data);
      if (response.data && response.data.img_url) {
        setImageUrl(response.data.img_url);
      }
    } catch (error) {
      console.error('Error sending data to server:', error);
    }
  };

  return (
    <div className="container-home">
      <div className="container-section">
        <RoundedContainer>
          <h3>Custom Image Generation</h3>
          <p>Move facial landmark by drag</p>
          <Landmark onFinalPositionsChange={handleFinalPositionsChange} />
          <p>Describe the Face you want to generate</p>
          <MultilineTextFields onTextInputChange={handleTextInputChange} />
          <Button variant="dark" onClick={handleGenerateImage}>Generate Image</Button>
        </RoundedContainer>
      </div>
      <div className="container-mdA">
        <RoundedContainer>
          <h3>Image Generation Result</h3>
          <img src={imageUrl} alt="Generated result" />
        </RoundedContainer>
      </div>
    </div>
  );
}

export default App;
