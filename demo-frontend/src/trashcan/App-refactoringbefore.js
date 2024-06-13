import React, { useState } from 'react';
import './App.css';

import Landmark from './Landmark';
import MultilineTextFields from './MultilineTextFields';
import styled from 'styled-components';
import Button from 'react-bootstrap/Button';
import 'bootstrap/dist/css/bootstrap.min.css';
import axios from 'axios';
import placeholderimg from './256.png';

import { Container } from "react-bootstrap";
import Sidebar from "./Sidebar";

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
  const [image, setImage] = useState(placeholderimg);

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
      const response = await axios.post('http://localhost:8000/imagetest', payload, {
        responseType: 'blob'
      });
      console.log('Data sent to server successfully:', response.data);
      const imageUrl = URL.createObjectURL(response.data);
      setImage(imageUrl);
    } catch (error) {
      console.error('Error sending data to server:', error);
    }
  };

  return (
    <Container>
      <Sidebar />
    <div className="container-home">
      <div className="container-section">
        <RoundedContainer>
          <h3>Custom Image Generation</h3>
          
          <p>Move facial landmark by drag</p>
          <Landmark onFinalPositionsChange={handleFinalPositionsChange} />
          <p></p>
          <p>Describe the Face you want to generate</p>
          <MultilineTextFields onTextInputChange={handleTextInputChange} />
          <Button variant="dark" onClick={handleGenerateImage}>Generate Image</Button>
        </RoundedContainer>
      </div>
      {/* <div className="container-section">
      <h3>Generate your Image!</h3>
        <p>You can generate human face image whatever you want. Try to change hair color, hair length, eye color, and the gender!</p>
      </div> */}
      <div className="container-section">
        <RoundedContainer>
          <h3>Image Generation Result</h3>
          <p></p>
          <img src={image} alt="Generated result" />
          <p></p>

        </RoundedContainer>
      </div>
    </div>
    </Container>
  );
}

export default App;
