import React, { useState } from 'react';
import '../App.css';
import './Homepage.css';
import Landmark from '../components/Landmark';
import MultilineTextFields from '../components/MultilineTextFields';
import Button from 'react-bootstrap/Button';
import 'bootstrap/dist/css/bootstrap.min.css';
import axios from 'axios';
import placeholderimg from '../assets/256.png';
import styled from 'styled-components';

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
const BlackBackground = styled.div`
  background-color: #212121;
`;

function Homepage() {
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
        const response = await axios.post('http://localhost:8000/sample', payload, {
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
      <BlackBackground>
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
        <div className="container-section">
          <RoundedContainer>
            <h3>Image Generation Result</h3>
            <p></p>
            <img src={image} alt="Generated result" />
            <p></p>
          </RoundedContainer>
        </div>
      </div>
      </BlackBackground>
    );
  }
  
  export default Homepage;
  

// import React, { useState } from 'react';
// import '../App.css';
// import './Homepage.css';
// import Landmark from '../components/Landmark';
// import MultilineTextFields from '../components/MultilineTextFields';
// import Button from 'react-bootstrap/Button';
// import 'bootstrap/dist/css/bootstrap.min.css';
// import axios from 'axios';
// import placeholderimg from '../assets/256.png';
// import spinner from '../assets/Spin-0.6s-200px.gif'; // Import the spinner image
// import styled from 'styled-components';

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

// function Homepage() {
//   const [textInput, setTextInput] = useState('');
//   const [finalPositions, setFinalPositions] = useState([]);
//   const [image, setImage] = useState(placeholderimg);
//   const [isLoading, setIsLoading] = useState(false); // Add loading state

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

//     setIsLoading(true); // Set loading to true

//     try {
//       const response = await axios.post('http://localhost:8000/sample', payload, {
//         responseType: 'blob'
//       });
//       console.log('Data sent to server successfully:', response.data);
//       const imageUrl = URL.createObjectURL(response.data);
//       setImage(imageUrl);
//     } catch (error) {
//       console.error('Error sending data to server:', error);
//     } finally {
//       setIsLoading(false); // Set loading to false
//     }
//   };

//   return (
//     <div className="container-home">
//       <div className="container-section">
//         <RoundedContainer>
//           <h3>Custom Image Generation</h3>
//           <p>Move facial landmark by drag</p>
//           <Landmark onFinalPositionsChange={handleFinalPositionsChange} />
//           <p></p>
//           <p>Describe the Face you want to generate</p>
//           <MultilineTextFields onTextInputChange={handleTextInputChange} />
//           <Button variant="dark" onClick={handleGenerateImage}>Generate Image</Button>
//         </RoundedContainer>
//       </div>
//       <div className="container-section">
//         <RoundedContainer>
//           <h3>Image Generation Result</h3>
//           <p></p>
//           {isLoading ? (
//             <img src={spinner} alt="Loading..." /> // Show spinner while loading
//           ) : (
//             <img src={image} alt="Generated result" /> // Show generated image
//           )}
//           <p></p>
//         </RoundedContainer>
//       </div>
//     </div>
//   );
// }

// export default Homepage;
