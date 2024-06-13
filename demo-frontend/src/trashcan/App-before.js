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
//   const [text, setText] = useState('');
//   const [responseMessage, setResponseMessage] = useState('');
//   const landmarkRef = useRef();

//   const handleTextChange = (newText) => {
//     setText(newText);
//   };

//   const handleGetRequest = () => {
//     axios.get('http://localhost:8000/testparam', {
//             // 쿼리 파라미터 설정
//             params: {
//                 url: 'yoojeong'
//             }
//         })
//       .then(response => {
//         console.log('GET request successful:', response.data);
//       })
//       .catch(error => {
//         console.error('Error making GET request:', error);
//       });
//   };

//   return (
//     <div className="container-home">
//       <div className="container-section">
//         <RoundedContainer>
//           <h3>Custom Image Generation</h3>
//           <p>Move facial landmark by drag</p>
//           <Landmark ref={landmarkRef} />
//           <p>Describe the Face you want to generate</p>
//           <MultilineTextFields onTextChange={handleTextChange} />
//           {/* <Button variant="dark" onClick={handleGenerateImage}>Generate Image</Button> */}
//         </RoundedContainer>
//       </div>
//       <div className="container-mdA">
//         <RoundedContainer>
//           <h3>Image Generation Result</h3>
//           <img src="https://via.placeholder.com/256" alt="placeholder" />
//           <Button variant="dark" onClick={handleGetRequest}>Make GET Request</Button>
//           {responseMessage && <p>Response: {responseMessage}</p>}
//         </RoundedContainer>
//       </div>
//     </div>
//   );
// }

// export default App;
// src/App.js
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
  const [text, setText] = useState('');
  const landmarkRef = useRef();

  const handleTextChange = (newText) => {
    setText(newText);
  };

  const handleGenerateImage = () => {
    if (landmarkRef.current) {
      landmarkRef.current.saveAndSendPositions();
    }

    axios.post('https://8000/test', {
      text
    })
    .then(response => {
      console.log('Text sent to server successfully:', response.data);
    })
    .catch(error => {
      console.error('Error sending text to server:', error);
    });
  };

  return (
    <div className="container-home">
      <div className="container-section">
        <RoundedContainer>
          <h3>Custom Image Generation</h3>
          <p>Move facial landmark by drag</p>
          <Landmark ref={landmarkRef} />
          <p>Describe the Face you want to generate</p>
          <MultilineTextFields onTextChange={handleTextChange} />
          <Button variant="dark" onClick={handleGenerateImage}>Generate Image</Button>
        </RoundedContainer>
      </div>
      <div className="container-mdA">
        <RoundedContainer>
          <h3>Image Generation Result</h3>
          <img src="https://via.placeholder.com/256" alt="placeholder" />
        </RoundedContainer>
      </div>
    </div>
  );
}

export default App;
