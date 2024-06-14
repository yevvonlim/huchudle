// // // export default Editingpage;
// // import React, { useState } from 'react';
// // import '../App.css';
// // import './Homepage.css';
// // import Landmark from '../components/Landmark';
// // import MultilineTextFields from '../components/MultilineTextFields';
// // import Button from 'react-bootstrap/Button';
// // import 'bootstrap/dist/css/bootstrap.min.css';
// // import axios from 'axios';
// // import placeholderimg from '../assets/256.png';
// // import styled from 'styled-components';

// // const RoundedContainer = styled.div`
// //   border-radius: 10px;
// //   background-color: #F5F6FA;
// //   padding: 20px;
// //   margin: 20px;
// //   display: flex;
// //   flex-direction: column;
// //   justify-content: center;
// //   align-items: center;
// // `;
// // const BlackBackground = styled.div`
// //   background-color: #212121;
// // `;

// // function Editingpage() {
// //     const [textInput, setTextInput] = useState('');
// //     const [finalPositions, setFinalPositions] = useState([]);
// //     const [image, setImage] = useState(placeholderimg);
// //     const [imageBase64, setImageBase64] = useState('');

// //     const handleTextInputChange = (text) => {
// //       setTextInput(text);
// //     };

// //     const handleFinalPositionsChange = (positions) => {
// //       setFinalPositions(positions);
// //     };

// //     const handleImageChange = (base64Image) => {
// //       setImageBase64(base64Image);
// //     };

// //     const handleGenerateImage = async () => {
// //       const payload = {
// //         image_base64: imageBase64.split(',')[1], // Remove the prefix
// //         caption: textInput,
// //         landmark: finalPositions.map(position => [position.x, position.y]),
// //       };
// //       console.log('Sending data to server:', payload);

// //       try {
// //         const response = await axios.post('http://localhost:8000/editlandmark', payload, {
// //           responseType: 'blob'
// //         });
// //         console.log('Data sent to server successfully:', response.data);
// //         const imageUrl = URL.createObjectURL(response.data);
// //         setImage(imageUrl);
// //       } catch (error) {
// //         console.error('Error sending data to server:', error);
// //       }
// //     };

// //     return (
// //       <BlackBackground>
// //         <div className="container-home">
// //           <div className="container-section">
// //             <RoundedContainer>
// //               <h3>Custom Image Editing</h3>
// //               <h4><b>MasaCtrl</b></h4>
// //               <p>Move facial landmark by drag</p>
// //               <Landmark onFinalPositionsChange={handleFinalPositionsChange} onImageChange={handleImageChange} />
// //               <p></p>
// //               <p>Describe the Face you want to Edit</p>
// //               <MultilineTextFields onTextInputChange={handleTextInputChange} />
// //               <Button variant="dark" onClick={handleGenerateImage}>Generate Image</Button>
// //             </RoundedContainer>
// //           </div>
// //           <div className="container-section">
// //             <RoundedContainer>
// //               <h3>Image Editing Result</h3>
// //               <p></p>
// //               <img src={image} alt="Generated result" />
// //               <p></p>
// //             </RoundedContainer>
// //           </div>
// //         </div>
// //       </BlackBackground>
// //     );
// // }

// // export default Editingpage;

// // import React, { useState } from 'react';
// // import '../App.css';
// // import './Homepage.css';
// // import Landmark from '../components/Landmark';
// // import MultilineTextFields from '../components/MultilineTextFields';
// // import Button from 'react-bootstrap/Button';
// // import 'bootstrap/dist/css/bootstrap.min.css';
// // import axios from 'axios';
// // import placeholderimg from '../assets/256.png';
// // import styled from 'styled-components';

// // const RoundedContainer = styled.div`
// //   border-radius: 10px;
// //   background-color: #F5F6FA;
// //   padding: 20px;
// //   margin: 20px;
// //   display: flex;
// //   flex-direction: column;
// //   justify-content: center;
// //   align-items: center;
// // `;
// // const BlackBackground = styled.div`
// //   background-color: #212121;
// // `;

// // function Editingpage() {
// //     const [textInput, setTextInput] = useState('');
// //     const [finalPositions, setFinalPositions] = useState([]);
// //     const [image, setImage] = useState(placeholderimg);
// //     const [imageBase64, setImageBase64] = useState('');

// //     const handleTextInputChange = (text) => {
// //       setTextInput(text);
// //     };

// //     const handleFinalPositionsChange = (positions) => {
// //       setFinalPositions(positions);
// //     };

// //     const handleImageChange = (base64Image) => {
// //       setImageBase64(base64Image);
// //     };

// //     const handleGenerateImage = async () => {
// //       const payload = {
// //         image_base64: imageBase64.split(',')[1], // Remove the prefix
// //         caption: textInput,
// //         landmark: finalPositions.map(position => [position.x, position.y]),
// //       };
// //       console.log('Sending data to server:', payload);

// //       try {
// //         const response = await axios.post('http://localhost:8000/editlandmark', payload, {
// //           responseType: 'blob'
// //         });
// //         console.log('Data sent to server successfully:', response.data);
// //         const imageUrl = URL.createObjectURL(response.data);
// //         setImage(imageUrl);
// //       } catch (error) {
// //         console.error('Error sending data to server:', error);
// //       }
// //     };

// //     return (
// //       <BlackBackground>
// //         <div className="container-home">
// //           <div className="container-section">
// //             <RoundedContainer>
// //               <h3>Custom Image Editing</h3>
// //               <h4><b>MasaCtrl</b></h4>
// //               <p>Move facial landmark by drag</p>
// //               <Landmark onFinalPositionsChange={handleFinalPositionsChange} onImageChange={handleImageChange} />
// //               <p></p>
// //               <p>Describe the Face you want to Edit</p>
// //               <MultilineTextFields onTextInputChange={handleTextInputChange} />
// //               <Button variant="dark" onClick={handleGenerateImage}>Generate Image</Button>
// //             </RoundedContainer>
// //           </div>
// //           <div className="container-section">
// //             <RoundedContainer>
// //               <h3>Image Editing Result</h3>
// //               <p></p>
// //               <img src={image} alt="Generated result" />
// //               <p></p>
// //             </RoundedContainer>
// //           </div>
// //         </div>
// //       </BlackBackground>
// //     );
// // }

// // export default Editingpage;

// // export default Editingpage;


// import React, { useState } from 'react';
// import '../App.css';
// import './Homepage.css';
// import Landmark from '../components/Landmark';
// import MultilineTextFields from '../components/MultilineTextFields';
// import Button from 'react-bootstrap/Button';
// import 'bootstrap/dist/css/bootstrap.min.css';
// import axios from 'axios';
// import placeholderimg from '../assets/256.png';
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
// const BlackBackground = styled.div`
//   background-color: #212121;
// `;

// function Editingpage() {
//     const [textInput, setTextInput] = useState('');
//     const [finalPositions, setFinalPositions] = useState([]);
//     const [image, setImage] = useState(placeholderimg);
//     const [imageBase64, setImageBase64] = useState('');

//     const handleTextInputChange = (text) => {
//       setTextInput(text);
//     };

//     const handleFinalPositionsChange = (positions) => {
//       setFinalPositions(positions);
//     };

//     const handleImageChange = (base64Image) => {
//       setImageBase64(base64Image);
//     };

//     const handleGenerateImage = async () => {
//       const payload = {
//         image_base64: imageBase64.split(',')[1], // Remove the prefix
//         caption: textInput,
//         landmark: finalPositions.map(position => [position.x, position.y]),
//       };
//       console.log('Sending data to server:', payload);

//       try {
//         const response = await axios.post('http://localhost:8000/editlandmark', payload, {
//           responseType: 'blob'
//         });
//         console.log('Data sent to server successfully:', response.data);
//         const imageUrl = URL.createObjectURL(response.data);
//         setImage(imageUrl);
//       } catch (error) {
//         console.error('Error sending data to server:', error);
//       }
//     };

//     return (
//       <BlackBackground>
//         <div className="container-home">
//           <div className="container-section">
//             <RoundedContainer>
//               <h3>Custom Image Editing</h3>
//               <h4><b>MasaCtrl</b></h4>
//               <p>Move facial landmark by drag</p>
//               <Landmark onFinalPositionsChange={handleFinalPositionsChange} onImageChange={handleImageChange} />
//               <p></p>
//               <p>Describe the Face you want to Edit</p>
//               <MultilineTextFields onTextInputChange={handleTextInputChange} />
//               <Button variant="dark" onClick={handleGenerateImage}>Generate Image</Button>
//             </RoundedContainer>
//           </div>
//           <div className="container-section">
//             <RoundedContainer>
//               <h3>Image Editing Result</h3>
//               <p></p>
//               <img src={image} alt="Generated result" />
//               <p></p>
//             </RoundedContainer>
//           </div>
//         </div>
//       </BlackBackground>
//     );
// }

// export default Editingpage;

// import React, { useState } from 'react';
// import '../App.css';
// import './Homepage.css';
// import Landmark from '../components/Landmark';
// import MultilineTextFields from '../components/MultilineTextFields';
// import Button from 'react-bootstrap/Button';
// import 'bootstrap/dist/css/bootstrap.min.css';
// import axios from 'axios';
// import placeholderimg from '../assets/256.png';
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
// const BlackBackground = styled.div`
//   background-color: #212121;
// `;

// function Editingpage() {
//   const [textInput, setTextInput] = useState('');
//   const [finalPositions, setFinalPositions] = useState([]);
//   const [image, setImage] = useState(placeholderimg);
//   const [imageBase64, setImageBase64] = useState('');

//   const handleTextInputChange = (text) => {
//     setTextInput(text);
//   };

//   const handleFinalPositionsChange = (positions) => {
//     setFinalPositions(positions);
//   };

//   const handleImageChange = (base64Image) => {
//     setImageBase64(base64Image);
//   };

//   const handleGenerateImage = async () => {
//     const payload = {
//       image_base64: imageBase64.split(',')[1], // Remove the prefix
//       caption: textInput,
//       landmark: finalPositions.map(position => [position.x, position.y]),
//     };
//     console.log('Sending data to server:', payload);

//     try {
//       const response = await axios.post('http://localhost:8000/editlandmark', payload, {
//         responseType: 'blob'
//       });
//       console.log('Data sent to server successfully:', response.data);
//       const imageUrl = URL.createObjectURL(response.data);
//       setImage(imageUrl);
//     } catch (error) {
//       console.error('Error sending data to server:', error);
//     }
//   };

//   return (
//     <BlackBackground>
//       <div className="container-home">
//         <div className="container-section">
//           <RoundedContainer>
//             <h3>Custom Image Editing</h3>
//             <h4><b>MasaCtrl</b></h4>
//             <p>Move facial landmark by drag</p>
//             <Landmark onFinalPositionsChange={handleFinalPositionsChange} onImageChange={handleImageChange} />
//             <p></p>
//             <p>Describe the Face you want to Edit</p>
//             <MultilineTextFields onTextInputChange={handleTextInputChange} />
//             <Button variant="dark" onClick={handleGenerateImage}>Generate Image</Button>
//           </RoundedContainer>
//         </div>
//         <div className="container-section">
//           <RoundedContainer>
//             <h3>Image Editing Result</h3>
//             <p></p>
//             <img src={image} alt="Generated result" />
//             <p></p>
//           </RoundedContainer>
//         </div>
//       </div>
//     </BlackBackground>
//   );
// // }
// import React, { useState, useRef, useEffect } from 'react';
// import '../App.css';
// import './Homepage.css';
// import Landmark from '../components/Landmark';
// import MultilineTextFields from '../components/MultilineTextFields';
// import Button from 'react-bootstrap/Button';
// import 'bootstrap/dist/css/bootstrap.min.css';
// import axios from 'axios';
// import placeholderimg from '../assets/256.png';
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

// const BlackBackground = styled.div`
//   background-color: #212121;
// `;

// function Editingpage() {
//   const [textInput, setTextInput] = useState('');
//   const [finalPositions, setFinalPositions] = useState([]);
//   const [image, setImage] = useState(placeholderimg);
//   const [imageBase64, setImageBase64] = useState('');

//   const finalPositionsRef = useRef(finalPositions);

//   useEffect(() => {
//     finalPositionsRef.current = finalPositions;
//   }, [finalPositions]);

//   const handleTextInputChange = (text) => {
//     setTextInput(text);
//   };

//   const handleFinalPositionsChange = (positions) => {
//     setFinalPositions(positions);
//   };

//   const handleImageChange = (base64Image) => {
//     setImageBase64(base64Image);
//   };

//   const handleGenerateImage = async () => {
//     // Update the ref with the latest positions
//     const currentPositions = finalPositionsRef.current;

//     const payload = {
//       image_base64: imageBase64.split(',')[1], // Remove the prefix
//       caption: textInput,
//       landmark: currentPositions.map(position => [position.x * 256 / 299, position.y * 256 / 299]),
//     };
//     console.log('Sending data to server:', payload);

//     try {
//       const response = await axios.post('http://localhost:8000/editlandmark', payload, {
//         responseType: 'blob'
//       });
//       console.log('Data sent to server successfully:', response.data);
//       const imageUrl = URL.createObjectURL(response.data);
//       setImage(imageUrl);
//     } catch (error) {
//       console.error('Error sending data to server:', error);
//     }
//   };

//   return (
//     <BlackBackground>
//       <div className="container-home">
//         <div className="container-section">
//           <RoundedContainer>
//             <h3>Custom Image Editing</h3>
//             <h4><b>MasaCtrl</b></h4>
//             <p>Move facial landmark by drag</p>
//             <Landmark onFinalPositionsChange={handleFinalPositionsChange} onImageChange={handleImageChange} />
//             <p></p>
//             <p>Describe the Face you want to Edit</p>
//             <MultilineTextFields onTextInputChange={handleTextInputChange} />
//             <Button variant="dark" onClick={handleGenerateImage}>Generate Image</Button>
//           </RoundedContainer>
//         </div>
//         <div className="container-section">
//           <RoundedContainer>
//             <h3>Image Editing Result</h3>
//             <p></p>
//             <img src={image} alt="Generated result" />
//             <p></p>
//           </RoundedContainer>
//         </div>
//       </div>
//     </BlackBackground>
//   );
// }

// export default Editingpage;
import React, { useState, useRef, useEffect } from 'react';
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

function Editingpage() {
  const [textInput, setTextInput] = useState('');
  const [finalPositions, setFinalPositions] = useState([]);
  const [image, setImage] = useState(placeholderimg);
  const [imageBase64, setImageBase64] = useState('');

  const finalPositionsRef = useRef(finalPositions);

  useEffect(() => {
    finalPositionsRef.current = finalPositions;
  }, [finalPositions]);

  const handleTextInputChange = (text) => {
    setTextInput(text);
  };

  const handleFinalPositionsChange = (positions) => {
    setFinalPositions(positions);
  };

  const handleImageChange = (base64Image) => {
    setImageBase64(base64Image);
  };

  const handleGenerateImage = async () => {
    // Update the ref with the latest positions
    const currentPositions = finalPositionsRef.current;

    const payload = {
      image_base64: imageBase64.split(',')[1], // Remove the prefix
      caption: textInput,
      landmark: currentPositions.map(position => [position.x * 256 / 299, position.y * 256 / 299]),
    };
    console.log('Sending data to server:', payload);

    try {
      const response = await axios.post('http://localhost:8000/editlandmark', payload, {
        responseType: 'blob'
      });
      console.log('Data sent to server successfully:', response.data);
      const imageUrl = URL.createObjectURL(response.data);
      setImage(imageUrl);
    } catch (error) {
      console.error('Error sending data to server:', error);
    }
  };

  useEffect(() => {
    // Trigger re-render when image state changes
    if (image) {
      console.log('Image updated:', image);
    }
  }, [image]);

  return (
    <BlackBackground>
      <div className="container-home">
        <div className="container-section">
          <RoundedContainer>
            <h3>Custom Image Editing</h3>
            <h4><b>MasaCtrl</b></h4>
            <p>Move facial landmark by drag</p>
            <Landmark onFinalPositionsChange={handleFinalPositionsChange} onImageChange={handleImageChange} />
            <p></p>
            <p>Describe the Face you want to Edit</p>
            <MultilineTextFields onTextInputChange={handleTextInputChange} />
            <Button variant="dark" onClick={handleGenerateImage}>Generate Image</Button>
          </RoundedContainer>
        </div>
        <div className="container-section">
          <RoundedContainer>
            <h3>Image Editing Result</h3>
            <p></p>
            <img src={image} alt="Generated result" />
            <p></p>
          </RoundedContainer>
        </div>
      </div>
    </BlackBackground>
  );
}

export default Editingpage;
