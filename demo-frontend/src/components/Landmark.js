import React, { useState, useRef, useEffect } from 'react';
import axios from 'axios';
import Draggable from 'react-draggable';
import 'bootstrap/dist/css/bootstrap.min.css';



const landmarks = {
  "landmark-0": { "x": 76.8, "y": 120.32 },
  "landmark-1": { "x": 74.24, "y": 133.12 },
  "landmark-2": { "x": 74.24, "y": 148.48 },
  "landmark-3": { "x": 76.8, "y": 163.84 },
  "landmark-4": { "x": 81.92, "y": 179.2 },
  "landmark-5": { "x": 89.6, "y": 194.56 },
  "landmark-6": { "x": 97.28, "y": 207.36 },
  "landmark-7": { "x": 104.96, "y": 220.16 },
  "landmark-8": { "x": 120.32, "y": 222.72 },
  "landmark-9": { "x": 140.8, "y": 222.72 },
  "landmark-10": { "x": 166.4, "y": 220.16 },
  "landmark-11": { "x": 189.44, "y": 212.48 },
  "landmark-12": { "x": 204.8, "y": 199.68 },
  "landmark-13": { "x": 217.6, "y": 181.76 },
  "landmark-14": { "x": 220.16, "y": 158.72 },
  "landmark-15": { "x": 220.16, "y": 133.12 },
  "landmark-16": { "x": 220.16, "y": 112.64 },
  "landmark-17": { "x": 74.24, "y": 92.16 },
  "landmark-18": { "x": 76.8, "y": 87.04 },
  "landmark-19": { "x": 84.48, "y": 87.04 },
  "landmark-20": { "x": 92.16, "y": 87.04 },
  "landmark-21": { "x": 99.84, "y": 89.6 },
  "landmark-22": { "x": 122.88, "y": 87.04 },
  "landmark-23": { "x": 138.24, "y": 81.92 },
  "landmark-24": { "x": 153.6, "y": 81.92 },
  "landmark-25": { "x": 168.96, "y": 84.48 },
  "landmark-26": { "x": 181.76, "y": 92.16 },
  "landmark-27": { "x": 110.08, "y": 104.96 },
  "landmark-28": { "x": 104.96, "y": 120.32 },
  "landmark-29": { "x": 102.4, "y": 133.12 },
  "landmark-30": { "x": 99.84, "y": 145.92 },
  "landmark-31": { "x": 97.28, "y": 156.16 },
  "landmark-32": { "x": 99.84, "y": 158.72 },
  "landmark-33": { "x": 107.52, "y": 161.28 },
  "landmark-34": { "x": 115.2, "y": 158.72 },
  "landmark-35": { "x": 122.88, "y": 156.16 },
  "landmark-36": { "x": 81.92, "y": 110.08 },
  "landmark-37": { "x": 87.04, "y": 104.96 },
  "landmark-38": { "x": 94.72, "y": 104.96 },
  "landmark-39": { "x": 102.4, "y": 110.08 },
  "landmark-40": { "x": 94.72, "y": 115.2 },
  "landmark-41": { "x": 87.04, "y": 115.2 },
  "landmark-42": { "x": 138.24, "y": 110.08 },
  "landmark-43": { "x": 145.92, "y": 102.4 },
  "landmark-44": { "x": 156.16, "y": 102.4 },
  "landmark-45": { "x": 163.84, "y": 107.52 },
  "landmark-46": { "x": 156.16, "y": 112.64 },
  "landmark-47": { "x": 145.92, "y": 112.64 },
  "landmark-48": { "x": 99.84, "y": 184.32 },
  "landmark-49": { "x": 99.84, "y": 179.2 },
  "landmark-50": { "x": 104.96, "y": 174.08 },
  "landmark-51": { "x": 112.64, "y": 176.64 },
  "landmark-52": { "x": 120.32, "y": 174.08 },
  "landmark-53": { "x": 138.24, "y": 179.2 },
  "landmark-54": { "x": 156.16, "y": 184.32 },
  "landmark-55": { "x": 140.8, "y": 197.12 },
  "landmark-56": { "x": 128, "y": 199.68 },
  "landmark-57": { "x": 115.2, "y": 199.68 },
  "landmark-58": { "x": 102.4, "y": 192 },
  "landmark-59": { "x": 99.84, "y": 192 },
  "landmark-60": { "x": 104.96, "y": 184.32 },
  "landmark-61": { "x": 112.64, "y": 181.76 },
  "landmark-62": { "x": 120.32, "y": 181.76 },
  "landmark-63": { "x": 138.24, "y": 181.76 },
  "landmark-64": { "x": 156.16, "y": 184.32 },
  "landmark-65": { "x": 140.8, "y": 192 },
  "landmark-66": { "x": 128, "y": 192 },
  "landmark-67": { "x": 115.2, "y": 192 }
};

const landmarks2 = {
  "landmark-0": { "x": 32.24, "y": 69.12 },
  "landmark-1": { "x": 32.24, "y": 99.84 },
  "landmark-2": { "x": 34.8, "y": 130.56 },
  "landmark-3": { "x": 42.48, "y": 163.84 },
  "landmark-4": { "x": 57.84, "y": 192 },
  "landmark-5": { "x": 70.64, "y": 215.04 },
  "landmark-6": { "x": 91.12, "y": 233.28 },
  "landmark-7": { "x": 119.28, "y": 245.76 },
  "landmark-8": { "x": 147.44, "y": 250.88 },
  "landmark-9": { "x": 178.16, "y": 245.76 },
  "landmark-10": { "x": 204.4, "y": 235.52 },
  "landmark-11": { "x": 224.24, "y": 217.6 },
  "landmark-12": { "x": 239.6, "y": 192 },
  "landmark-13": { "x": 252.4, "y": 166.4 },
  "landmark-14": { "x": 260.08, "y": 133.12 },
  "landmark-15": { "x": 262.64, "y": 99.84 },
  "landmark-16": { "x": 262.64, "y": 71.68 },
  "landmark-17": { "x": 62.96, "y": 69.12 },
  "landmark-18": { "x": 75.76, "y": 61.44 },
  "landmark-19": { "x": 93.68, "y": 56.32 },
  "landmark-20": { "x": 111.6, "y": 61.44 },
  "landmark-21": { "x": 124.4, "y": 69.12 },
  "landmark-22": { "x": 178.16, "y": 69.12 },
  "landmark-23": { "x": 190.96, "y": 61.44 },
  "landmark-24": { "x": 208.88, "y": 56.32 },
  "landmark-25": { "x": 224.24, "y": 61.44 },
  "landmark-26": { "x": 237.04, "y": 69.12 },
  "landmark-27": { "x": 150, "y": 97.28 },
  "landmark-28": { "x": 150, "y": 115.2 },
  "landmark-29": { "x": 150, "y": 133.12 },
  "landmark-30": { "x": 150, "y": 151.04 },
  "landmark-31": { "x": 126.96, "y": 163.84 },
  "landmark-32": { "x": 137.2, "y": 168.96 },
  "landmark-33": { "x": 150, "y": 171.52 },
  "landmark-34": { "x": 162.8, "y": 168.96 },
  "landmark-35": { "x": 173.04, "y": 163.84 },
  "landmark-36": { "x": 68.08, "y": 97.28 },
  "landmark-37": { "x": 86, "y": 87.04 },
  "landmark-38": { "x": 103.92, "y": 87.04 },
  "landmark-39": { "x": 119.28, "y": 97.28 },
  "landmark-40": { "x": 103.92, "y": 107.52 },
  "landmark-41": { "x": 86, "y": 107.52 },
  "landmark-42": { "x": 180.72, "y": 97.28 },
  "landmark-43": { "x": 198.64, "y": 87.04 },
  "landmark-44": { "x": 216.56, "y": 87.04 },
  "landmark-45": { "x": 231.92, "y": 97.28 },
  "landmark-46": { "x": 216.56, "y": 107.52 },
  "landmark-47": { "x": 198.64, "y": 107.52 },
  "landmark-48": { "x": 106.48, "y": 199.68 },
  "landmark-49": { "x": 121.84, "y": 192 },
  "landmark-50": { "x": 137.2, "y": 186.88 },
  "landmark-51": { "x": 152.56, "y": 189.44 },
  "landmark-52": { "x": 167.92, "y": 186.88 },
  "landmark-53": { "x": 183.28, "y": 192 },
  "landmark-54": { "x": 198.64, "y": 199.68 },
  "landmark-55": { "x": 183.28, "y": 212.48 },
  "landmark-56": { "x": 167.92, "y": 217.6  },
  "landmark-57": { "x": 152.56, "y": 222.72 },
  "landmark-58": { "x": 137.2, "y": 217.6 },
  "landmark-59": { "x": 121.84, "y": 212.48 },
  "landmark-60": { "x": 126.96, "y": 199.68 },
  "landmark-61": { "x": 137.2, "y": 194.56 },
  "landmark-62": { "x": 167.92, "y": 194.56 },
  "landmark-63": { "x": 183.28, "y": 194.56 },
  "landmark-64": { "x": 198.64, "y": 199.68 },
  "landmark-65": { "x": 183.28, "y": 209.92 },
  "landmark-66": { "x": 167.92, "y": 212.48 },
  "landmark-67": { "x": 152.56, "y": 209.92 }
};


  
const landmarks3 = {
  "landmark-0": { "x": 0.04, "y": 0.27 },
  "landmark-1": { "x": 0.04, "y": 0.39 },
  "landmark-2": { "x": 0.05, "y": 0.51 },
  "landmark-3": { "x": 0.08, "y": 0.64 },
  "landmark-4": { "x": 0.14, "y": 0.75 },
  "landmark-5": { "x": 0.19, "y": 0.84 },
  "landmark-6": { "x": 0.27, "y": 0.91 },
  "landmark-7": { "x": 0.38, "y": 0.96 },
  "landmark-8": { "x": 0.49, "y": 0.98 },
  "landmark-9": { "x": 0.61, "y": 0.96 },
  "landmark-10": { "x": 0.71, "y": 0.92 },
  "landmark-11": { "x": 0.79, "y": 0.85 },
  "landmark-12": { "x": 0.85, "y": 0.75 },
  "landmark-13": { "x": 0.90, "y": 0.65 },
  "landmark-14": { "x": 0.93, "y": 0.52 },
  "landmark-15": { "x": 0.94, "y": 0.39 },
  "landmark-16": { "x": 0.94, "y": 0.28 },
  "landmark-17": { "x": 0.16, "y": 0.27 },
  "landmark-18": { "x": 0.21, "y": 0.24 },
  "landmark-19": { "x": 0.28, "y": 0.22 },
  "landmark-20": { "x": 0.35, "y": 0.24 },
  "landmark-21": { "x": 0.40, "y": 0.27 },
  "landmark-22": { "x": 0.61, "y": 0.27 },
  "landmark-23": { "x": 0.66, "y": 0.24 },
  "landmark-24": { "x": 0.73, "y": 0.22 },
  "landmark-25": { "x": 0.79, "y": 0.24 },
  "landmark-26": { "x": 0.84, "y": 0.27 },
  "landmark-27": { "x": 0.50, "y": 0.38 },
  "landmark-28": { "x": 0.50, "y": 0.45 },
  "landmark-29": { "x": 0.50, "y": 0.52 },
  "landmark-30": { "x": 0.50, "y": 0.59 },
  "landmark-31": { "x": 0.41, "y": 0.64 },
  "landmark-32": { "x": 0.45, "y": 0.66 },
  "landmark-33": { "x": 0.50, "y": 0.67 },
  "landmark-34": { "x": 0.55, "y": 0.66 },
  "landmark-35": { "x": 0.59, "y": 0.64 },
  "landmark-36": { "x": 0.18, "y": 0.38 },
  "landmark-37": { "x": 0.25, "y": 0.34 },
  "landmark-38": { "x": 0.32, "y": 0.34 },
  "landmark-39": { "x": 0.38, "y": 0.38 },
  "landmark-40": { "x": 0.32, "y": 0.42 },
  "landmark-41": { "x": 0.25, "y": 0.42 },
  "landmark-42": { "x": 0.62, "y": 0.38 },
  "landmark-43": { "x": 0.69, "y": 0.34 },
  "landmark-44": { "x": 0.76, "y": 0.34 },
  "landmark-45": { "x": 0.82, "y": 0.38 },
  "landmark-46": { "x": 0.76, "y": 0.42 },
  "landmark-47": { "x": 0.69, "y": 0.42 },
  "landmark-48": { "x": 0.33, "y": 0.78 },
  "landmark-49": { "x": 0.39, "y": 0.75 },
  "landmark-50": { "x": 0.45, "y": 0.73 },
  "landmark-51": { "x": 0.51, "y": 0.74 },
  "landmark-52": { "x": 0.57, "y": 0.73 },
  "landmark-53": { "x": 0.63, "y": 0.75 },
  "landmark-54": { "x": 0.69, "y": 0.78 },
  "landmark-55": { "x": 0.63, "y": 0.83 },
  "landmark-56": { "x": 0.57, "y": 0.85 },
  "landmark-57": { "x": 0.51, "y": 0.87 },
  "landmark-58": { "x": 0.45, "y": 0.85 },
  "landmark-59": { "x": 0.39, "y": 0.83 },
  "landmark-60": { "x": 0.41, "y": 0.78 },
  "landmark-61": { "x": 0.45, "y": 0.76 },
  "landmark-62": { "x": 0.51, "y": 0.77 },
  "landmark-63": { "x": 0.57, "y": 0.76 },
  "landmark-64": { "x": 0.61, "y": 0.78 },
  "landmark-65": { "x": 0.57, "y": 0.82 },
  "landmark-66": { "x": 0.51, "y": 0.83 },
  "landmark-67": { "x": 0.45, "y": 0.82 }
};



// const Landmark = ({ onFinalPositionsChange, onImageChange }) => {
//   const initialCircles = Object.keys(landmarks).map(key => ({
//     id: key,
//     x: landmarks[key].x,
//     y: landmarks[key].y 
//   }));

//   const [circles, setCircles] = useState(initialCircles);
//   const [hoveredCircle, setHoveredCircle] = useState(null);
//   const [image, setImage] = useState(null);
//   const circleRefs = useRef([]);

//   useEffect(() => {
//     onFinalPositionsChange(circles);
//   }, [circles, onFinalPositionsChange]);

//   const handleStop = (e, data, index) => {
//     const newCircles = [...circles];
//     newCircles[index] = { ...newCircles[index], x: data.x, y: data.y };
//     setCircles(newCircles);
//   };

//   const handleImageUpload = async (event) => {
//     const file = event.target.files[0];
//     if (file) {
//       const reader = new FileReader();
//       reader.onloadend = async () => {
//         const base64Image = reader.result.split(',')[1]; // Get base64 string without the prefix
//         try {
//           const response = await axios.post('http://localhost:8000/pointlandmark', { image_base64: base64Image });
//           if (response.data && response.data.isSuccess) {
//             console.log(response.data);
//             const newLandmarks = response.data.landmark.map((point, index) => ({
//               id: index.toString(),
//               x: point[0] * 299 / 256,
//               y: point[1] * 299 / 256
//             }));
//             setCircles(newLandmarks);
//             setImage(response.data.resized_image); // Set the uploaded image as the background without prefix
//             onImageChange(response.data.resized_image); // Pass the image data to the parent component
//           }
//         } catch (error) {
//           console.error('Error uploading image and fetching landmarks:', error);
//         }
//       };
//       reader.readAsDataURL(file); // Read the file as a data URL
//     }
//   };

//   return (
//     <div style={{ display: 'flex', flexDirection: 'column', alignItems: 'center' }}>
//       <div style={{
//         position: 'relative',
//         width: '300px',
//         height: '300px',
//         border: '1px solid #BCBDC0',
//         borderRadius: '10px',
//         overflow: 'hidden',
//         backgroundSize: 'cover',
//         backgroundPosition: 'center',
//         backgroundImage: image ? `url(data:image/jpeg;base64,${image})` : 'none', // Add prefix here
//         margin: '20px 0'
//       }}>
//         {circles.map((circle, index) => (
//           <Draggable
//             key={circle.id}
//             defaultPosition={{ x: circle.x, y: circle.y }}
//             onStop={(e, data) => handleStop(e, data, index)}
//           >
//             <div
//               ref={el => circleRefs.current[index] = el}
//               style={{
//                 position: 'absolute',
//                 width: '10px',
//                 height: '10px',
//                 borderRadius: '50%',
//                 backgroundColor: hoveredCircle === index ? '#545454' : '#BCBDC0',
//                 cursor: 'pointer'
//               }}
//               onMouseEnter={() => setHoveredCircle(index)}
//               onMouseLeave={() => setHoveredCircle(null)}
//             />
//           </Draggable>
//         ))}
//       </div>
//       <div>
//         <input
//           type="file"
//           className="form-control"
//           id="imageUpload"
//           accept="image/*"
//           onChange={handleImageUpload}
//           style={{ width: '300px' }}
//         />
//       </div>
//     </div>
//   );
// };

// export default Landmark;


// const Landmark = ({ onFinalPositionsChange, onImageChange }) => {
//   const initialCircles = Object.keys(landmarks).map(key => ({
//     id: key,
//     x: landmarks[key].x,
//     y: landmarks[key].y 
//   }));

//   const [circles, setCircles] = useState(initialCircles);
//   const [hoveredCircle, setHoveredCircle] = useState(null);
//   const [backgroundImage, setBackgroundImage] = useState(null); // Renamed for clarity
//   const [originalImage, setOriginalImage] = useState(null);
//   const circleRefs = useRef([]);

//   useEffect(() => {
//     onFinalPositionsChange(circles);
//   }, [circles, onFinalPositionsChange]);

//   const handleStop = (e, data, index) => {
//     const newCircles = [...circles];
//     newCircles[index] = { ...newCircles[index], x: data.x, y: data.y };
//     setCircles(newCircles);
//   };

//   const handleImageUpload = async (event) => {
//     const file = event.target.files[0];
//     if (file) {
//       const reader = new FileReader();
//       reader.onloadend = async () => {
//         const base64Image = reader.result.split(',')[1]; // Get base64 string without the prefix
//         setOriginalImage(reader.result); // Store the original image
//         try {
//           const response = await axios.post('http://localhost:8000/pointlandmark', { image_base64: base64Image });
//           if (response.data && response.data.isSuccess) {
//             const newLandmarks = response.data.landmark.map((point, index) => ({
//               id: index.toString(),
//               x: point[0] * 299 / 256,
//               y: point[1] * 299 / 256
//             }));
//             setCircles(newLandmarks);
//             setBackgroundImage(`data:image/jpeg;base64,${response.data.resized_image}`); // Set the resized image
//             onImageChange(reader.result); // Pass the original image data to the parent component
//           }
//         } catch (error) {
//           console.error('Error uploading image and fetching landmarks:', error);
//         }
//       };
//       reader.readAsDataURL(file); // Read the file as a data URL
//     }
//   };

//   return (
//     <div style={{ display: 'flex', flexDirection: 'column', alignItems: 'center' }}>
//       <div style={{
//         position: 'relative',
//         width: '300px',
//         height: '300px',
//         border: '1px solid #BCBDC0',
//         borderRadius: '10px',
//         overflow: 'hidden',
//         backgroundSize: 'cover',
//         backgroundPosition: 'center',
//         backgroundImage: backgroundImage ? `url(${backgroundImage})` : 'none',
//         margin: '20px 0'
//       }}>
//         {circles.map((circle, index) => (
//           <Draggable
//             key={circle.id}
//             defaultPosition={{ x: circle.x, y: circle.y }}
//             onStop={(e, data) => handleStop(e, data, index)}
//           >
//             <div
//               ref={el => circleRefs.current[index] = el}
//               style={{
//                 position: 'absolute',
//                 width: '10px',
//                 height: '10px',
//                 borderRadius: '50%',
//                 backgroundColor: hoveredCircle === index ? '#545454' : '#BCBDC0',
//                 cursor: 'pointer'
//               }}
//               onMouseEnter={() => setHoveredCircle(index)}
//               onMouseLeave={() => setHoveredCircle(null)}
//             />
//           </Draggable>
//         ))}
//       </div>
//       <div>
//         <input
//           type="file"
//           className="form-control"
//           id="imageUpload"
//           accept="image/*"
//           onChange={handleImageUpload}
//           style={{ width: '300px' }}
//         />
//       </div>
//     </div>
//   );
// };

// export default Landmark;

// const Landmark = ({ onFinalPositionsChange, onImageChange }) => {
//   const initialCircles = Object.keys(landmarks).map(key => ({
//     id: key,
//     x: landmarks[key].x,
//     y: landmarks[key].y 
//   }));

//   const [circles, setCircles] = useState(initialCircles);
//   const [hoveredCircle, setHoveredCircle] = useState(null);
//   const [backgroundImage, setBackgroundImage] = useState(null);
//   const circleRefs = useRef([]);

//   useEffect(() => {
//     onFinalPositionsChange(circles);
//   }, [circles, onFinalPositionsChange]);

//   const handleStop = (e, data, index) => {
//     const newCircles = [...circles];
//     newCircles[index] = { ...newCircles[index], x: data.x, y: data.y };
//     setCircles(newCircles);
//   };

//   const handleImageUpload = async (event) => {
//     const file = event.target.files[0];
//     if (file) {
//       const reader = new FileReader();
//       reader.onloadend = async () => {
//         const base64Image = reader.result.split(',')[1]; // Get base64 string without the prefix
//         onImageChange(reader.result); // Pass the original image data to the parent component
//         try {
//           const response = await axios.post('http://localhost:8000/pointlandmark', { image_base64: base64Image });
//           if (response.data && response.data.isSuccess) {
//             const newLandmarks = response.data.landmark.map((point, index) => ({
//               id: index.toString(),
//               x: point[0] * 299 / 256,
//               y: point[1] * 299 / 256
//             }));
//             setCircles(newLandmarks);
//             setBackgroundImage(`data:image/jpeg;base64,${response.data.resized_image}`); // Set the resized image
//           }
//         } catch (error) {
//           console.error('Error uploading image and fetching landmarks:', error);
//         }
//       };
//       reader.readAsDataURL(file); // Read the file as a data URL
//     }
//   };

//   return (
//     <div style={{ display: 'flex', flexDirection: 'column', alignItems: 'center' }}>
//       <div style={{
//         position: 'relative',
//         width: '300px',
//         height: '300px',
//         border: '1px solid #BCBDC0',
//         borderRadius: '10px',
//         overflow: 'hidden',
//         backgroundSize: 'cover',
//         backgroundPosition: 'center',
//         backgroundImage: backgroundImage ? `url(${backgroundImage})` : 'none',
//         margin: '20px 0'
//       }}>
//         {circles.map((circle, index) => (
//           <Draggable
//             key={circle.id}
//             defaultPosition={{ x: circle.x, y: circle.y }}
//             onStop={(e, data) => handleStop(e, data, index)}
//           >
//             <div
//               ref={el => circleRefs.current[index] = el}
//               style={{
//                 position: 'absolute',
//                 width: '10px',
//                 height: '10px',
//                 borderRadius: '50%',
//                 backgroundColor: hoveredCircle === index ? '#545454' : '#BCBDC0',
//                 cursor: 'pointer'
//               }}
//               onMouseEnter={() => setHoveredCircle(index)}
//               onMouseLeave={() => setHoveredCircle(null)}
//             />
//           </Draggable>
//         ))}
//       </div>
//       <div>
//         <input
//           type="file"
//           className="form-control"
//           id="imageUpload"
//           accept="image/*"
//           onChange={handleImageUpload}
//           style={{ width: '300px' }}
//         />
//       </div>
//     </div>
//   );
// };

// export default Landmark;




const Landmark = ({ onFinalPositionsChange, onImageChange }) => {
  const initialCircles = Object.keys(landmarks).map(key => ({
    id: key,
    x: landmarks[key].x,
    y: landmarks[key].y 
  }));

  const [circles, setCircles] = useState(initialCircles);
  const [hoveredCircle, setHoveredCircle] = useState(null);
  const [backgroundImage, setBackgroundImage] = useState(null);
  const circleRefs = useRef([]);

  useEffect(() => {
    onFinalPositionsChange(circles);
  }, [circles, onFinalPositionsChange]);

  const handleStop = (e, data, index) => {
    const newCircles = [...circles];
    newCircles[index] = { ...newCircles[index], x: data.x, y: data.y };
    setCircles(newCircles);
  };

  const handleImageUpload = async (event) => {
    const file = event.target.files[0];
    if (file) {
      const reader = new FileReader();
      reader.onloadend = async () => {
        const base64Image = reader.result.split(',')[1]; // Get base64 string without the prefix
        onImageChange(reader.result); // Pass the original image data to the parent component
        try {
          const response = await axios.post('http://localhost:8000/pointlandmark', { image_base64: base64Image });
          if (response.data && response.data.isSuccess) {
            const newLandmarks = response.data.landmark.map((point, index) => ({
              id: index.toString(),
              x: point[0]*299/256, //* 299 / 256,
              y: point[1]*299/256
            }));
            setCircles(newLandmarks);
            setBackgroundImage(`data:image/jpeg;base64,${response.data.resized_image}`); // Set the resized image
          }
          console.log('image landmark: ', response.data.landmark);
        } catch (error) {
          console.error('Error uploading image and fetching landmarks:', error);
        }
      };
      reader.readAsDataURL(file); // Read the file as a data URL
    }
  };

  return (
    <div style={{ display: 'flex', flexDirection: 'column', alignItems: 'center' }}>
      <div style={{
        position: 'relative',
        width: '300px',
        height: '300px',
        border: '1px solid #BCBDC0',
        borderRadius: '10px',
        overflow: 'hidden',
        backgroundSize: 'cover',
        backgroundPosition: 'center',
        backgroundImage: backgroundImage ? `url(${backgroundImage})` : 'none',
        margin: '20px 0'
      }}>
        {circles.map((circle, index) => (
          <Draggable
            key={circle.id}
            defaultPosition={{ x: circle.x, y: circle.y }}
            onStop={(e, data) => handleStop(e, data, index)}
          >
            <div
              ref={el => circleRefs.current[index] = el}
              style={{
                position: 'absolute',
                width: '10px',
                height: '10px',
                borderRadius: '50%',
                backgroundColor: hoveredCircle === index ? '#545454' : '#BCBDC0',
                cursor: 'pointer'
              }}
              onMouseEnter={() => setHoveredCircle(index)}
              onMouseLeave={() => setHoveredCircle(null)}
            />
          </Draggable>
        ))}
      </div>
      <div>
        <input
          type="file"
          className="form-control"
          id="imageUpload"
          accept="image/*"
          onChange={handleImageUpload}
          style={{ width: '300px' }}
        />
      </div>
    </div>
  );
};

export default Landmark;