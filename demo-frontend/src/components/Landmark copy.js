import React, { useState, useRef, useEffect } from 'react';
import axios from 'axios';
import Draggable from 'react-draggable';
import 'bootstrap/dist/css/bootstrap.min.css';


const landmarks =
  {
    "landmark-0": { "x": 0.30, "y": 0.47 },
    "landmark-1": { "x": 0.29, "y": 0.52 },
    "landmark-2": { "x": 0.29, "y": 0.58 },
    "landmark-3": { "x": 0.30, "y": 0.64 },
    "landmark-4": { "x": 0.32, "y": 0.70 },
    "landmark-5": { "x": 0.35, "y": 0.76 },
    "landmark-6": { "x": 0.38, "y": 0.81 },
    "landmark-7": { "x": 0.41, "y": 0.86 },
    "landmark-8": { "x": 0.47, "y": 0.87 },
    "landmark-9": { "x": 0.55, "y": 0.87 },
    "landmark-10": { "x": 0.65, "y": 0.86 },
    "landmark-11": { "x": 0.74, "y": 0.83 },
    "landmark-12": { "x": 0.80, "y": 0.78 },
    "landmark-13": { "x": 0.85, "y": 0.71 },
    "landmark-14": { "x": 0.86, "y": 0.62 },
    "landmark-15": { "x": 0.86, "y": 0.52 },
    "landmark-16": { "x": 0.86, "y": 0.44 },
    "landmark-17": { "x": 0.29, "y": 0.36 },
    "landmark-18": { "x": 0.30, "y": 0.34 },
    "landmark-19": { "x": 0.33, "y": 0.34 },
    "landmark-20": { "x": 0.36, "y": 0.34 },
    "landmark-21": { "x": 0.39, "y": 0.35 },
    "landmark-22": { "x": 0.48, "y": 0.34 },
    "landmark-23": { "x": 0.54, "y": 0.32 },
    "landmark-24": { "x": 0.60, "y": 0.32 },
    "landmark-25": { "x": 0.66, "y": 0.33 },
    "landmark-26": { "x": 0.71, "y": 0.36 },
    "landmark-27": { "x": 0.43, "y": 0.41 },
    "landmark-28": { "x": 0.41, "y": 0.47 },
    "landmark-29": { "x": 0.40, "y": 0.52 },
    "landmark-30": { "x": 0.39, "y": 0.57 },
    "landmark-31": { "x": 0.38, "y": 0.61 },
    "landmark-32": { "x": 0.39, "y": 0.62 },
    "landmark-33": { "x": 0.42, "y": 0.63 },
    "landmark-34": { "x": 0.45, "y": 0.62 },
    "landmark-35": { "x": 0.48, "y": 0.61 },
    "landmark-36": { "x": 0.32, "y": 0.43 },
    "landmark-37": { "x": 0.34, "y": 0.41 },
    "landmark-38": { "x": 0.37, "y": 0.41 },
    "landmark-39": { "x": 0.40, "y": 0.43 },
    "landmark-40": { "x": 0.37, "y": 0.45 },
    "landmark-41": { "x": 0.34, "y": 0.45 },
    "landmark-42": { "x": 0.54, "y": 0.43 },
    "landmark-43": { "x": 0.57, "y": 0.40 },
    "landmark-44": { "x": 0.61, "y": 0.40 },
    "landmark-45": { "x": 0.64, "y": 0.42 },
    "landmark-46": { "x": 0.61, "y": 0.44 },
    "landmark-47": { "x": 0.57, "y": 0.44 },
    "landmark-48": { "x": 0.39, "y": 0.72 },
    "landmark-49": { "x": 0.39, "y": 0.70 },
    "landmark-50": { "x": 0.41, "y": 0.68 },
    "landmark-51": { "x": 0.44, "y": 0.69 },
    "landmark-52": { "x": 0.47, "y": 0.68 },
    "landmark-53": { "x": 0.54, "y": 0.70 },
    "landmark-54": { "x": 0.61, "y": 0.72 },
    "landmark-55": { "x": 0.55, "y": 0.77 },
    "landmark-56": { "x": 0.50, "y": 0.78 },
    "landmark-57": { "x": 0.45, "y": 0.78 },
    "landmark-58": { "x": 0.40, "y": 0.75 },
    "landmark-59": { "x": 0.39, "y": 0.75 },
    "landmark-60": { "x": 0.41, "y": 0.72 },
    "landmark-61": { "x": 0.44, "y": 0.71 },
    "landmark-62": { "x": 0.47, "y": 0.71 },
    "landmark-63": { "x": 0.54, "y": 0.71 },
    "landmark-64": { "x": 0.61, "y": 0.72 },
    "landmark-65": { "x": 0.55, "y": 0.75 },
    "landmark-66": { "x": 0.50, "y": 0.75 },
    "landmark-67": { "x": 0.45, "y": 0.75 }
  };
  
const landmarks2 = {
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



const Landmark = ({ onFinalPositionsChange }) => {
  const initialCircles = Object.keys(landmarks).map(key => ({
    id: key,
    x: landmarks[key].x * 256,
    y: landmarks[key].y * 256
  }));

  const [circles, setCircles] = useState(initialCircles);
  const [hoveredCircle, setHoveredCircle] = useState(null);
  const [image, setImage] = useState(null);
  const circleRefs = useRef([]);

  useEffect(() => {
    onFinalPositionsChange(circles);
  }, [circles, onFinalPositionsChange]);

  const handleStop = (e, data, index) => {
    const newCircles = [...circles];
    newCircles[index] = { ...newCircles[index], x: data.x, y: data.y };
    setCircles(newCircles);
  };

  const handleImageUpload = (event) => {
    const file = event.target.files[0];
    if (file) {
      const reader = new FileReader();
      reader.onloadend = () => {
        setImage(reader.result);
      };
      reader.readAsDataURL(file);
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
        overflow: 'hidden', // Ensure the image respects the border radius
        backgroundSize: 'cover',
        backgroundPosition: 'center',
        backgroundImage: image ? `url(${image})` : 'none',
        margin: '20px 0' // Optional: Add some margin for better spacing
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
          style={{ width: '300px' }} // Set the width of the input button
        />
      </div>
    </div>
  );
};

export default Landmark;