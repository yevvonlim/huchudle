// import React, { useRef, useState, useMemo } from 'react';
// import styled from 'styled-components';
// import OriginLandmark from './images/landmark.png'; // Ensure the path is correct

// const ImageUpload = () => {
//     const fileInputRef = useRef(null);
//     const [imageFile, setImageFile] = useState(null);

//     const handleClickFileInput = () => {
//         fileInputRef.current?.click();
//     };

//     const uploadProfile = (e) => {
//         const fileList = e.target.files;
//         const length = fileList?.length;
//         if (!fileList || length === 0) {
//             return;
//         }
//         if (length > 1) {
//             alert('파일은 하나만 업로드 가능합니다.');
//             return;
//         }
//         if (fileList[0].size > 1024 * 1024 * 10) {
//             alert('파일 크기는 10MB 이하로 업로드 가능합니다.');
//             return;
//         }
//         if (fileList && fileList[0]) {
//             const url = URL.createObjectURL(fileList[0]);
//             setImageFile({
//                 file: fileList[0],
//                 thumbnail: url,
//                 type: fileList[0].type
//             });
//         }
//     };

//     const showImage = useMemo(() => {
//         if (!imageFile) {
//             return <img src={OriginLandmark} alt="landmark" onClick={handleClickFileInput} />;
//         }
//         return <ShowFileImage src={imageFile.thumbnail} alt="thumbnail" onClick={handleClickFileInput} />;
//     }, [imageFile]);

//     const ShowFileImage = styled.img`
//         width: 100%;
//         height: 100%;
//         object-fit: cover;
//     `;

//     return (
//         <div>
//             {showImage}
//             <input type="file" accept="image/*" ref={fileInputRef}  onChange={uploadProfile} />
//             <button onClick={handleClickFileInput}>Upload</button>
//         </div>
//     );
// };

// export default ImageUpload;

// import React, { useState } from 'react';

// const ImageUpload = () => {
//   const [image, setImage] = useState(null);
//   const [imagePreviewUrl, setImagePreviewUrl] = useState('');

//   const handleImageChange = (e) => {
//     e.preventDefault();
    
//     let reader = new FileReader();
//     let file = e.target.files[0];

//     reader.onloadend = () => {
//       setImage(file);
//       setImagePreviewUrl(reader.result);
//     }

//     if (file) {
//       reader.readAsDataURL(file);
//     }
//   };

//   return (
//     <div>
//       <input 
//         type="file" 
//         accept="image/*" 
//         onChange={handleImageChange} 
//       />
//       {imagePreviewUrl && (
//         <div>
//           <img src={imagePreviewUrl} alt="Image Preview" style={{width: '256px', height: 'auto'}} />
//         </div>
//       )}
//     </div>
//   );
// };

// export default ImageUpload;



// //#####################Axios#####################
// import React, { useState } from 'react';
// import axios from 'axios';

// const ImageUpload = () => {
//   const [image, setImage] = useState(null);
//   const [imagePreviewUrl, setImagePreviewUrl] = useState('');
//   const [uploadStatus, setUploadStatus] = useState('');

//   const handleImageChange = (e) => {
//     e.preventDefault();

//     let file = e.target.files[0];
//     let reader = new FileReader();

//     reader.onloadend = () => {
//       setImage(file);
//       setImagePreviewUrl(reader.result);
//     };

//     if (file) {
//       reader.readAsDataURL(file);
//     }
//   };

//   const handleSubmit = async (e) => {
//     e.preventDefault();
//     if (!image) {
//       setUploadStatus('Please select an image first.');
//       return;
//     }

//     const formData = new FormData();
//     formData.append('image', image);

//     try {
//       const response = await axios.post('http://localhost:5000/upload', formData);

//       if (response.status === 200) {
//         setUploadStatus('Image uploaded successfully!');
//       } else {
//         setUploadStatus('Image upload failed.');
//       }
//     } catch (error) {
//       setUploadStatus('Image upload failed.');
//       console.error('Error uploading image:', error);
//     }
//   };

//   return (
//     <div>
//       <input
//         type="file"
//         accept="image/*"
//         onChange={handleImageChange}
//       />
//       {imagePreviewUrl && (
//         <div>
//           <h2>Image Preview:</h2>
//           <img src={imagePreviewUrl} alt="Image Preview" style={{ width: '300px', height: 'auto' }} />
//         </div>
//       )}
//       <button onClick={handleSubmit}>Upload Image</button>
//       {uploadStatus && <p>{uploadStatus}</p>}
//     </div>
//   );
// };

// export default ImageUpload;
// //###############################################Axios End#############################################
import React, { useState } from 'react';
import axios from 'axios';
import Button from 'react-bootstrap/Button';
import 'bootstrap/dist/css/bootstrap.min.css';
const ImageUpload = () => {
  const [image, setImage] = useState(null);
  const [imagePreviewUrl, setImagePreviewUrl] = useState('');
  const [uploadStatus, setUploadStatus] = useState('');

  const handleImageChange = (e) => {
    e.preventDefault();

    let file = e.target.files[0];
    let reader = new FileReader();

    reader.onloadend = () => {
      setImage(file);
      setImagePreviewUrl(reader.result);
    };

    if (file) {
      reader.readAsDataURL(file);
    }
  };

  const handleSubmit = async (e) => {
    e.preventDefault();
    if (!image) {
      setUploadStatus('Please select an image first.');
      return;
    }

    const formData = new FormData();
    formData.append('image', image);

    try {
      const response = await axios.post('http://localhost:5000/upload', formData);

      if (response.status === 200) {
        setUploadStatus('Image uploaded successfully!');
      } else {
        setUploadStatus('Image upload failed.');
      }
    } catch (error) {
      setUploadStatus('Image upload failed.');
      console.error('Error uploading image:', error);
    }
  };

  return (
    <div>
      <div className="mb-3">
        <label htmlFor="imageUpload" className="form-label"></label>
        <input
          type="file"
          className="form-control"
          id="imageUpload"
          accept="image/*"
          onChange={handleImageChange}
        />
      </div>
      {imagePreviewUrl && (
        <div>
          <h2>Image Preview:</h2>
          <img src={imagePreviewUrl} alt="Image Preview" style={{ width: '300px', height: 'auto' }} />
        </div>
      )}

      <Button variant="success" onClick={handleSubmit}>Upload Image</Button>{' '}
      {uploadStatus && <p>{uploadStatus}</p>}
    </div>
  );
};

export default ImageUpload;
