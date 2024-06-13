// src/MainContent.js

import React from 'react';

const MainContent = () => {
  return (
    <>
      <div className="position-relative overflow-hidden p-3 p-md-5 m-md-3 text-center bg-light">
        <div className="col-md-5 p-lg-5 mx-auto my-5">
          <h1 className="display-4 fw-normal">Scalable Human Image Generation and Editing</h1>
          <p className="lead fw-normal">Conducted research in Scalble Human Image Generation and Editing based on DiT model.</p>
          <a className="btn btn-outline-secondary" href="#">Detail</a>
        </div>
        <div className="product-device shadow-sm d-none d-md-block"></div>
        <div className="product-device product-device-2 shadow-sm d-none d-md-block"></div>
      </div>

      <div className="d-md-flex flex-md-equal w-100 my-md-3 ps-md-3">
        {/* <div className="text-bg-dark me-md-3 pt-3 px-3 pt-md-5 px-md-5 text-center overflow-hidden">
          <div className="my-3 py-3">
          <p className="lead">We proposes a model capable of editing facial expressions in existing human images using multi-modal conditioning that is text and facial landmarks. Facial landmarks refer to the localization of essential points on facial images. The most common set of facial landmarks comprising 68 points is adopted. These landmarks are used to adjust facial expressions, giving conditioning to the model. By using facial landmarks, more subtle facial expressions and emotions can be edited to generated human images. Text conditioning enables the adjustment of features such as eye color, hair color, and hairstyle to craft the user’s desired image.

In order to achieve the goal, we suggest a method that uses DiT models and gives text and facial landmark conditioning. DiT models are typically designed for single embedding, while accommodating both text and facial landmark conditioning. We also present modifications to the model structure to enable the handling of sequential data.</p>
          
            </div>
        </div> */}
        <div className="bg-light me-md-3 pt-3 px-3 pt-md-5 px-md-5 text-center overflow-hidden">
          <div className="my-3 p-3">
          <p className="lead">We proposes a model capable of editing facial expressions in existing human images using multi-modal conditioning that is text and facial landmarks. Facial landmarks refer to the localization of essential points on facial images. The most common set of facial landmarks comprising 68 points is adopted. These landmarks are used to adjust facial expressions, giving conditioning to the model. By using facial landmarks, more subtle facial expressions and emotions can be edited to generated human images. Text conditioning enables the adjustment of features such as eye color, hair color, and hairstyle to craft the user’s desired image.

In order to achieve the goal, we suggest a method that uses DiT models and gives text and facial landmark conditioning. DiT models are typically designed for single embedding, while accommodating both text and facial landmark conditioning. We also present modifications to the model structure to enable the handling of sequential data.</p>
          
          </div>
          {/* <div className="bg-dark shadow-sm mx-auto" style={{ width: '80%', height: '300px', borderRadius: '21px 21px 0 0' }}></div> */}
        </div>
      </div>

      <div className="d-md-flex flex-md-equal w-100 my-md-3 ps-md-3">
        <div className="bg-light me-md-3 pt-3 px-3 pt-md-5 px-md-5 text-center overflow-hidden">
          <div className="my-3 p-3">
            <h2 className="display-5">Another headline</h2>
            <p className="lead">And an even wittier subheading.</p>
          </div>
          <div className="bg-dark shadow-sm mx-auto" style={{ width: '80%', height: '300px', borderRadius: '21px 21px 0 0' }}></div>
        </div>
        <div className="text-bg-primary me-md-3 pt-3 px-3 pt-md-5 px-md-5 text-center overflow-hidden">
          <div className="my-3 py-3">
            <h2 className="display-5">Another headline</h2>
            <p className="lead">And an even wittier subheading.</p>
          </div>
          <div className="bg-light shadow-sm mx-auto" style={{ width: '80%', height: '300px', borderRadius: '21px 21px 0 0' }}></div>
        </div>
      </div>

      <div className="d-md-flex flex-md-equal w-100 my-md-3 ps-md-3">
        <div className="bg-light me-md-3 pt-3 px-3 pt-md-5 px-md-5 text-center overflow-hidden">
          <div className="my-3 p-3">
            <h2 className="display-5">Another headline</h2>
            <p className="lead">And an even wittier subheading.</p>
          </div>
          <div className="bg-body shadow-sm mx-auto" style={{ width: '80%', height: '300px', borderRadius: '21px 21px 0 0' }}></div>
        </div>
        <div className="bg-light me-md-3 pt-3 px-3 pt-md-5 px-md-5 text-center overflow-hidden">
          <div className="my-3 py-3">
            <h2 className="display-5">Another headline</h2>
            <p className="lead">And an even wittier subheading.</p>
          </div>
          <div className="bg-body shadow-sm mx-auto" style={{ width: '80%', height: '300px', borderRadius: '21px 21px 0 0' }}></div>
        </div>
      </div>

      <div className="d-md-flex flex-md-equal w-100 my-md-3 ps-md-3">
        <div className="bg-light me-md-3 pt-3 px-3 pt-md-5 px-md-5 text-center overflow-hidden">
          <div className="my-3 p-3">
            <h2 className="display-5">Another headline</h2>
            <p className="lead">And an even wittier subheading.</p>
          </div>
          <div className="bg-body shadow-sm mx-auto" style={{ width: '80%', height: '300px', borderRadius: '21px 21px 0 0' }}></div>
        </div>
        <div className="bg-light me-md-3 pt-3 px-3 pt-md-5 px-md-5 text-center overflow-hidden">
          <div className="my-3 py-3">
            <h2 className="display-5">Another headline</h2>
            <p className="lead">And an even wittier subheading.</p>
          </div>
          <div className="bg-body shadow-sm mx-auto" style={{ width: '80%', height: '300px', borderRadius: '21px 21px 0 0' }}></div>
        </div>
      </div>
    </>
  );
};

export default MainContent;
