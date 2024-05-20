import StepProgressBar from 'react-step-progress';
import 'react-step-progress/dist/index.css';
import * as React from 'react';
import MultilineTextFields from './MultilineTextFields';

import { useRef, useState, useMemo } from 'react';
import styled from 'styled-components';
import OriginLandmark from './images/landmark.png'; // Ensure the path is correct
import ImageUpload from './ImageUpload';


const step1Content = <h1>Step 1 Content
  <MultilineTextFields />
</h1>;
const step2Content = <h1>Step 2 Content
    <ImageUpload />

</h1>;
const step3Content = <h1>Step 3 Content</h1>;



function step2Validator() {
  // return a boolean


}

function step3Validator() {
  // return a boolean
}

function onFormSubmit() {
  // handle the submit logic here
}



function App() {
  return (
    <StepProgressBar
      startingStep={0}
      onSubmit={onFormSubmit}
      steps={[
        {
          label: 'Step 1',
          subtitle: '10%',
          name: 'step 1',
          content: step1Content
        },
        {
          label: 'Step 2',
          subtitle: '50%',
          name: 'step 2',
          content: step2Content,
          validator: step2Validator
        },
        {
          label: 'Step 3',
          subtitle: '100%',
          name: 'step 3',
          content: step3Content,
          validator: step3Validator
        }
      ]}
    />
  );
}


export default App;
