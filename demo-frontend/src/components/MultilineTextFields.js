import React from 'react';
import Box from '@mui/material/Box';
import TextField from '@mui/material/TextField';

function MultilineTextFields({ onTextInputChange }) {
  const handleChange = (event) => {
    onTextInputChange(event.target.value);
  };

  return (
    <Box
      component="form"
      sx={{
        '& .MuiTextField-root': { m: 1, width: '40ch' },
      }}
      noValidate
      autoComplete="off"
    >
      <div>
        <TextField
          id="outlined-textarea"
          label="Text Explanation"
          placeholder="Create a person who has brown hair and blue eyes."
          rows={4}
          multiline
          onChange={handleChange}
        />
      </div>
    </Box>
  );
}

export default MultilineTextFields;
