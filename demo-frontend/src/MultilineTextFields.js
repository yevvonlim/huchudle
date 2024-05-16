import React from 'react';
import Box from '@mui/material/Box';
import TextField from '@mui/material/TextField';

function MultilineTextFields({ onTextChange }) {
  const handleChange = (event) => {
    onTextChange(event.target.value);
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
          label="Prompt"
          placeholder="Create a person who has a brown hair and blue eyes."
          rows={4}
          multiline
          onChange={handleChange}
        />
      </div>
    </Box>
  );
}

export default MultilineTextFields;
