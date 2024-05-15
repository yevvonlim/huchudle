import * as React from 'react';
import Box from '@mui/material/Box';
import TextField from '@mui/material/TextField';

function MultilineTextFields() {
  return (
    <Box
      component="form"
      sx={{
        '& .MuiTextField-root': { m: 1, width: '25ch' },
      }}
      noValidate
      autoComplete="off"
    >
      <div>
        <TextField
          id="outlined-textarea"
          label="Text Explanation"
          placeholder="Create a person who has a brown hair and a blue eyes."
          rows = {4}
          multiline
        />
      </div>
    </Box>
  );
}
export default MultilineTextFields;