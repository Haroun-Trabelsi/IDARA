import React from 'react';
import { Box, Typography, Button } from '@mui/material';
import { useNavigate } from 'react-router-dom';

const ErrorPage: React.FC = () => {
  const navigate = useNavigate();

  return (
    <Box
      sx={{
        minHeight: '100vh',
        display: 'flex',
        alignItems: 'center',
        justifyContent: 'center',
        bgcolor: '#0f172a',
        p: 2,
      }}
    >
      <Box sx={{ textAlign: 'center', color: '#e2e8f0' }}>
        <Typography variant="h4" sx={{ mb: 2 }}>
          Invalid Invitation
        </Typography>
        <Typography sx={{ mb: 2 }}>
          The invitation link is invalid or has expired. Please log in manually.
        </Typography>
        <Button
          variant="contained"
          onClick={() => navigate('/login')}
          sx={{ backgroundColor: '#4299e1', '&:hover': { backgroundColor: '#3182ce' } }}
        >
          Go to Login
        </Button>
      </Box>
    </Box>
  );
};

export default ErrorPage;