import React from 'react';
import { Box, Typography, Link, ThemeProvider, createTheme, CssBaseline } from '@mui/material';
import { useLocation } from 'react-router-dom';

const darkTheme = createTheme({
  palette: {
    mode: "dark",
    primary: { main: "#4299e1" },
    secondary: { main: "#f59e0b" },
    background: { default: "#0f172a", paper: "#1a202c" },
    text: { primary: "#e2e8f0", secondary: "#a0aec0" },
  },
  components: {
    MuiCssBaseline: { styleOverrides: { body: { backgroundColor: "#0f172a" } } },
  },
});

const CheckEmailPage: React.FC = () => {
  const location = useLocation();
  const email = location.state?.email || 'the provided email';

  return (
    <ThemeProvider theme={darkTheme}>
      <CssBaseline />
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
        <Box
          sx={{
            p: 4,
            width: '100%',
            maxWidth: '480px',
            bgcolor: '#1a202c',
            borderRadius: '16px',
            border: '1px solid #2d3748',
            textAlign: 'center',
          }}
        >
          <Typography variant="h4" sx={{ color: '#e2e8f0', mb: 3 }}>
            Check Your Email
          </Typography>
          <Typography sx={{ color: '#a0aec0', mb: 2 }}>
            If the email address <strong>{email}</strong> is associated with a ftrack account, then we’ve sent you an email with a link to reset your password.
          </Typography>
          <Typography sx={{ color: '#a0aec0', mb: 2 }}>
            If you don't see the email, please check your spam,{' '}
            <Link href="#" sx={{ color: '#4299e1', textDecoration: 'none', '&:hover': { textDecoration: 'underline' } }}>
              
            </Link>{' '}
            or{' '}
            <Link href="/login" sx={{ color: '#4299e1', textDecoration: 'none', '&:hover': { textDecoration: 'underline' } }}>
              go back to sign in
            </Link>.
          </Typography>
        </Box>
      </Box>
    </ThemeProvider>
  );
};

export default CheckEmailPage;