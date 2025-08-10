// components/HeaderComponents/ResetPasswordPage.tsx
import React, { useState } from 'react';
import { 
  Box, 
  TextField, 
  Button, 
  Typography, 
  Paper, 
  ThemeProvider, 
  createTheme, 
  CssBaseline,
  IconButton,
  InputAdornment,
  CircularProgress
} from '@mui/material';
import { Visibility, VisibilityOff } from '@mui/icons-material';
import { useParams, useNavigate } from 'react-router-dom';
import axios from 'axios';
import { BACKEND_URL } from '../../constants/index'; // Importer BACKEND_URL

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
    MuiPaper: { styleOverrides: { root: { backgroundImage: "none" } } },
  },
});

const ResetPasswordPage: React.FC = () => {
  const { token } = useParams<{ token: string }>();
  const navigate = useNavigate();
  const [newPassword, setNewPassword] = useState('');
  const [showPassword, setShowPassword] = useState(false);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!token) {
      setError('Invalid reset link');
      return;
    }
    setLoading(true);
    setError(null);

    try {
      await axios.post(`${BACKEND_URL}/auth/reset-password`, { token, newPassword });
      navigate('/login', { state: { message: 'Password reset successfully. Please log in with your new password.' } });
    } catch (err: any) {
      setError(err.response?.data?.message || 'Failed to reset password');
    } finally {
      setLoading(false);
    }
  };

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
        <Paper
          elevation={0}
          sx={{
            p: 4,
            width: '100',
            maxWidth: '480px',
            bgcolor: '#1a202c',
            borderRadius: '16px',
            border: '1px solid #2d3748',
          }}
        >
          <Typography variant="h4" sx={{ color: '#e2e8f0', mb: 3, textAlign: 'center' }}>
            Reset Password
          </Typography>
          <Typography sx={{ color: '#a0aec0', mb: 3 }}>
            Enter a new password for your account. Resetting your password will sign you out of all devices. You will need to enter your new password on all your devices.
          </Typography>
          {error && (
            <Typography sx={{ color: '#f56565', mb: 2, textAlign: 'center' }}>{error}</Typography>
          )}
          <Box component="form" onSubmit={handleSubmit} sx={{ mb: 3 }}>
            <TextField
              placeholder="New Password"
              type={showPassword ? 'text' : 'password'}
              value={newPassword}
              onChange={(e) => setNewPassword(e.target.value)}
              fullWidth
              variant="outlined"
              required
              sx={{
                mb: 2,
                '& .MuiOutlinedInput-root': {
                  borderRadius: '8px',
                  bgcolor: '#2d3748',
                  '& fieldset': { borderColor: '#4a5568' },
                  '&:hover fieldset': { borderColor: '#718096' },
                  '&.Mui-focused fieldset': { borderColor: '#4299e1' },
                },
                '& .MuiInputBase-input': {
                  color: '#e2e8f0',
                  fontSize: '1rem',
                  padding: '16px',
                },
                '& .MuiInputBase-input::placeholder': {
                  color: '#a0aec0',
                  opacity: 1,
                },
              }}
              InputProps={{
                endAdornment: (
                  <InputAdornment position="end">
                    <IconButton
                      aria-label="toggle password visibility"
                      onClick={() => setShowPassword(!showPassword)}
                      edge="end"
                      sx={{ color: '#a0aec0' }}
                    >
                      {showPassword ? <VisibilityOff /> : <Visibility />}
                    </IconButton>
                  </InputAdornment>
                ),
              }}
            />
            <Typography sx={{ color: '#a0aec0', mb: 2, fontSize: '0.85rem' }}>
              At least 8 characters with mix of letters, numbers and symbols.
            </Typography>
            <Button
              type="submit"
              disabled={loading || newPassword.length < 8}
              fullWidth
              sx={{
                mb: 3,
                backgroundColor: '#4299e1',
                color: 'white',
                borderRadius: '8px',
                py: 2,
                fontSize: '1rem',
                fontWeight: '600',
                textTransform: 'none',
                '&:hover': { backgroundColor: '#3182ce' },
                '&:disabled': { backgroundColor: '#4a5568', color: '#a0aec0' },
              }}
              startIcon={loading ? <CircularProgress size={20} color="inherit" /> : null}
            >
              {loading ? 'Resetting...' : 'Reset Password'}
            </Button>
          </Box>
        </Paper>
      </Box>
    </ThemeProvider>
  );
};

export default ResetPasswordPage;