import React, { useState, useEffect } from "react";
import { 
  Box, 
  TextField, 
  Button, 
  Typography, 
  Paper, 
  IconButton, 
  Divider,
  Link,
  CircularProgress,
  Alert,
  ThemeProvider,
  createTheme,
  CssBaseline
} from "@mui/material";
import { Google as GoogleIcon, Apple as AppleIcon, Email as EmailIcon } from "@mui/icons-material";
import { useNavigate } from 'react-router-dom';
import { useAuth } from 'contexts/AuthContext';

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

interface LoginFormData {
  email: string;
  password: string;
}

const LoginPage: React.FC = () => {
  const navigate = useNavigate();
  const { login, account } = useAuth();
  const [formData, setFormData] = useState<LoginFormData>({ email: '', password: '' });
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const handleChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const { name, value } = e.target;
    setFormData(prev => ({ ...prev, [name]: value }));
  };

  const handleSubmit = async (e: React.FormEvent<HTMLFormElement>) => {
    e.preventDefault();
    setLoading(true);
    setError(null);

    try {
      await login({ email: formData.email, password: formData.password });
    } catch (err: any) {
      setError(err.response?.data?.message || 'Invalid email or password');
    } finally {
      setLoading(false);
    }
  };

  const isSubmitDisabled = !formData.email || !formData.password;

  // Vérification après mise à jour de l'account
  useEffect(() => {
    if (account) {
      if (account.mustCompleteProfile === true) {
        navigate('/CompleteProfil');
      } else {
        navigate('/');
      }
    }
  }, [account, navigate]);

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
          p: 2
        }}
      >
        <Paper
          elevation={0}
          sx={{
            p: 4,
            width: '100%',
            maxWidth: '480px',
            bgcolor: '#1a202c',
            borderRadius: '16px',
            border: '1px solid #2d3748'
          }}
        >
          <Typography
            variant="h4"
            sx={{
              textAlign: 'center',
              fontSize: '2rem',
              fontWeight: 'bold',
              color: '#e2e8f0',
              mb: 3
            }}
          >
            Sign in to your account
          </Typography>

          {error && (
            <Alert severity="error" sx={{ mb: 2 }}>
              {error}
            </Alert>
          )}

          <Box component="form" onSubmit={handleSubmit} sx={{ mb: 3 }}>
            <TextField
              placeholder="Email"
              name="email"
              type="email"
              value={formData.email}
              onChange={handleChange}
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
                  padding: '16px'
                },
                '& .MuiInputBase-input::placeholder': {
                  color: '#a0aec0',
                  opacity: 1,
                },
              }}
            />

            <TextField
              placeholder="Password"
              name="password"
              type="password"
              value={formData.password}
              onChange={handleChange}
              fullWidth
              variant="outlined"
              required
              sx={{
                mb: 3,
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
                  padding: '16px'
                },
                '& .MuiInputBase-input::placeholder': {
                  color: '#a0aec0',
                  opacity: 1,
                },
              }}
            />

            <Button
              type="submit"
              disabled={isSubmitDisabled || loading}
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
              startIcon={loading ? <CircularProgress size={20} color="inherit" /> : <EmailIcon />}
            >
              {loading ? 'Signing in...' : 'Sign in with email'}
            </Button>
          </Box>

          <Box sx={{ mb: 3 }}>
            <Divider sx={{ mb: 2, borderColor: '#4a5568' }}>
              <Typography variant="body2" sx={{ color: '#a0aec0', px: 2 }}>Other sign in options</Typography>
            </Divider>

            <Box sx={{ display: 'flex', gap: 2, justifyContent: 'center' }}>
              <IconButton
                sx={{
                  border: '2px solid #4a5568',
                  borderRadius: '8px',
                  width: '60px',
                  height: '60px',
                  '&:hover': { borderColor: '#718096', backgroundColor: '#2d3748' },
                }}
              >
                <GoogleIcon sx={{ color: '#4285f4', fontSize: '24px' }} />
              </IconButton>
              <IconButton
                sx={{
                  border: '2px solid #4a5568',
                  borderRadius: '8px',
                  width: '60px',
                  height: '60px',
                  '&:hover': { borderColor: '#718096', backgroundColor: '#2d3748' },
                }}
              >
                <AppleIcon sx={{ color: '#e2e8f0', fontSize: '24px' }} />
              </IconButton>
            </Box>
          </Box>

          <Box sx={{ textAlign: 'center' }}>
            <Typography variant="body2" sx={{ color: '#a0aec0' }}>
              Don't have an account?{' '}
              <Link
                href="/register"
                sx={{
                  color: '#4299e1',
                  textDecoration: 'none',
                  fontWeight: '600',
                  '&:hover': { textDecoration: 'underline' },
                }}
              >
                Sign up
              </Link>
            </Typography>
          </Box>
        </Paper>
      </Box>
    </ThemeProvider>
  );
};

export default LoginPage;