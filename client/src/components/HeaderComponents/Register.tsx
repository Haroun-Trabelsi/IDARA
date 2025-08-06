import React, { useState } from 'react';
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
  ThemeProvider,
  createTheme,
  CssBaseline,
  FormControlLabel,
  Checkbox
} from '@mui/material';
import { Google as GoogleIcon, Apple as AppleIcon, Email as EmailIcon } from '@mui/icons-material';
import { useNavigate } from 'react-router-dom';
import { useAuth } from 'contexts/AuthContext';
import { validateProfile, validatePassword } from '../../utils/validation';

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

interface RegisterFormData {
  name: string;
  surname: string;
  organizationName: string;
  email: string;
  password: string;
}

const RegisterPage: React.FC = () => {
  const navigate = useNavigate();
  const { register } = useAuth();
  const [formData, setFormData] = useState<RegisterFormData>({
    name: '',
    surname: '',
    organizationName: '',
    email: '',
    password: ''
  });
  const [errors, setErrors] = useState({
    name: '',
    surname: '',
    organizationName: '',
    email: '',
    password: ''
  });
  const [loading, setLoading] = useState(false);
  const [acceptOffers, setAcceptOffers] = useState(false);

  const handleChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const { name, value } = e.target;
    setFormData(prev => ({ ...prev, [name]: value }));
    setErrors(prev => ({ ...prev, [name]: '' }));
  };

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    setLoading(true);
    setErrors({ name: '', surname: '', organizationName: '', email: '', password: '' });

    const profileValidation = validateProfile({
      name: formData.name,
      surname: formData.surname,
      email: formData.email
    });

    const passwordValidation = validatePassword({
      currentPassword: formData.password,
      newPassword: formData.password,
      confirmPassword: formData.password
    });

    if (!profileValidation.isValid || !passwordValidation.isValid) {
      setErrors({
        name: profileValidation.errors.name,
        surname: profileValidation.errors.surname,
        organizationName: '',
        email: profileValidation.errors.email,
        password: passwordValidation.errors.newPassword || passwordValidation.errors.currentPassword
      });
      setLoading(false);
      return;
    }

    try {
      const registrationData = {
        ...formData,
        role: 'user' as const,
        canInvite: true,
        isVerified: false,
        mfaEnabled: false,
      };
      await register(registrationData);
      console.log('Registration successful, navigating to /verify-email');
      navigate('/verify-email');
    } catch (err: any) {
      console.error('Registration error:', err);
      setErrors(prev => ({
        ...prev,
        email: err.response?.data?.message.includes('email') ? err.response?.data?.message : prev.email,
        password: err.response?.data?.message.includes('password') ? err.response?.data?.message : prev.password
      }));
    } finally {
      setLoading(false);
    }
  };

  const isSubmitDisabled = !formData.name || !formData.surname || !formData.organizationName || !formData.email || !formData.password || !!errors.name || !!errors.surname || !!errors.email || !!errors.password;

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
            Sign up with email
          </Typography>

          <Box component="form" onSubmit={handleSubmit} sx={{ mb: 3 }}>
            <TextField
              placeholder="Name"
              name="name"
              type="text"
              value={formData.name}
              onChange={handleChange}
              fullWidth
              variant="outlined"
              required
              error={!!errors.name}
              helperText={errors.name}
              sx={{
                mb: 2,
                '& .MuiOutlinedInput-root': {
                  borderRadius: '8px',
                  bgcolor: '#2d3748',
                  '& fieldset': { borderColor: '#4a5568' },
                  '&:hover fieldset': { borderColor: '#718096' },
                  '&.Mui-focused fieldset': { borderColor: '#4299e1' },
                },
                '& .MuiInputBase-input': { color: '#e2e8f0', fontSize: '1rem', padding: '16px' },
                '& .MuiInputBase-input::placeholder': { color: '#a0aec0', opacity: 1 },
                '& .MuiFormHelperText-root': { color: '#f56565' }
              }}
            />

            <TextField
              placeholder="Surname"
              name="surname"
              type="text"
              value={formData.surname}
              onChange={handleChange}
              fullWidth
              variant="outlined"
              required
              error={!!errors.surname}
              helperText={errors.surname}
              sx={{
                mb: 2,
                '& .MuiOutlinedInput-root': {
                  borderRadius: '8px',
                  bgcolor: '#2d3748',
                  '& fieldset': { borderColor: '#4a5568' },
                  '&:hover fieldset': { borderColor: '#718096' },
                  '&.Mui-focused fieldset': { borderColor: '#4299e1' },
                },
                '& .MuiInputBase-input': { color: '#e2e8f0', fontSize: '1rem', padding: '16px' },
                '& .MuiInputBase-input::placeholder': { color: '#a0aec0', opacity: 1 },
                '& .MuiFormHelperText-root': { color: '#f56565' }
              }}
            />

            <TextField
              placeholder="Organization Name"
              name="organizationName"
              type="text"
              value={formData.organizationName}
              onChange={handleChange}
              fullWidth
              variant="outlined"
              required
              error={!!errors.organizationName}
              helperText={errors.organizationName}
              sx={{
                mb: 2,
                '& .MuiOutlinedInput-root': {
                  borderRadius: '8px',
                  bgcolor: '#2d3748',
                  '& fieldset': { borderColor: '#4a5568' },
                  '&:hover fieldset': { borderColor: '#718096' },
                  '&.Mui-focused fieldset': { borderColor: '#4299e1' },
                },
                '& .MuiInputBase-input': { color: '#e2e8f0', fontSize: '1rem', padding: '16px' },
                '& .MuiInputBase-input::placeholder': { color: '#a0aec0', opacity: 1 },
                '& .MuiFormHelperText-root': { color: '#f56565' }
              }}
            />

            <TextField
              placeholder="Email"
              name="email"
              type="email"
              value={formData.email}
              onChange={handleChange}
              fullWidth
              variant="outlined"
              required
              error={!!errors.email}
              helperText={errors.email}
              sx={{
                mb: 2,
                '& .MuiOutlinedInput-root': {
                  borderRadius: '8px',
                  bgcolor: '#2d3748',
                  '& fieldset': { borderColor: '#4a5568' },
                  '&:hover fieldset': { borderColor: '#718096' },
                  '&.Mui-focused fieldset': { borderColor: '#4299e1' },
                },
                '& .MuiInputBase-input': { color: '#e2e8f0', fontSize: '1rem', padding: '16px' },
                '& .MuiInputBase-input::placeholder': { color: '#a0aec0', opacity: 1 },
                '& .MuiFormHelperText-root': { color: '#f56565' }
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
              error={!!errors.password}
              helperText={errors.password}
              sx={{
                mb: 2,
                '& .MuiOutlinedInput-root': {
                  borderRadius: '8px',
                  bgcolor: '#2d3748',
                  '& fieldset': { borderColor: '#4a5568' },
                  '&:hover fieldset': { borderColor: '#718096' },
                  '&.Mui-focused fieldset': { borderColor: '#4299e1' },
                },
                '& .MuiInputBase-input': { color: '#e2e8f0', fontSize: '1rem', padding: '16px' },
                '& .MuiInputBase-input::placeholder': { color: '#a0aec0', opacity: 1 },
                '& .MuiFormHelperText-root': { color: '#f56565' }
              }}
            />

            <FormControlLabel
              control={
                <Checkbox
                  checked={acceptOffers}
                  onChange={(e) => setAcceptOffers(e.target.checked)}
                  sx={{ color: '#4a5568', '&.Mui-checked': { color: '#4299e1' } }}
                />
              }
              label={<Typography variant="body2" sx={{ color: '#a0aec0' }}>Send me special offers, personalized recommendations.</Typography>}
              sx={{ mb: 2 }}
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
              {loading ? 'Creating account...' : 'Continue with email'}
            </Button>
          </Box>

          <Box sx={{ mb: 3 }}>
            <Divider sx={{ mb: 2, borderColor: '#4a5568' }}>
              <Typography variant="body2" sx={{ color: '#a0aec0', px: 2 }}>Other sign up options</Typography>
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

          <Typography
            variant="body2"
            sx={{ textAlign: 'center', color: '#a0aec0', mb: 2, fontSize: '0.85rem' }}
          >
            By signing up, you agree to our{' '}
            <Link href="#" sx={{ color: '#4299e1', textDecoration: 'none' }}>Terms of Use</Link>
            {' '}and{' '}
            <Link href="#" sx={{ color: '#4299e1', textDecoration: 'none' }}>Privacy Policy</Link>
            .
          </Typography>

          <Box sx={{ textAlign: 'center' }}>
            <Typography variant="body2" sx={{ color: '#a0aec0' }}>
              Already have an account?{' '}
              <Link href="/login" sx={{ color: '#4299e1', textDecoration: 'none', fontWeight: '600', '&:hover': { textDecoration: 'underline' } }}>
                Log in
              </Link>
            </Typography>
          </Box>
        </Paper>
      </Box>
    </ThemeProvider>
  );
};

export default RegisterPage;