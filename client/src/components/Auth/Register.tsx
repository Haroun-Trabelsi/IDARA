// src/components/Auth/RegisterPage.tsx
"use client";

import React, { useState, useEffect, useCallback } from 'react';
import {
  Box,
  TextField,
  Button,
  Typography,
  CircularProgress,
  FormControlLabel,
  Checkbox,
  IconButton,
  Menu,
  MenuItem,
  Select,
  FormControl,
  InputLabel,
  InputAdornment
} from '@mui/material';
import {
  Email as EmailIcon,
  Language as LanguageIcon,
  Visibility,
  VisibilityOff,
  KeyboardArrowDown
} from '@mui/icons-material';
import { useNavigate, useLocation } from 'react-router-dom';
import { validateProfile, validatePassword } from '../../utils/validation';
import axios from 'axios';
import debounce from 'lodash.debounce';

// Import shared AppTheme wrapper that provides ThemeProvider + CssBaseline
import AppTheme from './AppTheme';

const teamSizeOptions = [
  { value: '1', label: '1' },
  { value: '2-10', label: '2-10' },
  { value: '11-20', label: '11-20' },
  { value: '21-50', label: '21-50' },
  { value: '51-100', label: '51-100' },
  { value: '101-200', label: '101-200' },
  { value: '201-500', label: '201-500' },
  { value: '500+', label: '500+' },
];

const regionOptions = [
  { value: 'Europe', label: 'Europe' },
  { value: 'US East', label: 'US East' },
  { value: 'US West', label: 'US West' },
  { value: 'South America', label: 'South America' },
  { value: 'East Asia', label: 'East Asia' },
  { value: 'Australia', label: 'Australia' },
  { value: 'Singapore', label: 'Singapore' },
  { value: 'China', label: 'China' },
];

interface RegisterFormData {
  name: string;
  surname: string;
  organizationName: string;
  email: string;
  password: string;
  teamSize: string;
  region: string;
}

const RegisterPage: React.FC = () => {
  const navigate = useNavigate();
  const location = useLocation();
  const [formData, setFormData] = useState<RegisterFormData>(() => {
    const stateData = (location.state as any)?.formData;
    const initialData = stateData || {};
    return {
      name: initialData.name || '',
      surname: initialData.surname || '',
      organizationName: initialData.organizationName || '',
      email: initialData.email || '',
      password: initialData.password || '',
      teamSize: initialData.teamSize || '',
      region: initialData.region || ''
    };
  });
  const [accountId, setAccountId] = useState<string | null>(
  (location.state as any)?.accountId || null
  );
  const [errors, setErrors] = useState({
    name: '',
    surname: '',
    organizationName: '',
    email: '',
    password: '',
    teamSize: '',
    region: ''
  });
  const [loading, setLoading] = useState(false);
  const [acceptOffers, setAcceptOffers] = useState(false);
  const [showPassword, setShowPassword] = useState(false);
  const [languageAnchor, setLanguageAnchor] = useState<null | HTMLElement>(null);
  const [selectedLanguage, setSelectedLanguage] = useState('EN');
  const [orgNameChecking, setOrgNameChecking] = useState(false);

  useEffect(() => {
  console.log('Formulaire initialisé avec:', formData);
  console.log('Account ID:', accountId);
  console.log('Données de location.state:', (location.state as any)?.formData);
  }, [formData, location.state, accountId]);

  const debouncedCheckOrganizationName = useCallback(
    debounce(async (name: string) => {
      if (!name || name.trim() === '') {
        console.log('Nom d\'organisation vide, requête non envoyée:', name);
        return;
      }
      console.log('Envoi de la requête pour vérifier le nom d\'organisation:', name);
      setOrgNameChecking(true);
      try {
        await axios.post('http://localhost:8080/col/check-organization-name', { organizationName: name.trim() });
        setErrors(prev => ({ ...prev, organizationName: '' }));
      } catch (err: any) {
        console.log('Erreur lors de la vérification du nom d\'organisation:', err.response?.data || err);
        setErrors(prev => ({
          ...prev,
          organizationName: err.response?.data?.message || 'Erreur lors de la vérification du nom de l\'organisation'
        }));
      } finally {
        setOrgNameChecking(false);
      }
    }, 500),
    []
  );

  const handleChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const { name, value } = e.target;
    setFormData(prev => ({ ...prev, [name]: value }));
    setErrors(prev => ({ ...prev, [name]: '' }));
    if (name === 'organizationName') {
      debouncedCheckOrganizationName(value);
    }
  };

  const handleSelectChange = (name: string, value: string) => {
    setFormData(prev => ({ ...prev, [name]: value }));
    setErrors(prev => ({ ...prev, [name]: '' }));
  };

  const handleLanguageClick = (event: React.MouseEvent<HTMLElement>) => {
    setLanguageAnchor(event.currentTarget);
  };

  const handleLanguageClose = () => {
    setLanguageAnchor(null);
  };

  const handleLanguageSelect = (language: string) => {
    setSelectedLanguage(language);
    handleLanguageClose();
  };

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    setLoading(true);
    setErrors({ name: '', surname: '', organizationName: '', email: '', password: '', teamSize: '', region: '' });

    const profileValidation = validateProfile({
      name: formData.name,
      surname: formData.surname,
      email: formData.email,
      teamSize: formData.teamSize,
      region: formData.region,
      organizationName: formData.organizationName
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
        organizationName: profileValidation.errors.organizationName,
        email: profileValidation.errors.email,
        password: passwordValidation.errors.newPassword || passwordValidation.errors.currentPassword,
        teamSize: profileValidation.errors.teamSize,
        region: profileValidation.errors.region
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
        status: 'AdministratorOrganization',
      };

      console.log('Envoi des données:', accountId ? 'Mise à jour du compte' : 'Inscription', registrationData);

      let newAccountId = accountId;
      if (accountId) {
        // update existing account
        const response = await axios.put('http://localhost:8080/auth/update-account', {
          ...registrationData,
          accountId
        });
        console.log('Mise à jour réussie:', response.data);
        newAccountId = accountId;
      } else {
        // create new account
        const response = await axios.post('http://localhost:8080/auth/register', registrationData);
        setAccountId(response.data.data._id);
        console.log('Inscription réussie, nouvel accountId:', response.data.data._id);
        newAccountId = response.data.data._id;
      }

      console.log('Redirection vers /verify-email avec email:', formData.email);
      navigate('/verify-email', { state: { email: formData.email, formData, accountId: newAccountId } });
    } catch (err: any) {
      console.error('Erreur:', accountId ? 'Mise à jour' : 'Inscription', err);
      setErrors(prev => ({
        ...prev,
        email: err.response?.data?.message?.includes('email') ? err.response?.data?.message : prev.email,
        password: err.response?.data?.message?.includes('password') ? err.response?.data?.message : prev.password,
        teamSize: err.response?.data?.message?.includes('teamSize') ? err.response?.data?.message : prev.teamSize,
        region: err.response?.data?.message?.includes('region') ? err.response?.data?.message : prev.region,
        organizationName: err.response?.data?.message?.includes('organization') ? err.response?.data?.message : prev.organizationName
      }));
    } finally {
      setLoading(false);
    }
  };

  const isSubmitDisabled = !formData.name || !formData.surname || !formData.organizationName || !formData.email || !formData.password || !formData.teamSize || !formData.region || !!errors.name || !!errors.surname || !!errors.organizationName || !!errors.email || !!errors.password || !!errors.teamSize || !!errors.region;

  return (
    <AppTheme>
      <Box
        sx={{
          minHeight: '100vh',
          display: 'flex',
          flexDirection: 'column',
          alignItems: 'center',
          justifyContent: 'center',
          bgcolor: '#0a0a0a',
          backgroundImage: "radial-gradient(circle at 50% 50%, rgba(66, 153, 225, 0.03) 0%, transparent 50%)",
          p: 3,
          position: 'relative'
        }}
      >
        {/* Header with logo and language selector */}
        <Box
          sx={{
            position: 'absolute',
            top: 24,
            left: 24,
            right: 24,
            display: 'flex',
            justifyContent: 'space-between',
            alignItems: 'center'
          }}
        >
          <Box sx={{ display: 'flex', alignItems: 'center', gap: 2 }}>
            <img 
              src="/ICONEFRACK.png" 
              alt="ftrack" 
              style={{ 
                height: '32px', 
                width: 'auto',
                filter: 'brightness(0) invert(1)'
              }} 
            />
          </Box>
          
          <Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
            <Typography 
              variant="body2" 
              sx={{ color: '#b3b3b3', fontSize: '0.9rem' }}
            >
              {selectedLanguage}
            </Typography>
            <IconButton
              onClick={handleLanguageClick}
              sx={{ 
                color: '#b3b3b3',
                '&:hover': { 
                  color: '#4299e1',
                  backgroundColor: 'rgba(66, 153, 225, 0.1)' 
                }
              }}
            >
              <LanguageIcon />
            </IconButton>
          </Box>
          
          <Menu
            anchorEl={languageAnchor}
            open={Boolean(languageAnchor)}
            onClose={handleLanguageClose}
            PaperProps={{
              sx: {
                bgcolor: '#1a1a1a',
                border: '1px solid rgba(255, 255, 255, 0.1)',
                '& .MuiMenuItem-root': {
                  color: '#ffffff',
                  '&:hover': {
                    backgroundColor: 'rgba(66, 153, 225, 0.1)',
                  }
                }
              }
            }}
          >
            <MenuItem onClick={() => handleLanguageSelect('EN')}>English</MenuItem>
            <MenuItem onClick={() => handleLanguageSelect('FR')}>Français</MenuItem>
            <MenuItem onClick={() => handleLanguageSelect('ES')}>Español</MenuItem>
          </Menu>
        </Box>

        {/* Main content */}
        <Box
          sx={{
            width: '100%',
            maxWidth: '420px',
            mt: 8
          }}
        >
          <Typography
            variant="h4"
            sx={{
              textAlign: 'center',
              color: '#ffffff',
              mb: 1,
              fontWeight: 500
            }}
          >
            Create Account
          </Typography>
          
          <Typography
            variant="body2"
            sx={{
              textAlign: 'center',
              color: '#888888',
              mb: 4
            }}
          >
            Join ftrack and start managing your creative projects
          </Typography>

          <Box component="form" onSubmit={handleSubmit}>
            <Box sx={{ display: 'flex', gap: 2, mb: 3 }}>
              <TextField
                placeholder="First name"
                name="name"
                type="text"
                value={formData.name}
                onChange={handleChange}
                fullWidth
                variant="outlined"
                required
                error={!!errors.name}
                helperText={errors.name}
              />
              <TextField
                placeholder="Last name"
                name="surname"
                type="text"
                value={formData.surname}
                onChange={handleChange}
                fullWidth
                variant="outlined"
                required
                error={!!errors.surname}
                helperText={errors.surname}
              />
            </Box>

            <TextField
              placeholder="Organization name"
              name="organizationName"
              type="text"
              value={formData.organizationName}
              onChange={handleChange}
              fullWidth
              variant="outlined"
              required
              error={!!errors.organizationName}
              helperText={errors.organizationName}
              sx={{ mb: 3 }}
              InputProps={{
                endAdornment: orgNameChecking ? (
                  <InputAdornment position="end">
                    <CircularProgress size={16} sx={{ color: '#888888' }} />
                  </InputAdornment>
                ) : null,
              }}
            />

            <FormControl 
              fullWidth 
              variant="outlined"
              error={!!errors.teamSize}
              sx={{ mb: 3 }}
            >
              <InputLabel 
                sx={{ 
                  color: '#888888',
                  '&.Mui-focused': { color: '#4299e1' },
                  fontSize: '0.95rem',
                  transform: formData.teamSize ? 'translate(14px, -9px) scale(0.75)' : 'translate(14px, 16px) scale(1)',
                  transition: 'all 0.2s ease',
                  backgroundColor: formData.teamSize ? '#0a0a0a' : 'transparent',
                  padding: formData.teamSize ? '0 4px' : '0',
                }}
              >
                Team size interested in ftrack Studio
              </InputLabel>
              <Select
                value={formData.teamSize}
                onChange={(e) => handleSelectChange('teamSize', e.target.value as string)}
                IconComponent={KeyboardArrowDown}
                sx={{
                  backgroundColor: 'rgba(255, 255, 255, 0.05)',
                  borderRadius: '6px',
                  border: '1px solid rgba(255, 255, 255, 0.1)',
                  '& .MuiOutlinedInput-notchedOutline': { border: 'none' },
                  '&:hover': {
                    backgroundColor: 'rgba(255, 255, 255, 0.08)',
                    borderColor: 'rgba(66, 153, 225, 0.3)',
                  },
                  '&.Mui-focused': {
                    backgroundColor: 'rgba(255, 255, 255, 0.08)',
                    borderColor: '#4299e1',
                    boxShadow: '0 0 0 2px rgba(66, 153, 225, 0.1)',
                  },
                  '& .MuiSelect-select': {
                    color: '#ffffff',
                    fontSize: '0.95rem',
                    padding: '14px 16px',
                  },
                  '& .MuiSelect-icon': { color: '#888888' },
                }}
                MenuProps={{
                  PaperProps: {
                    sx: {
                      bgcolor: '#1a1a1a',
                      border: '1px solid rgba(255, 255, 255, 0.1)',
                      maxHeight: '300px',
                      '& .MuiMenuItem-root': {
                        color: '#ffffff',
                        fontSize: '0.95rem',
                        '&:hover': {
                          backgroundColor: 'rgba(66, 153, 225, 0.1)',
                        },
                        '&.Mui-selected': {
                          backgroundColor: 'rgba(66, 153, 225, 0.2)',
                          '&:hover': {
                            backgroundColor: 'rgba(66, 153, 225, 0.3)',
                          },
                        },
                      },
                    },
                  },
                }}
              >
                {teamSizeOptions.map((option) => (
                  <MenuItem key={option.value} value={option.value}>
                    {option.label}
                  </MenuItem>
                ))}
              </Select>
            </FormControl>

            <FormControl 
              fullWidth 
              variant="outlined"
              error={!!errors.region}
              sx={{ mb: 3 }}
            >
              <InputLabel 
                sx={{ 
                  color: '#888888',
                  '&.Mui-focused': { color: '#4299e1' },
                  fontSize: '0.95rem',
                  transform: formData.region ? 'translate(14px, -9px) scale(0.75)' : 'translate(14px, 16px) scale(1)',
                  transition: 'all 0.2s ease',
                  backgroundColor: formData.region ? '#0a0a0a' : 'transparent',
                  padding: formData.region ? '0 4px' : '0',
                }}
              >
                Region where your workspace will be hosted
              </InputLabel>
              <Select
                value={formData.region}
                onChange={(e) => handleSelectChange('region', e.target.value as string)}
                IconComponent={KeyboardArrowDown}
                sx={{
                  backgroundColor: 'rgba(255, 255, 255, 0.05)',
                  borderRadius: '6px',
                  border: '1px solid rgba(255, 255, 255, 0.1)',
                  '& .MuiOutlinedInput-notchedOutline': { border: 'none' },
                  '&:hover': {
                    backgroundColor: 'rgba(255, 255, 255, 0.08)',
                    borderColor: 'rgba(66, 153, 225, 0.3)',
                  },
                  '&.Mui-focused': {
                    backgroundColor: 'rgba(255, 255, 255, 0.08)',
                    borderColor: '#4299e1',
                    boxShadow: '0 0 0 2px rgba(66, 153, 225, 0.1)',
                  },
                  '& .MuiSelect-select': {
                    color: '#ffffff',
                    fontSize: '0.95rem',
                    padding: '14px 16px',
                  },
                  '& .MuiSelect-icon': { color: '#888888' },
                }}
                MenuProps={{
                  PaperProps: {
                    sx: {
                      bgcolor: '#1a1a1a',
                      border: '1px solid rgba(255, 255, 255, 0.1)',
                      maxHeight: '300px',
                      '& .MuiMenuItem-root': {
                        color: '#ffffff',
                        fontSize: '0.95rem',
                        '&:hover': {
                          backgroundColor: 'rgba(66, 153, 225, 0.1)',
                        },
                        '&.Mui-selected': {
                          backgroundColor: 'rgba(66, 153, 225, 0.2)',
                          '&:hover': {
                            backgroundColor: 'rgba(66, 153, 225, 0.3)',
                          },
                        },
                      },
                    },
                  },
                }}
              >
                {regionOptions.map((option) => (
                  <MenuItem key={option.value} value={option.value}>
                    {option.label}
                  </MenuItem>
                ))}
              </Select>
            </FormControl>

            <TextField
              placeholder="Email address"
              name="email"
              type="email"
              value={formData.email}
              onChange={handleChange}
              fullWidth
              variant="outlined"
              required
              error={!!errors.email}
              helperText={errors.email}
              sx={{ mb: 3 }}
            />

            <TextField
              placeholder="Password"
              name="password"
              type={showPassword ? 'text' : 'password'}
              value={formData.password}
              onChange={handleChange}
              fullWidth
              variant="outlined"
              required
              error={!!errors.password}
              helperText={errors.password}
              sx={{ mb: 3 }}
              InputProps={{
                endAdornment: (
                  <IconButton
                    onClick={() => setShowPassword(!showPassword)}
                    edge="end"
                    sx={{ color: '#888888' }}
                  >
                    {showPassword ? <VisibilityOff /> : <Visibility />}
                  </IconButton>
                ),
              }}
            />

            <FormControlLabel
              control={
                <Checkbox
                  checked={acceptOffers}
                  onChange={(e) => setAcceptOffers(e.target.checked)}
                  sx={{ 
                    color: '#888888', 
                    '&.Mui-checked': { color: '#4299e1' },
                    '& .MuiSvgIcon-root': { fontSize: '1.2rem' }
                  }}
                />
              }
              label={
                <Typography variant="body2" sx={{ color: '#b3b3b3', fontSize: '0.85rem' }}>
                  I want to contribute to improving the product by suggesting new features and receiving the latest updates
                </Typography>
              }
              sx={{ mb: 4 }}
            />

            <Button
              type="submit"
              disabled={isSubmitDisabled || loading}
              fullWidth
              variant="contained"
              sx={{
                mb: 4,
                backgroundColor: '#4299e1',
                color: '#000000',
                py: 1.75,
                fontSize: '0.95rem',
                fontWeight: 600,
                '&:hover': { 
                  backgroundColor: '#3182ce',
                  boxShadow: '0 4px 12px rgba(66, 153, 225, 0.3)'
                },
                '&:disabled': { 
                  backgroundColor: '#333333', 
                  color: '#666666' 
                },
                boxShadow: '0 2px 8px rgba(66, 153, 225, 0.2)',
              }}
              startIcon={loading ? <CircularProgress size={18} sx={{ color: '#666666' }} /> : <EmailIcon />}
            >
              {loading ? 'Processing...' : (accountId ? 'Update Account' : 'Create Account')}
            </Button>

            <Typography
              variant="body2"
              sx={{ 
                textAlign: 'center', 
                color: '#666666', 
                mb: 3, 
                fontSize: '0.8rem',
                lineHeight: 1.4
              }}
            >
              By creating an account, you agree to our{' '}
              <Box component="span" sx={{ color: '#4299e1', cursor: 'pointer', '&:hover': { textDecoration: 'underline' } }}>
                Terms of Service
              </Box>
              {' '}and{' '}
              <Box component="span" sx={{ color: '#4299e1', cursor: 'pointer', '&:hover': { textDecoration: 'underline' } }}>
                Privacy Policy
              </Box>
            </Typography>

            <Box sx={{ textAlign: 'center' }}>
              <Typography variant="body2" sx={{ color: '#888888', fontSize: '0.9rem', mb: 2 }}>
                Already have an account?{' '}
                <Box 
                  component="span" 
                  sx={{ 
                    color: '#4299e1', 
                    cursor: 'pointer',
                    fontWeight: 500,
                    '&:hover': { textDecoration: 'underline' } 
                  }}
                  onClick={() => {
                    navigate('/login');
                  }}
                >
                  Sign in
                </Box>
              </Typography>
              <Typography 
                variant="body2" 
                sx={{ 
                  color: '#555555', 
                  fontSize: '0.75rem' 
                }}
              >
                © 2025 ftrack. All rights reserved.
              </Typography>
            </Box>
          </Box>
        </Box>
      </Box>
    </AppTheme>
  );
};

export default RegisterPage;
