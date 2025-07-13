import React, { type ChangeEventHandler, useState, useEffect, useRef } from 'react';
import { useModalStore } from 'store/useModalStore';
import { useAuth } from 'contexts/AuthContext';
import { 
  Dialog, 
  DialogTitle, 
  TextField, 
  Button, 
  CircularProgress, 
  Typography, 
  Box,
  Divider,
  IconButton,
  Checkbox,
  FormControlLabel,
  Link
} from '@mui/material';
import { Google as GoogleIcon, Apple as AppleIcon, Email as EmailIcon } from '@mui/icons-material';
import { type FormData } from '@types';
import axios from 'utils/axios';

interface Props {}

const AuthModal: React.FC<Props> = () => {
  const { login, register } = useAuth();
  const { currentModal, setCurrentModal } = useModalStore();

  const isRegisterMode = currentModal === 'REGISTER';
  const isVerifyMode = currentModal === 'VERIFY';
  const isOpen = ['LOGIN', 'REGISTER', 'VERIFY'].includes(currentModal);
  const onClose = () => setCurrentModal('');

  const [formData, setFormData] = useState<FormData>({ name: '', surname: '', organizationName: '', email: '', password: '' });
  const [verificationCode, setVerificationCode] = useState('');
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState('');
  const [successMessage, setSuccessMessage] = useState('');
  const [resendEnabled, setResendEnabled] = useState(false);
  const [resendMessage, setResendMessage] = useState('');
  const [acceptOffers, setAcceptOffers] = useState(false);

  const timerRef = useRef<NodeJS.Timeout | null>(null);

  useEffect(() => {
    if (successMessage && isVerifyMode) {
      timerRef.current = setTimeout(() => setResendEnabled(true), 900000); // 15 minutes
      return () => {
        if (timerRef.current) clearTimeout(timerRef.current);
      };
    }
  }, [successMessage, isVerifyMode]);

  const handleChange: ChangeEventHandler<HTMLInputElement> = (e) => {
    const { name, value } = e.target;
    if (name === 'verificationCode') {
      setVerificationCode(value);
    } else {
      setFormData((prev) => ({ ...prev, [name]: value }));
    }
  };

  const clickSubmit = async () => {
    setLoading(true);
    setError('');
    setSuccessMessage('');

    try {
      if (isVerifyMode) {
        const response = await axios.post('/auth/verify', { code: verificationCode, email: formData.email });
        if (response.status === 200) {
          await login({ email: formData.email, password: formData.password });
          setCurrentModal('LOGIN');
        }
      } else if (isRegisterMode) {
        // Définir role comme un littéral 'user' ou 'admin'
        const registrationData = {
          ...formData,
          role: 'user' as const, // Utilisation de 'as const' pour forcer le type littéral
          canInvite: false,
          isVerified: false,
          mfaEnabled: false,
        };
        await register(registrationData);
        setSuccessMessage(`Enter the 6-digit code we sent to ${formData.email} to finish your login`);
        setCurrentModal('VERIFY');
      } else {
        await login({ email: formData.email, password: formData.password });
        onClose();
      }
    } catch (error: any) {
      if (isVerifyMode && error.response?.status === 400) {
        setError('Incorrect code');
      } else if (isRegisterMode && error.response?.status === 400) {
        setError('Registration failed. Please check your details.');
      } else {
        setError(typeof error === 'string' ? error : JSON.stringify(error));
      }
    }

    setLoading(false);
  };

  const resendCode = async () => {
    setLoading(true);
    setResendMessage('');
    try {
      // Définir role comme un littéral 'user' ou 'admin'
      const registrationData = {
        ...formData,
        role: 'user' as const, // Utilisation de 'as const' pour forcer le type littéral
        canInvite: false,
        isVerified: false,
        mfaEnabled: false,
      };
      await register(registrationData);
      setResendMessage('Verification code resent. Please check your email.');
      setResendEnabled(false);
      timerRef.current = setTimeout(() => setResendEnabled(true), 900000);
    } catch (error: any) {
      setError(typeof error === 'string' ? error : JSON.stringify(error));
    }
    setLoading(false);
  };

  const isSubmitButtonDisabled = isVerifyMode
    ? !verificationCode || verificationCode.length !== 6
    : isRegisterMode
    ? !formData.name || !formData.surname || !formData.organizationName || !formData.email || !formData.password
    : !formData.email || !formData.password;

  return (
    <Dialog 
      open={isOpen} 
      onClose={onClose} 
      maxWidth="sm"
      fullWidth
      PaperProps={{ 
        style: { 
          borderRadius: '16px',
          padding: '0',
          maxWidth: '480px',
          backgroundColor: '#ffffff'
        } 
      }}
    >
      <Box sx={{ p: 4 }}>
        {isVerifyMode ? (
          <>
            <DialogTitle 
              sx={{ 
                textAlign: 'center', 
                fontSize: '2rem', 
                fontWeight: 'bold',
                color: '#2d3748',
                p: 0,
                mb: 3
              }}
            >
              Check your inbox
            </DialogTitle>
            
            <Typography 
              variant="body1" 
              sx={{ 
                textAlign: 'center', 
                mb: 3,
                color: '#4a5568',
                fontSize: '1rem'
              }}
            >
              Enter the 6-digit code we sent to{' '}
              <strong style={{ color: '#4299e1' }}>{formData.email}</strong>{' '}
              to finish your login.
            </Typography>

            <TextField
              placeholder="6-digit code"
              name="verificationCode"
              type="text"
              value={verificationCode}
              onChange={handleChange}
              fullWidth
              variant="outlined"
              sx={{ 
                mb: 2,
                '& .MuiOutlinedInput-root': {
                  borderRadius: '8px',
                  backgroundColor: '#f7fafc',
                  '& fieldset': {
                    borderColor: '#e2e8f0',
                  },
                  '&:hover fieldset': {
                    borderColor: '#cbd5e0',
                  },
                  '&.Mui-focused fieldset': {
                    borderColor: '#4299e1',
                  },
                }
              }}
              required
              inputProps={{ 
                maxLength: 6,
                style: { fontSize: '1.1rem', padding: '16px' }
              }}
            />

            {error && (
              <Typography 
                variant="body2" 
                sx={{ 
                  mb: 2, 
                  color: '#e53e3e', 
                  textAlign: 'center',
                  fontSize: '0.9rem'
                }}
              >
                {error}
              </Typography>
            )}

            <Button
              onClick={clickSubmit}
              disabled={isSubmitButtonDisabled || loading}
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
                '&:hover': { 
                  backgroundColor: '#3182ce' 
                },
                '&:disabled': {
                  backgroundColor: '#a0aec0',
                  color: 'white'
                }
              }}
              startIcon={loading ? <CircularProgress size={20} color="inherit" /> : <EmailIcon />}
            >
              {loading ? 'Logging in...' : 'Log in'}
            </Button>

            <Box sx={{ textAlign: 'center', mb: 2 }}>
              <Typography variant="body2" sx={{ color: '#718096' }}>
                Didn't receive the code?{' '}
                <Button
                  onClick={resendCode}
                  disabled={!resendEnabled || loading}
                  sx={{ 
                    color: '#4299e1',
                    textTransform: 'none',
                    p: 0,
                    minWidth: 'auto',
                    fontWeight: '600'
                  }}
                >
                  Resend code
                </Button>
              </Typography>
            </Box>

            {resendMessage && (
              <Typography 
                variant="body2" 
                sx={{ 
                  mb: 2, 
                  color: '#38a169', 
                  textAlign: 'center',
                  fontSize: '0.9rem'
                }}
              >
                {resendMessage}
              </Typography>
            )}

            <Box sx={{ textAlign: 'center' }}>
              <Button
                onClick={() => setCurrentModal('LOGIN')}
                sx={{ 
                  color: '#4299e1',
                  textTransform: 'none',
                  fontWeight: '600'
                }}
              >
                Log in to a different account
              </Button>
            </Box>
          </>
        ) : (
          <>
            <DialogTitle 
              sx={{ 
                textAlign: 'center', 
                fontSize: '2rem', 
                fontWeight: 'bold',
                color: '#2d3748',
                p: 0,
                mb: 3
              }}
            >
              {isRegisterMode ? 'Sign up with email' : 'Sign in to your account'}
            </DialogTitle>

            <Box sx={{ mb: 3 }}>
              {isRegisterMode && (
                <>
                  <TextField
                    placeholder="Name"
                    name="name"
                    type="text"
                    value={formData.name || ''}
                    onChange={handleChange}
                    fullWidth
                    variant="outlined"
                    sx={{ 
                      mb: 2,
                      '& .MuiOutlinedInput-root': {
                        borderRadius: '8px',
                        backgroundColor: '#f7fafc',
                        '& fieldset': {
                          borderColor: '#e2e8f0',
                        },
                        '&:hover fieldset': {
                          borderColor: '#cbd5e0',
                        },
                        '&.Mui-focused fieldset': {
                          borderColor: '#4299e1',
                        },
                      }
                    }}
                    required
                    inputProps={{ 
                      style: { fontSize: '1rem', padding: '16px' }
                    }}
                  />
                  <TextField
                    placeholder="Surname"
                    name="surname"
                    type="text"
                    value={formData.surname || ''}
                    onChange={handleChange}
                    fullWidth
                    variant="outlined"
                    sx={{ 
                      mb: 2,
                      '& .MuiOutlinedInput-root': {
                        borderRadius: '8px',
                        backgroundColor: '#f7fafc',
                        '& fieldset': {
                          borderColor: '#e2e8f0',
                        },
                        '&:hover fieldset': {
                          borderColor: '#cbd5e0',
                        },
                        '&.Mui-focused fieldset': {
                          borderColor: '#4299e1',
                        },
                      }
                    }}
                    required
                    inputProps={{ 
                      style: { fontSize: '1rem', padding: '16px' }
                    }}
                  />
                  <TextField
                    placeholder="Organization Name"
                    name="organizationName"
                    type="text"
                    value={formData.organizationName || ''}
                    onChange={handleChange}
                    fullWidth
                    variant="outlined"
                    sx={{ 
                      mb: 2,
                      '& .MuiOutlinedInput-root': {
                        borderRadius: '8px',
                        backgroundColor: '#f7fafc',
                        '& fieldset': {
                          borderColor: '#e2e8f0',
                        },
                        '&:hover fieldset': {
                          borderColor: '#cbd5e0',
                        },
                        '&.Mui-focused fieldset': {
                          borderColor: '#4299e1',
                        },
                      }
                    }}
                    required
                    inputProps={{ 
                      style: { fontSize: '1rem', padding: '16px' }
                    }}
                  />
                </>
              )}
              
              <TextField
                placeholder="Email"
                name="email"
                type="email"
                value={formData.email || ''}
                onChange={handleChange}
                fullWidth
                variant="outlined"
                sx={{ 
                  mb: 2,
                  '& .MuiOutlinedInput-root': {
                    borderRadius: '8px',
                    backgroundColor: '#f7fafc',
                    '& fieldset': {
                      borderColor: '#e2e8f0',
                    },
                    '&:hover fieldset': {
                      borderColor: '#cbd5e0',
                    },
                    '&.Mui-focused fieldset': {
                      borderColor: '#4299e1',
                    },
                  }
                }}
                required
                inputProps={{ 
                  style: { fontSize: '1rem', padding: '16px' }
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
                sx={{ 
                  mb: isRegisterMode ? 2 : 3,
                  '& .MuiOutlinedInput-root': {
                    borderRadius: '8px',
                    backgroundColor: '#f7fafc',
                    '& fieldset': {
                      borderColor: '#e2e8f0',
                    },
                    '&:hover fieldset': {
                      borderColor: '#cbd5e0',
                    },
                    '&.Mui-focused fieldset': {
                      borderColor: '#4299e1',
                    },
                  }
                }}
                required
                inputProps={{ 
                  style: { fontSize: '1rem', padding: '16px' }
                }}
              />

              {isRegisterMode && (
                <FormControlLabel
                  control={
                    <Checkbox
                      checked={acceptOffers}
                      onChange={(e) => setAcceptOffers(e.target.checked)}
                      sx={{
                        color: '#a0aec0',
                        '&.Mui-checked': {
                          color: '#4299e1',
                        },
                      }}
                    />
                  }
                  label={
                    <Typography variant="body2" sx={{ color: '#4a5568' }}>
                      Send me special offers, personalized recommendations.
                    </Typography>
                  }
                  sx={{ mb: 2 }}
                />
              )}
            </Box>

            {error && (
              <Typography 
                variant="body2" 
                sx={{ 
                  mb: 2, 
                  color: '#e53e3e', 
                  textAlign: 'center',
                  fontSize: '0.9rem'
                }}
              >
                {error}
              </Typography>
            )}

            {loading ? (
              <Box sx={{ display: 'flex', justifyContent: 'center', mb: 3 }}>
                <CircularProgress sx={{ color: '#4299e1' }} />
              </Box>
            ) : (
              <Button
                onClick={clickSubmit}
                disabled={isSubmitButtonDisabled}
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
                  '&:hover': { 
                    backgroundColor: '#3182ce' 
                  },
                  '&:disabled': {
                    backgroundColor: '#a0aec0',
                    color: 'white'
                  }
                }}
                startIcon={<EmailIcon />}
              >
                {isRegisterMode ? 'Continue with email' : 'Sign in with email'}
              </Button>
            )}

            <Box sx={{ mb: 3 }}>
              <Divider sx={{ mb: 2, color: '#a0aec0' }}>
                <Typography variant="body2" sx={{ color: '#718096', px: 2 }}>
                  Other sign {isRegisterMode ? 'up' : 'in'} options
                </Typography>
              </Divider>
              
              <Box sx={{ display: 'flex', gap: 2, justifyContent: 'center' }}>
                <IconButton
                  sx={{
                    border: '2px solid #e2e8f0',
                    borderRadius: '8px',
                    width: '60px',
                    height: '60px',
                    '&:hover': {
                      borderColor: '#cbd5e0',
                      backgroundColor: '#f7fafc'
                    }
                  }}
                >
                  <GoogleIcon sx={{ color: '#4285f4', fontSize: '24px' }} />
                </IconButton>
                <IconButton
                  sx={{
                    border: '2px solid #e2e8f0',
                    borderRadius: '8px',
                    width: '60px',
                    height: '60px',
                    '&:hover': {
                      borderColor: '#cbd5e0',
                      backgroundColor: '#f7fafc'
                    }
                  }}
                >
                  <AppleIcon sx={{ color: '#000000', fontSize: '24px' }} />
                </IconButton>
              </Box>
            </Box>

            {isRegisterMode && (
              <Typography 
                variant="body2" 
                sx={{ 
                  textAlign: 'center', 
                  color: '#718096',
                  mb: 2,
                  fontSize: '0.85rem'
                }}
              >
                By signing up, you agree to our{' '}
                <Link href="#" sx={{ color: '#4299e1', textDecoration: 'none' }}>
                  Terms of Use
                </Link>
                {' '}and{' '}
                <Link href="#" sx={{ color: '#4299e1', textDecoration: 'none' }}>
                  Privacy Policy
                </Link>
                .
              </Typography>
            )}

            <Box sx={{ textAlign: 'center' }}>
              <Typography variant="body2" sx={{ color: '#4a5568' }}>
                {isRegisterMode ? 'Already have an account?' : "Don't have an account?"}{' '}
                <Button
                  onClick={() => setCurrentModal(isRegisterMode ? 'LOGIN' : 'REGISTER')}
                  sx={{ 
                    color: '#4299e1',
                    textTransform: 'none',
                    p: 0,
                    minWidth: 'auto',
                    fontWeight: '600'
                  }}
                >
                  {isRegisterMode ? 'Log in' : 'Sign up'}
                </Button>
              </Typography>
            </Box>
          </>
        )}
      </Box>
    </Dialog>
  );
};

export default AuthModal;