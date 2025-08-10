// src/components/Auth/Login.tsx
"use client";

import React, { useState, useEffect } from "react";
import { 
  Box, 
  TextField, 
  Button, 
  Typography, 
  CircularProgress,
  ThemeProvider,
  createTheme,
  CssBaseline,
  Dialog,
  DialogTitle,
  DialogContent,
  DialogActions,
  Button as DialogButton,
  IconButton,
  Menu,
  MenuItem,
} from "@mui/material";
import { 
  Email as EmailIcon,
  CheckCircle as CheckCircleIcon,
  Language as LanguageIcon,
} from "@mui/icons-material";
import { useNavigate } from 'react-router-dom';
import { useAuth } from 'contexts/AuthContext';
import axios from 'axios';
import { BACKEND_URL } from '../../constants/index'; // Importer BACKEND_URL

const darkTheme = createTheme({
  palette: {
    mode: "dark",
    primary: { main: "#4299e1" },
    secondary: { main: "#ff6b35" },
    background: {
      default: "#0a0a0a",
      paper: "#141414",
    },
    text: {
      primary: "#ffffff",
      secondary: "#b3b3b3",
    },
  },
  typography: {
    fontFamily: '"Roboto", "Arial", sans-serif',
    h4: {
      fontSize: "1.75rem",
      fontWeight: 500,
    },
    body1: {
      fontSize: "1rem",
    },
    body2: {
      fontSize: "0.875rem",
    },
  },
  components: {
    MuiCssBaseline: {
      styleOverrides: {
        body: {
          backgroundColor: "#0a0a0a",
          backgroundImage:
            "radial-gradient(circle at 50% 50%, rgba(66, 153, 225, 0.03) 0%, transparent 50%)",
        },
      },
    },
    MuiButton: {
      styleOverrides: {
        root: {
          borderRadius: "6px",
          textTransform: "none",
          fontWeight: 500,
          fontSize: "0.95rem",
        },
      },
    },
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
  const [message, setMessage] = useState<string | null>(null);
  const [openDialog, setOpenDialog] = useState(false);
  const [languageAnchor, setLanguageAnchor] = useState<null | HTMLElement>(null);
  const [selectedLanguage, setSelectedLanguage] = useState("EN");

  const handleChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const { name, value } = e.target;
    setFormData(prev => ({ ...prev, [name]: value }));
  };

  const handleSubmit = async (e: React.FormEvent<HTMLFormElement>) => {
    e.preventDefault();
    setLoading(true);
    setMessage(null);

    try {
      await login({ email: formData.email, password: formData.password });
    } catch (err: any) {
      setMessage(err.response?.data?.message || 'Invalid email or password');
    } finally {
      setLoading(false);
    }
  };

  const isSubmitDisabled = !formData.email || !formData.password;

  // Vérification après mise à jour de l'account avec logique basée sur le rôle
  useEffect(() => {
    if (account) {
      if (account.mustCompleteProfile === true) {
        navigate('/CompleteProfil');
      } else if (account.role === 'admin') {
        navigate('/AdminDashboard');
      } else {
        navigate('/');
      }
    }
  }, [account, navigate]);

  // Gestion de la demande de réinitialisation
  const handleForgotPassword = () => {
    setOpenDialog(true);
  };

  const handleConfirmReset = async () => {
    setLoading(true);
    try {
      await axios.post(`${BACKEND_URL}/auth/forgot-password`, { email: formData.email });
      setOpenDialog(false);
      navigate('/check-email', { state: { email: formData.email } });
    } catch (err: any) {
      setMessage(err.response?.data?.message || 'Failed to request password reset');
    } finally {
      setLoading(false);
    }
  };

  const handleCancelReset = () => {
    setOpenDialog(false);
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

  return (
    <ThemeProvider theme={darkTheme}>
      <CssBaseline />
      <Box
        sx={{
          minHeight: "100vh",
          display: "flex",
          flexDirection: "column",
          alignItems: "center",
          justifyContent: "center",
          bgcolor: "#0a0a0a",
          backgroundImage:
            "radial-gradient(circle at 50% 50%, rgba(66, 153, 225, 0.03) 0%, transparent 50%)",
          p: 3,
          position: "relative",
        }}
      >
        {/* Header with logo and language selector */}
        <Box
          sx={{
            position: "absolute",
            top: 24,
            left: 24,
            right: 24,
            display: "flex",
            justifyContent: "space-between",
            alignItems: "center",
          }}
        >
          <Box sx={{ display: "flex", alignItems: "center", gap: 2 }}>
            <img
              src="/ICONEFRACK.png"
              alt="ftrack"
              style={{
                height: "32px",
                width: "auto",
                filter: "brightness(0) invert(1)",
              }}
            />
          </Box>

          <Box sx={{ display: "flex", alignItems: "center", gap: 1 }}>
            <Typography variant="body2" sx={{ color: "#b3b3b3", fontSize: "0.9rem" }}>
              {selectedLanguage}
            </Typography>
            <IconButton
              onClick={handleLanguageClick}
              sx={{
                color: "#b3b3b3",
                "&:hover": {
                  color: "#4299e1",
                  backgroundColor: "rgba(66, 153, 225, 0.1)",
                },
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
                bgcolor: "#1a1a1a",
                border: "1px solid rgba(255, 255, 255, 0.1)",
                "& .MuiMenuItem-root": {
                  color: "#ffffff",
                  "&:hover": {
                    backgroundColor: "rgba(66, 153, 225, 0.1)",
                  },
                },
              },
            }}
          >
            <MenuItem onClick={() => handleLanguageSelect("EN")}>English</MenuItem>
            <MenuItem onClick={() => handleLanguageSelect("FR")}>Français</MenuItem>
            <MenuItem onClick={() => handleLanguageSelect("ES")}>Español</MenuItem>
          </Menu>
        </Box>

        {/* Main content */}
        <Box sx={{ width: "100%", maxWidth: "480px", textAlign: "center", mt: 8 }}>
          <Typography
            variant="h4"
            sx={{
              color: "#ffffff",
              fontWeight: 500,
              mb: 3,
            }}
          >
            Sign in to your account
          </Typography>

          {message && (
            <Box
              sx={{
                display: "flex",
                alignItems: "center",
                justifyContent: "center",
                gap: 1,
                mb: 3,
                p: 2,
                borderRadius: 1,
                backgroundColor: message.includes("successfully")
                  ? "rgba(72, 187, 120, 0.1)"
                  : "rgba(229, 62, 62, 0.1)",
                border: message.includes("successfully")
                  ? "1px solid rgba(72, 187, 120, 0.2)"
                  : "1px solid rgba(229, 62, 62, 0.2)",
              }}
            >
              <CheckCircleIcon
                sx={{
                  color: message.includes("successfully") ? "#48bb78" : "#e53e3e",
                  fontSize: 18,
                }}
              />
              <Typography
                variant="body2"
                sx={{
                  color: message.includes("successfully") ? "#48bb78" : "#e53e3e",
                }}
              >
                {message}
              </Typography>
            </Box>
          )}

          <Box component="form" onSubmit={handleSubmit}>
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
                "& .MuiOutlinedInput-root": {
                  bgcolor: "rgba(255, 255, 255, 0.05)",
                  borderRadius: "6px",
                  "& fieldset": {
                    borderColor: "rgba(255, 255, 255, 0.2)",
                  },
                  "&:hover fieldset": {
                    borderColor: "rgba(255, 255, 255, 0.3)",
                  },
                  "&.Mui-focused fieldset": {
                    borderColor: "#4299e1",
                  },
                },
                "& .MuiInputBase-input": {
                  color: "#ffffff",
                  fontSize: "1rem",
                  padding: "16px"
                },
                "& .MuiInputBase-input::placeholder": {
                  color: "#b3b3b3",
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
                "& .MuiOutlinedInput-root": {
                  bgcolor: "rgba(255, 255, 255, 0.05)",
                  borderRadius: "6px",
                  "& fieldset": {
                    borderColor: "rgba(255, 255, 255, 0.2)",
                  },
                  "&:hover fieldset": {
                    borderColor: "rgba(255, 255, 255, 0.3)",
                  },
                  "&.Mui-focused fieldset": {
                    borderColor: "#4299e1",
                  },
                },
                "& .MuiInputBase-input": {
                  color: "#ffffff",
                  fontSize: "1rem",
                  padding: "16px"
                },
                "& .MuiInputBase-input::placeholder": {
                  color: "#b3b3b3",
                  opacity: 1,
                },
              }}
            />

            <Button
              type="submit"
              disabled={isSubmitDisabled || loading}
              fullWidth
              variant="contained"
              sx={{
                bgcolor: "#4299e1",
                color: "#000000",
                borderRadius: "6px",
                py: 1.75,
                mb: 3,
                textTransform: "none",
                fontWeight: 600,
                fontSize: "0.95rem",
                "&:hover": {
                  bgcolor: "#3182ce",
                  boxShadow: "0 4px 12px rgba(66, 153, 225, 0.3)",
                },
                "&:disabled": {
                  bgcolor: "#333333",
                  color: "#666666",
                },
                boxShadow: "0 2px 8px rgba(66, 153, 225, 0.2)",
              }}
              startIcon={loading ? <CircularProgress size={18} sx={{ color: "#666666" }} /> : <EmailIcon />}
            >
              {loading ? 'Signing in...' : 'Sign in with email'}
            </Button>
          </Box>

          <Box sx={{ mb: 3 }}>
            <Typography
              variant="body2"
              sx={{
                color: "#b3b3b3",
              }}
            >
              Forgot your password?{" "}
              <Button
                onClick={handleForgotPassword}
                sx={{
                  color: "#4299e1",
                  textTransform: "none",
                  p: 0,
                  "&:hover": { bgcolor: "rgba(66, 153, 225, 0.1)" },
                }}
              >
                Reset it here
              </Button>
            </Typography>
          </Box>

          <Button
            variant="outlined"
            onClick={() => navigate('/register')}
            fullWidth
            sx={{
              borderColor: "rgba(66, 153, 225, 0.3)",
              color: "#4299e1",
              borderRadius: "6px",
              py: 1.75,
              mb: 4,
              textTransform: "none",
              fontWeight: 500,
              fontSize: "0.95rem",
              backgroundColor: "rgba(66, 153, 225, 0.05)",
              "&:hover": {
                borderColor: "#4299e1",
                bgcolor: "rgba(66, 153, 225, 0.1)",
              },
            }}
          >
            Don't have an account? Sign up
          </Button>

          {/* Footer */}
          <Box sx={{ textAlign: "center" }}>
            <Typography
              variant="body2"
              sx={{
                color: "#666666",
                mb: 3,
                fontSize: "0.8rem",
                lineHeight: 1.4,
              }}
            >
              Need help?{" "}
              <Box
                component="span"
                sx={{
                  color: "#4299e1",
                  cursor: "pointer",
                  "&:hover": { textDecoration: "underline" },
                }}
              >
                Contact support
              </Box>
            </Typography>

            <Typography
              variant="body2"
              sx={{
                color: "#555555",
                fontSize: "0.75rem",
              }}
            >
              © 2025 ftrack. All rights reserved.
            </Typography>
          </Box>

          {/* Popup pour confirmer la réinitialisation */}
          <Dialog 
            open={openDialog} 
            onClose={handleCancelReset}
            PaperProps={{
              sx: {
                bgcolor: "#1a1a1a",
                border: "1px solid rgba(255, 255, 255, 0.1)",
              },
            }}
          >
            <DialogTitle sx={{ color: "#ffffff", bgcolor: "#1a1a1a" }}>
              Confirm Password Reset
            </DialogTitle>
            <DialogContent sx={{ bgcolor: "#1a1a1a" }}>
              <Typography sx={{ color: "#b3b3b3" }}>
                Are you sure you want to reset the password for this email: <strong style={{ color: "#4299e1" }}>{formData.email || 'N/A'}</strong>?
              </Typography>
            </DialogContent>
            <DialogActions sx={{ bgcolor: "#1a1a1a", p: 2 }}>
              <DialogButton 
                onClick={handleCancelReset} 
                sx={{ 
                  color: "#b3b3b3",
                  "&:hover": { bgcolor: "rgba(255, 255, 255, 0.1)" }
                }}
              >
                Cancel
              </DialogButton>
              <DialogButton 
                onClick={handleConfirmReset} 
                sx={{ 
                  color: "#4299e1",
                  "&:hover": { bgcolor: "rgba(66, 153, 225, 0.1)" }
                }} 
                disabled={loading}
              >
                {loading ? <CircularProgress size={20} sx={{ color: "#4299e1" }} /> : 'Confirm'}
              </DialogButton>
            </DialogActions>
          </Dialog>
        </Box>
      </Box>
    </ThemeProvider>
  );
};

export default LoginPage;