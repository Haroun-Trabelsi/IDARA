// Modified VerifyEmailPage.tsx
"use client";

import React, { useState, useEffect } from "react";
import {
  Box,
  Typography,
  Button,
  TextField,
  CircularProgress,
  ThemeProvider,
  createTheme,
  CssBaseline,
  IconButton,
  Menu,
  MenuItem,
} from "@mui/material";
import { 
  CheckCircle as CheckCircleIcon,
  Language as LanguageIcon,
} from "@mui/icons-material";
import { useNavigate, useLocation } from "react-router-dom";
import axios from "utils/axios";
import { useAuth } from "../../contexts/AuthContext";

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
    MuiTextField: {
      styleOverrides: {
        root: {
          '& .MuiOutlinedInput-root': {
            backgroundColor: 'rgba(255, 255, 255, 0.05)',
            borderRadius: '6px',
            border: '1px solid rgba(255, 255, 255, 0.1)',
            transition: 'all 0.3s ease',
            '& fieldset': {
              border: 'none',
            },
            '&:hover': {
              backgroundColor: 'rgba(255, 255, 255, 0.08)',
              borderColor: 'rgba(66, 153, 225, 0.3)',
            },
            '&.Mui-focused': {
              backgroundColor: 'rgba(255, 255, 255, 0.08)',
              borderColor: '#4299e1',
              boxShadow: '0 0 0 2px rgba(66, 153, 225, 0.1)',
            },
          },
          '& .MuiInputBase-input': {
            color: '#ffffff',
            fontSize: '0.95rem',
            padding: '14px 16px',
          },
          '& .MuiInputLabel-root': {
            color: '#888888',
            fontSize: '0.95rem',
            '&.Mui-focused': {
              color: '#4299e1',
            },
          },
          '& .MuiFormHelperText-root': { color: '#f56565' }
        },
      },
    },
  },
});

export default function VerifyEmailPage() {
  const navigate = useNavigate();
  const { loginWithToken } = useAuth();
  const location = useLocation();
  const [message, setMessage] = useState<string | null>(null);
  const [loading, setLoading] = useState(false);
  const [verificationCode, setVerificationCode] = useState("");
  const [email, setEmail] = useState<string>(location.state?.email || localStorage.getItem('pendingVerificationEmail') || "");
  const [formData, setFormData] = useState(() => {
    const savedData = localStorage.getItem('pendingRegistrationData');
    const stateData = location.state?.formData;
    const initialData = stateData || (savedData ? JSON.parse(savedData) : {});
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
  const [accountId] = useState<string | null>(location.state?.accountId || localStorage.getItem('pendingAccountId') || null);
  const [timer, setTimer] = useState(30);
  const [canResend, setCanResend] = useState(false);
  const [languageAnchor, setLanguageAnchor] = useState<null | HTMLElement>(null);
  const [selectedLanguage, setSelectedLanguage] = useState("EN");

  useEffect(() => {
    console.log('Email initial dans VerifyEmailPage:', email);
    console.log('FormData initial dans VerifyEmailPage:', formData);
    console.log('Account ID:', accountId);
    console.log('Données de location.state:', location.state);
    console.log('Données de localStorage (pendingRegistrationData):', localStorage.getItem('pendingRegistrationData'));
    console.log('pendingVerificationEmail:', localStorage.getItem('pendingVerificationEmail'));
    if (location.state?.email) {
      console.log('Email mis à jour depuis location.state:', location.state.email);
      setEmail(location.state.email);
      localStorage.setItem('pendingVerificationEmail', location.state.email);
    }
    if (location.state?.formData) {
      console.log('FormData mis à jour depuis location.state:', location.state.formData);
      setFormData(location.state.formData);
      localStorage.setItem('pendingRegistrationData', JSON.stringify(location.state.formData));
    }
    if (!email) {
      console.log('Aucun email trouvé, redirection vers /register');
      setMessage("No email provided. Please register again.");
      setTimeout(() => navigate('/register', { state: { formData, accountId } }), 3000);
    }
  }, [email, location.state, navigate, formData, accountId]);

  useEffect(() => {
    let interval: NodeJS.Timeout | null = null;
    if (timer > 0 && !canResend) {
      interval = setInterval(() => {
        setTimer((prev) => prev - 1);
      }, 1000);
    } else if (timer === 0) {
      setCanResend(true);
      if (interval) clearInterval(interval);
    }
    return () => {
      if (interval) clearInterval(interval);
    };
  }, [timer, canResend]);

  const handleVerifyEmail = async () => {
    if (!verificationCode || verificationCode.length !== 6) {
      setMessage("Please enter a valid 6-digit verification code.");
      return;
    }
    if (!email) {
      setMessage("No email provided. Please register again.");
      setTimeout(() => navigate('/register', { state: { formData, accountId } }), 3000);
      return;
    }

    setLoading(true);
    try {
      console.log("Verifying code:", { code: verificationCode, email });
      const response = await axios.post(
        "/auth/verify-email",
        {
          code: verificationCode,
          email,
        },
        {
          headers: { "Content-Type": "application/json" },
        }
      );
      setMessage(response.data.message || "Email verified successfully!");
      localStorage.removeItem('pendingVerificationEmail');
      localStorage.removeItem('pendingRegistrationData');
      localStorage.removeItem('pendingAccountId');
      loginWithToken(response.data.token, response.data.data);
      setTimeout(() => navigate("/select-mode"), 3000);
    } catch (err: any) {
      console.error("Verification error:", err.response?.data, err.response?.status);
      setMessage(err.response?.data?.message || "Invalid verification code.");
    } finally {
      setLoading(false);
    }
  };

  const handleResendVerification = async () => {
    if (!email || loading) {
      setMessage("No email provided. Please register again.");
      setTimeout(() => navigate('/register', { state: { formData, accountId } }), 3000);
      return;
    }

    setLoading(true);
    setCanResend(false);
    setTimer(30);
    try {
      console.log("Resending code for email:", email);
      const response = await axios.post(
        "/auth/resend-verification",
        { email },
        {
          headers: { "Content-Type": "application/json" },
        }
      );
      setMessage(
        response.data.message || "Verification code resent successfully! Please check your inbox."
      );
      setTimeout(() => setMessage(null), 3000);
    } catch (err: any) {
      console.error("Resend error:", err.response?.data, err.response?.status);
      setMessage(err.response?.data?.message || "Failed to resend verification code.");
    } finally {
      setLoading(false);
    }
  };

  const handleBackToRegister = () => {
    console.log('Retour à /register avec formData et accountId:', { formData, accountId });
    localStorage.removeItem('pendingVerificationEmail');
    navigate("/register", { state: { formData, accountId } });
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
            Check Your Inbox
          </Typography>

          <Typography
            variant="body1"
            sx={{
              color: "#b3b3b3",
              mb: 3,
              fontSize: "1rem",
            }}
          >
            Enter the 6-digit code we sent to{" "}
            <strong style={{ color: "#4299e1" }}>{email || "your email"}</strong>{" "}
            to verify your account.
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

          <TextField
            fullWidth
            label="6-digit code"
            variant="outlined"
            value={verificationCode}
            onChange={(e) => {
              const value = e.target.value.replace(/\D/g, "");
              if (value.length <= 6) setVerificationCode(value);
            }}
            inputProps={{ maxLength: 6 }}
            sx={{ mb: 3 }}
          />

          <Typography
            variant="body2"
            sx={{
              color: "#b3b3b3",
              mb: 3,
            }}
          >
            Didn't receive the code?{" "}
            {canResend ? (
              <Button
                onClick={handleResendVerification}
                disabled={loading}
                sx={{
                  color: "#4299e1",
                  textTransform: "none",
                  p: 0,
                  "&:hover": { bgcolor: "rgba(66, 153, 225, 0.1)" },
                }}
              >
                Resend code
              </Button>
            ) : (
              `Resend code in ${timer} seconds`
            )}
          </Typography>

          <Button
            variant="contained"
            onClick={handleVerifyEmail}
            disabled={loading || verificationCode.length !== 6 || !email}
            fullWidth
            sx={{
              bgcolor: "#4299e1",
              color: "#000000",
              borderRadius: "6px",
              py: 1.75,
              mb: 2,
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
          >
            {loading ? <CircularProgress size={18} sx={{ color: "#666666" }} /> : "Verify"}
          </Button>

          <Button
            variant="outlined"
            onClick={handleBackToRegister}
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
              "&:disabled": {
                borderColor: "#333333",
                color: "#666666",
                bgcolor: "rgba(255, 255, 255, 0.02)",
              },
            }}
          >
            Back to Register
          </Button>

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
        </Box>
      </Box>
    </ThemeProvider>
  );
}