"use client";

import React, { useState, useEffect } from "react";
import { 
  Box, 
  Typography, 
  Button, 
  TextField,
  Dialog,
  DialogTitle,
  DialogContent,
  Alert,
  CircularProgress 
} from "@mui/material";
import { useNavigate } from 'react-router-dom';
import axios from 'utils/axios';
import { useAuth } from '../../contexts/AuthContext'; // Ajuste le chemin selon ton projet

export default function VerifyEmailPage() {
  const navigate = useNavigate();
  const { account } = useAuth(); // Récupérer l'account depuis le contexte
  const [open, setOpen] = useState(true);
  const [message, setMessage] = useState<string | null>(null);
  const [loading, setLoading] = useState(false);
  const [verificationCode, setVerificationCode] = useState('');
  const email = account?.email || null; // Utiliser l'email depuis l'account
  const [timer, setTimer] = useState(30);
  const [canResend, setCanResend] = useState(false);

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
    if (!verificationCode || verificationCode.length !== 6 || !email) {
      setMessage('Please enter a valid 6-digit verification code and ensure email is set.');
      return;
    }

    setLoading(true);
    try {
      console.log('Verifying code:', { code: verificationCode, email });
      const response = await axios.post('/auth/verify-email', {
        code: verificationCode,
        email
      }, {
        headers: { 'Content-Type': 'application/json' }
      });
      setMessage(response.data.message || 'Email verified successfully! Redirecting to login...');
      setTimeout(() => navigate('/'), 3000);
    } catch (err: any) {
      console.error('Verification error:', err.response?.data, err.response?.status);
      setMessage(err.response?.data?.message || 'Invalid verification code.');
    } finally {
      setLoading(false);
    }
  };

  const handleResendVerification = async () => {
    if (!email || loading) return;

    setLoading(true);
    setCanResend(false);
    setTimer(30);
    try {
      console.log('Resending code for email:', email);
      const response = await axios.post('/auth/resend-verification', { email }, {
        headers: { 'Content-Type': 'application/json' }
      });
      setMessage(response.data.message || 'Verification code resent successfully! Please check your inbox.');
      setTimeout(() => setMessage(null), 3000);
    } catch (err: any) {
      console.error('Resend error:', err.response?.data, err.response?.status);
      setMessage(err.response?.data?.message || 'Failed to resend verification code.');
    } finally {
      setLoading(false);
    }
  };

  const handleClose = () => {
    setOpen(false);
    navigate('/');
  };

  return (
    <Box
      sx={{
        display: 'flex',
        flexDirection: 'column',
        alignItems: 'center',
        justifyContent: 'center',
        height: '100vh',
        bgcolor: '#f7fafc',
        p: 3
      }}
    >
      <Dialog
        open={open}
        onClose={handleClose}
        maxWidth="sm"
        fullWidth
        PaperProps={{
          sx: {
            bgcolor: 'white',
            borderRadius: 2,
            boxShadow: '0 4px 6px rgba(0, 0, 0, 0.1)'
          }
        }}
      >
        <DialogTitle sx={{ textAlign: 'center', bgcolor: '#f7fafc', color: '#4a5568', fontWeight: 600 }}>
          Check Your Inbox
        </DialogTitle>
        <DialogContent sx={{ bgcolor: '#f7fafc', p: 3 }}>
          <Typography variant="body2" sx={{ textAlign: 'center', color: '#718096', mb: 3 }}>
            Enter the 6-digit code we sent to {email || 'your email'} to finish your login.
          </Typography>

          {message && (
            <Alert severity={message.includes('successfully') ? 'success' : 'error'} sx={{ mb: 3 }}>
              {message}
            </Alert>
          )}

          <TextField
            fullWidth
            label="6-digit code"
            variant="outlined"
            value={verificationCode}
            onChange={(e) => {
              const value = e.target.value.replace(/\D/g, '');
              if (value.length <= 6) setVerificationCode(value);
            }}
            inputProps={{ maxLength: 6 }}
            sx={{ 
              mb: 2,
              '& .MuiOutlinedInput-root': { bgcolor: 'white' }
            }}
          />

          <Typography variant="body2" sx={{ textAlign: 'center', color: '#718096', mb: 2 }}>
            Didn't receive the code?{' '}
            {canResend ? (
              <Button
                onClick={handleResendVerification}
                disabled={loading}
                sx={{ 
                  color: '#4299e1',
                  textTransform: 'none',
                  p: 0,
                  '&:hover': { bgcolor: 'rgba(66, 153, 225, 0.1)' }
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
            sx={{ 
              bgcolor: '#4299e1',
              '&:hover': { bgcolor: '#3182ce' },
              textTransform: 'none',
              width: '100%',
              py: 1.5,
              mb: 2
            }}
          >
            {loading ? <CircularProgress size={24} sx={{ color: 'white' }} /> : 'Verify'}
          </Button>

          <Button
            variant="outlined"
            onClick={handleClose}
            sx={{ 
              borderColor: '#e2e8f0',
              color: '#4a5568',
              '&:hover': { borderColor: '#cbd5e0', bgcolor: 'rgba(0, 0, 0, 0.04)' },
              textTransform: 'none',
              width: '100%',
              py: 1.5
            }}
          >
            Cancel
          </Button>
        </DialogContent>
      </Dialog>
    </Box>
  );
}