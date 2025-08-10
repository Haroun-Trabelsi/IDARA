"use client";
import React from "react";
import { 
  Box, 
  TextField, 
  Typography, 
  ThemeProvider, 
  createTheme, 
  CssBaseline, 
  Button,
  FormControlLabel,
  Checkbox,
  Paper,
  InputAdornment,
  IconButton,
  CircularProgress,
  Alert
} from "@mui/material";
import { Visibility, VisibilityOff } from "@mui/icons-material";
import { useState } from "react";
import { useNavigate } from 'react-router-dom';

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

export default function UserProfileForm() {
  const navigate = useNavigate();
  const [formData, setFormData] = useState({
    name: "",
    surname: "",
    newPassword: "",
    confirmPassword: "",
    receiveUpdates: false,
  });
  const [showPassword, setShowPassword] = useState(false);
  const [showConfirmPassword, setShowConfirmPassword] = useState(false);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const handleInputChange = (field: string) => (event: React.ChangeEvent<HTMLInputElement>) => {
    setFormData(prev => ({
      ...prev,
      [field]: event.target.value,
    }));
  };

  const handleCheckboxChange = (event: React.ChangeEvent<HTMLInputElement>) => {
    setFormData(prev => ({
      ...prev,
      receiveUpdates: event.target.checked,
    }));
  };

  
  
  
  const handleSubmit = async (event: React.FormEvent) => {
    event.preventDefault();
    setLoading(true);
    setError(null);
  
    if (formData.newPassword !== formData.confirmPassword) {
      setError("Passwords do not match.");
      setLoading(false);
      return;
    }
  
    try {
      const response = await fetch('http://localhost:8080/col/update-profile', { // Ajusté à /api/update-profile
        method: 'POST',
        headers: { 
          'Content-Type': 'application/json',
          // Ajouter le token ici
          'Authorization': `Bearer ${localStorage.getItem('token') || ''}` // Inclure le token
        },
        credentials: 'include', // Pour inclure les cookies/auth
        body: JSON.stringify({
          name: formData.name,
          surname: formData.surname,
          newPassword: formData.newPassword,
          receiveUpdates: formData.receiveUpdates,
        }),
      });
  
      const data = await response.json();
      if (data.message) {
        navigate('/'); // Rediriger vers / après succès
      } else {
        setError(data.error || 'Failed to update profile.');
      }
    } catch (err) {
      setError('An error occurred. Please try again.');
    } finally {
      setLoading(false);
    }
  };

  return (
    <ThemeProvider theme={darkTheme}>
      <CssBaseline />
      <Box sx={{ 
        minHeight: "100vh", 
        display: "flex", 
        alignItems: "center", 
        justifyContent: "center",
        bgcolor: "#0f172a",
        p: 3
      }}>
        <Paper 
          elevation={3}
          sx={{ 
            p: 4, 
            maxWidth: 500, 
            width: "100%",
            bgcolor: "#1a202c",
            borderRadius: 2,
            border: "1px solid #2d3748"
          }}
        >
          <Typography 
            variant="h4" 
            component="h1" 
            sx={{ 
              mb: 1, 
              color: "#e2e8f0",
              fontWeight: 600,
              textAlign: "center"
            }}
          >
            Additional Information
          </Typography>
          
          <Typography 
            variant="body1" 
            sx={{ 
              mb: 3, 
              color: "#a0aec0",
              textAlign: "center",
              fontSize: "16px"
            }}
          >
            We need some additional information from you
          </Typography>

          <Box component="form" onSubmit={handleSubmit} sx={{ mt: 2 }}>
            {error && (
              <Alert severity="error" sx={{ mb: 2 }}>
                {error}
              </Alert>
            )}

            <TextField
              fullWidth
              label="Name"
              value={formData.name}
              onChange={handleInputChange("name")}
              margin="normal"
              required
              sx={{
                "& .MuiOutlinedInput-root": {
                  bgcolor: "#2d3748",
                  "& fieldset": { borderColor: "#4a5568" },
                  "&:hover fieldset": { borderColor: "#718096" },
                  "&.Mui-focused fieldset": { borderColor: "#4299e1" },
                },
                "& .MuiInputBase-input": { color: "#e2e8f0" },
                "& .MuiInputLabel-root": { color: "#a0aec0" },
                "& .MuiInputLabel-root.Mui-focused": { color: "#4299e1" },
              }}
            />

            <TextField
              fullWidth
              label="Surname"
              value={formData.surname}
              onChange={handleInputChange("surname")}
              margin="normal"
              required
              sx={{
                "& .MuiOutlinedInput-root": {
                  bgcolor: "#2d3748",
                  "& fieldset": { borderColor: "#4a5568" },
                  "&:hover fieldset": { borderColor: "#718096" },
                  "&.Mui-focused fieldset": { borderColor: "#4299e1" },
                },
                "& .MuiInputBase-input": { color: "#e2e8f0" },
                "& .MuiInputLabel-root": { color: "#a0aec0" },
                "& .MuiInputLabel-root.Mui-focused": { color: "#4299e1" },
              }}
            />

            <TextField
              fullWidth
              label="New Password"
              type={showPassword ? "text" : "password"}
              value={formData.newPassword}
              onChange={handleInputChange("newPassword")}
              margin="normal"
              required
              InputProps={{
                endAdornment: (
                  <InputAdornment position="end">
                    <IconButton
                      onClick={() => setShowPassword(!showPassword)}
                      edge="end"
                      sx={{ color: "#a0aec0" }}
                    >
                      {showPassword ? <VisibilityOff /> : <Visibility />}
                    </IconButton>
                  </InputAdornment>
                ),
              }}
              sx={{
                "& .MuiOutlinedInput-root": {
                  bgcolor: "#2d3748",
                  "& fieldset": { borderColor: "#4a5568" },
                  "&:hover fieldset": { borderColor: "#718096" },
                  "&.Mui-focused fieldset": { borderColor: "#4299e1" },
                },
                "& .MuiInputBase-input": { color: "#e2e8f0" },
                "& .MuiInputLabel-root": { color: "#a0aec0" },
                "& .MuiInputLabel-root.Mui-focused": { color: "#4299e1" },
              }}
            />

            <TextField
              fullWidth
              label="Confirm New Password"
              type={showConfirmPassword ? "text" : "password"}
              value={formData.confirmPassword}
              onChange={handleInputChange("confirmPassword")}
              margin="normal"
              required
              InputProps={{
                endAdornment: (
                  <InputAdornment position="end">
                    <IconButton
                      onClick={() => setShowConfirmPassword(!showConfirmPassword)}
                      edge="end"
                      sx={{ color: "#a0aec0" }}
                    >
                      {showConfirmPassword ? <VisibilityOff /> : <Visibility />}
                    </IconButton>
                  </InputAdornment>
                ),
              }}
              sx={{
                "& .MuiOutlinedInput-root": {
                  bgcolor: "#2d3748",
                  "& fieldset": { borderColor: "#4a5568" },
                  "&:hover fieldset": { borderColor: "#718096" },
                  "&.Mui-focused fieldset": { borderColor: "#4299e1" },
                },
                "& .MuiInputBase-input": { color: "#e2e8f0" },
                "& .MuiInputLabel-root": { color: "#a0aec0" },
                "& .MuiInputLabel-root.Mui-focused": { color: "#4299e1" },
              }}
            />

            <FormControlLabel
              control={
                <Checkbox
                  checked={formData.receiveUpdates}
                  onChange={handleCheckboxChange}
                  sx={{
                    color: "#4a5568",
                    "&.Mui-checked": { color: "#4299e1" },
                  }}
                />
              }
              label="I would like to receive product updates and company news"
              sx={{ mt: 2, "& .MuiFormControlLabel-label": { color: "#a0aec0", fontSize: "14px" } }}
            />

            <Button
              type="submit"
              fullWidth
              variant="contained"
              disabled={loading}
              sx={{
                mt: 3,
                py: 1.5,
                bgcolor: "#4299e1",
                "&:hover": { bgcolor: "#3182ce" },
                textTransform: "none",
                fontSize: "16px",
                fontWeight: 600,
              }}
              startIcon={loading ? <CircularProgress size={20} color="inherit" /> : null}
            >
              {loading ? 'Updating...' : 'Update Profile'}
            </Button>
          </Box>
        </Paper>
      </Box>
    </ThemeProvider>
  );
}