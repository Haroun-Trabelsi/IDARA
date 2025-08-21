// src/components/Auth/FtrackSetup.tsx
"use client";

import React, { useState } from "react";
import {
  Box,
  TextField,
  Button,
  Typography,
  CircularProgress,
  CssBaseline,
  IconButton,
  Menu,
  MenuItem,
  InputAdornment,
  FormControlLabel,
  Checkbox,
} from "@mui/material";
import {
  Language as LanguageIcon,
  Visibility,
  VisibilityOff,
} from "@mui/icons-material";
import { useNavigate } from "react-router-dom";
import axios from "axios";

// Import the shared AppTheme wrapper (adjust path if needed)
import AppTheme from "./AppTheme";

export default function FtrackSetup(): React.ReactElement {
  const navigate = useNavigate();

  const [languageAnchor, setLanguageAnchor] = useState<null | HTMLElement>(
    null
  );
  const [selectedLanguage, setSelectedLanguage] = useState("EN");

  const [serverUrl, setServerUrl] = useState("");
  const [username, setUsername] = useState("");
  const [apiKey, setApiKey] = useState("");
  const [showKey, setShowKey] = useState(false);
  const [remember, setRemember] = useState(true);

  const [error, setError] = useState("");
  const [loading, setLoading] = useState(false);
  const [success, setSuccess] = useState("");

  const handleLanguageClick = (event: React.MouseEvent<HTMLElement>) => {
    setLanguageAnchor(event.currentTarget);
  };
  const handleLanguageClose = () => setLanguageAnchor(null);
  const handleLanguageSelect = (lang: string) => {
    setSelectedLanguage(lang);
    handleLanguageClose();
  };

  const isUrlValid = (url: string) => {
    try {
      if (!/^https?:\/\//i.test(url)) return false;
      new URL(url);
      return true;
    } catch {
      return false;
    }
  };

  const isSubmitDisabled =
    !serverUrl || !username || !apiKey || loading || !isUrlValid(serverUrl);

  const handleSubmit = async (e?: React.FormEvent) => {
    e?.preventDefault?.();
    setError("");
    setSuccess("");

    if (!serverUrl || !username || !apiKey) {
      setError("All fields are required");
      return;
    }
    if (!isUrlValid(serverUrl)) {
      setError("Server URL looks invalid. Include https:// or http://");
      return;
    }

    setLoading(true);
    try {
      const token = localStorage.getItem("token");
      await axios.post(
        "/auth/ftrack",
        {
          serverUrl: serverUrl.trim(),
          username: username.trim(),
          apiKey: apiKey.trim(),
        },
        { headers: { ...(token ? { Authorization: `Bearer ${token}` } : {}) } }
      );

      setSuccess("Ftrack settings saved successfully");
      if (remember) {
        localStorage.setItem(
          "ftrackSettings",
          JSON.stringify({ serverUrl, username })
        );
      } else {
        localStorage.removeItem("ftrackSettings");
      }

      setTimeout(() => navigate("/dashboard"), 700);
    } catch (err: any) {
      setError(
        err?.response?.data?.message ||
          err?.message ||
          "Failed to save settings — check console/network"
      );
      console.error("Ftrack submit error:", err);
    } finally {
      setLoading(false);
    }
  };

  // Load saved settings on mount
  React.useEffect(() => {
    try {
      const saved = localStorage.getItem("ftrackSettings");
      if (saved) {
        const s = JSON.parse(saved);
        setServerUrl(s.serverUrl || "");
        setUsername(s.username || "");
      }
    } catch {}
  }, []);

  return (
    <AppTheme>
      {/* CssBaseline is already applied by AppTheme, but having it here is harmless */}
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
        {/* Header: logo + language */}
        <Box
          sx={{
            position: "absolute",
            top: 20,
            left: 20,
            right: 20,
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
                height: 32,
                width: "auto",
                filter: "brightness(0) invert(1)",
              }}
            />
            <Typography sx={{ color: "#b3b3b3", fontSize: "0.95rem" }}>
              Ftrack — Connect
            </Typography>
          </Box>

          <Box sx={{ display: "flex", alignItems: "center", gap: 1 }}>
            <Typography variant="body2" sx={{ color: "#b3b3b3", fontSize: 14 }}>
              {selectedLanguage}
            </Typography>
            <IconButton
              onClick={handleLanguageClick}
              sx={{
                color: "#b3b3b3",
                "&:hover": {
                  color: "#4299e1",
                  backgroundColor: "rgba(66, 153, 225, 0.08)",
                },
              }}
            >
              <LanguageIcon />
            </IconButton>
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
                      backgroundColor: "rgba(66, 153, 225, 0.08)",
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
        </Box>

        {/* Centered area */}
        <Box sx={{ width: "100%", maxWidth: 480, mt: 6 }}>
          <Typography
            variant="h4"
            sx={{
              textAlign: "center",
              color: "#ffffff",
              mb: 1,
              fontWeight: 500,
            }}
          >
            Connect to Ftrack
          </Typography>

          <Typography
            variant="body2"
            sx={{ textAlign: "center", color: "#888888", mb: 3 }}
          >
            Enter your Ftrack server details and API key to connect.
          </Typography>

          <Box
            component="form"
            onSubmit={handleSubmit}
            sx={{
              background:
                "linear-gradient(180deg, rgba(255,255,255,0.02), rgba(255,255,255,0.01))",
              p: 3,
              borderRadius: 2,
              border: "1px solid rgba(255,255,255,0.04)",
            }}
          >
            <TextField
              placeholder="https://ftrack.example.com"
              label="Server URL"
              value={serverUrl}
              onChange={(e) => setServerUrl(e.target.value)}
              fullWidth
              variant="outlined"
              required
              helperText={
                serverUrl && !isUrlValid(serverUrl)
                  ? "Include protocol (https:// or http://)"
                  : " "
              }
              error={!!serverUrl && !isUrlValid(serverUrl)}
              sx={{ mb: 2 }}
            />

            <TextField
              placeholder="username"
              label="Username"
              value={username}
              onChange={(e) => setUsername(e.target.value)}
              fullWidth
              variant="outlined"
              required
              sx={{ mb: 2 }}
            />

            <TextField
              placeholder="API key"
              label="API Key"
              value={apiKey}
              onChange={(e) => setApiKey(e.target.value)}
              fullWidth
              variant="outlined"
              required
              type={showKey ? "text" : "password"}
              sx={{ mb: 1 }}
              InputProps={{
                endAdornment: (
                  <InputAdornment position="end">
                    <IconButton
                      onClick={() => setShowKey((s) => !s)}
                      edge="end"
                      aria-label="toggle api key visibility"
                      sx={{ color: "#888888" }}
                    >
                      {showKey ? <VisibilityOff /> : <Visibility />}
                    </IconButton>
                  </InputAdornment>
                ),
              }}
            />

            {error && (
              <Typography sx={{ color: "#ff6b6b", mt: 1, mb: 1 }}>{error}</Typography>
            )}
            {success && (
              <Typography sx={{ color: "#51d88a", mt: 1, mb: 1 }}>{success}</Typography>
            )}

            <FormControlLabel
              control={
                <Checkbox
                  checked={remember}
                  onChange={(e) => setRemember(e.target.checked)}
                  sx={{
                    color: "#888888",
                    "&.Mui-checked": { color: "#4299e1" },
                    "& .MuiSvgIcon-root": { fontSize: "1.2rem" },
                  }}
                />
              }
              label={
                <Typography variant="body2" sx={{ color: "#b3b3b3", fontSize: "0.9rem" }}>
                  Remember server & username (local)
                </Typography>
              }
              sx={{ mt: 1 }}
            />

            <Button
              type="submit"
              disabled={isSubmitDisabled}
              fullWidth
              variant="contained"
              sx={{
                mt: 3,
                backgroundColor: "#4299e1",
                color: "#000000",
                py: 1.25,
                fontSize: "0.95rem",
                fontWeight: 600,
                "&:hover": { backgroundColor: "#3182ce", boxShadow: "0 4px 12px rgba(66,153,225,0.2)" },
                "&:disabled": { backgroundColor: "#333333", color: "#666666" },
              }}
              startIcon={loading ? <CircularProgress size={18} sx={{ color: "#666666" }} /> : null}
            >
              {loading ? "Connecting…" : "Save & Continue"}
            </Button>

            <Typography
              variant="body2"
              sx={{
                textAlign: "center",
                color: "#666666",
                mt: 2,
                fontSize: "0.85rem",
                lineHeight: 1.4,
              }}
            >
              By connecting, you agree to your server's usage policies.
            </Typography>

            <Box sx={{ textAlign: "center", mt: 3 }}>
              <Typography variant="body2" sx={{ color: "#888888", fontSize: "0.9rem", mb: 1.5 }}>
                Need help?{" "}
                <Box
                  component="span"
                  sx={{
                    color: "#4299e1",
                    cursor: "pointer",
                    fontWeight: 500,
                    "&:hover": { textDecoration: "underline" },
                  }}
                  onClick={() => window.open("https://ftrack.com/support", "_blank")}
                >
                  Support
                </Box>
              </Typography>
              <Typography variant="caption" sx={{ color: "#555555", fontSize: "0.75rem" }}>
                © 2025 ftrack. All rights reserved.
              </Typography>
            </Box>
          </Box>
        </Box>
      </Box>
    </AppTheme>
  );
}
