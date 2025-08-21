// src/theme/AppTheme.tsx
"use client";

import React from "react";
import { ThemeProvider, createTheme, CssBaseline } from "@mui/material";

export const darkTheme = createTheme({
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
    h6: {
      fontSize: "1.25rem",
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
    MuiCard: {
      styleOverrides: {
        root: {
          backgroundColor: "rgba(255, 255, 255, 0.05)",
          borderRadius: "6px",
          border: "1px solid rgba(255, 255, 255, 0.1)",
          boxShadow: "none",
        },
      },
    },
    MuiTextField: {
      styleOverrides: {
        root: {
          "& .MuiOutlinedInput-root": {
            backgroundColor: "rgba(255, 255, 255, 0.05)",
            borderRadius: "6px",
            border: "1px solid rgba(255, 255, 255, 0.1)",
            transition: "all 0.3s ease",
            "& fieldset": { border: "none" },
            "&:hover": {
              backgroundColor: "rgba(255, 255, 255, 0.08)",
              borderColor: "rgba(66, 153, 225, 0.3)",
            },
            "&.Mui-focused": {
              backgroundColor: "rgba(255, 255, 255, 0.08)",
              borderColor: "#4299e1",
              boxShadow: "0 0 0 2px rgba(66, 153, 225, 0.1)",
            },
          },
          "& .MuiInputBase-input": {
            color: "#ffffff",
            fontSize: "0.95rem",
            padding: "14px 16px",
          },
          "& .MuiInputBase-input::placeholder": {
            color: "#888888",
            opacity: 1,
          },
          "& .MuiFormHelperText-root": {
            color: "#f56565",
          },
        },
      },
    },
  },
});

/** AppTheme wrapper — use this at the top of pages/components to apply theme + baseline */
export default function AppTheme({ children }: { children: React.ReactNode }) {
  return (
    <ThemeProvider theme={darkTheme}>
      <CssBaseline />
      {children}
    </ThemeProvider>
  );
}
