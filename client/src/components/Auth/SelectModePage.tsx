// src/components/SelectModePage.tsx
"use client";

import React, { useState } from "react";
import {
  Box,
  Typography,
  IconButton,
  Menu,
  MenuItem,
  Card,
  CardContent,
  CardActions,
  Button,
} from "@mui/material";
import {
  Language as LanguageIcon,
  Computer as ComputerIcon,
  Cloud as CloudIcon,
} from "@mui/icons-material";
import { useNavigate } from "react-router-dom";
import AppTheme from "./AppTheme"; // adjust path if needed

export default function SelectModePage() {
  const navigate = useNavigate();
  const [languageAnchor, setLanguageAnchor] = useState<null | HTMLElement>(null);
  const [selectedLanguage, setSelectedLanguage] = useState("EN");
  const [hovered, setHovered] = useState<"standalone" | "connected" | null>(null);

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

  const handleSelect = (choice: string) => {
    if (choice === "Connected to Ftrack") {
      navigate("/ftrack-setup");
      return;
    }
    if (choice === "Stand Alone") {
      navigate("/Projects");
      return;
    }
  };

  return (
    <AppTheme>
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
        <Box sx={{ width: "100%", maxWidth: "900px", textAlign: "center", mt: 8 }}>
          <Typography variant="h4" sx={{ color: "#ffffff", fontWeight: 500, mb: 3 }}>
            Choose Your Mode
          </Typography>

          <Box sx={{ display: "flex", gap: 4, justifyContent: "center", flexWrap: "wrap" }}>
            <Card
              sx={{
                width: 300,
                textAlign: "center",
                transition: "filter 0.3s ease, transform 0.15s ease",
                transform: hovered === "standalone" ? "translateY(-6px)" : "none",
                filter: hovered === "connected" ? "blur(2px)" : "none",
                backdropFilter: "blur(12px)",
              }}
              onMouseEnter={() => setHovered("standalone")}
              onMouseLeave={() => setHovered(null)}
            >
              <CardContent>
                <ComputerIcon sx={{ fontSize: 60, color: "#4299e1", mb: 2 }} />
                <Typography variant="h6">Stand Alone</Typography>
                <Typography variant="body2" sx={{ mt: 1, color: "#b3b3b3" }}>
                  Operate independently without external connections.
                </Typography>
              </CardContent>
              <CardActions sx={{ justifyContent: "center" }}>
                <Button
                  variant="contained"
                  color="primary"
                  onClick={() => handleSelect("Stand Alone")}
                  sx={{ bgcolor: "#4299e1", color: "#000000", "&:hover": { bgcolor: "#3182ce" } }}
                >
                  Select
                </Button>
              </CardActions>
            </Card>

            <Card
              sx={{
                width: 300,
                textAlign: "center",
                transition: "filter 0.3s ease, transform 0.15s ease",
                transform: hovered === "connected" ? "translateY(-6px)" : "none",
                filter: hovered === "standalone" ? "blur(2px)" : "none",
                backdropFilter: "blur(12px)",
              }}
              onMouseEnter={() => setHovered("connected")}
              onMouseLeave={() => setHovered(null)}
            >
              <CardContent>
                <CloudIcon sx={{ fontSize: 60, color: "#4299e1", mb: 2 }} />
                <Typography variant="h6">Connected to Ftrack</Typography>
                <Typography variant="body2" sx={{ mt: 1, color: "#b3b3b3" }}>
                  Integrate with Ftrack for enhanced collaboration.
                </Typography>
              </CardContent>
              <CardActions sx={{ justifyContent: "center" }}>
                <Button
                  variant="contained"
                  color="primary"
                  onClick={() => handleSelect("Connected to Ftrack")}
                  sx={{ bgcolor: "#4299e1", color: "#000000", "&:hover": { bgcolor: "#3182ce" } }}
                >
                  Select
                </Button>
              </CardActions>
            </Card>
          </Box>

          <Box sx={{ textAlign: "center", mt: 4 }}>
            <Typography variant="body2" sx={{ color: "#555555", fontSize: "0.75rem" }}>
              © 2025 ftrack. All rights reserved.
            </Typography>
          </Box>
        </Box>
      </Box>
    </AppTheme>
  );
}
