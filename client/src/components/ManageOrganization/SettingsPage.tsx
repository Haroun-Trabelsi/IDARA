"use client";

import React, { useState } from "react";
import {
  Box,
  Typography,
  Card,
  CardContent,
  Avatar,
  Button,
  Alert,
  CircularProgress,
  IconButton,
  Tooltip,
  Chip,
} from "@mui/material";
import { Business, Settings, HelpOutline, Edit } from "@mui/icons-material";
import { useAuth } from "contexts/AuthContext";
import axios from "utils/axios";

interface SettingsPageProps {
  orgData: { organizationName: string; teamSize: string; region: string; id: string };
  message?: { type: "success" | "error"; text: string } | null;
  error: string | null;
  loading: boolean;
  renderSettingsField: (label: string, value: string, field: "organizationName" | "teamSize" | "region", editable?: boolean) => React.ReactNode;
}

export default function SettingsPage({ orgData, message, error, loading, renderSettingsField }: SettingsPageProps) {
  const { token } = useAuth();
  const [localMessage, setLocalMessage] = useState<{ type: "success" | "error"; text: string } | null>(null);

  const handleDeleteOrganization = async () => {
    if (!token) return;
    try {
      await axios.delete("/col/organization", { headers: { Authorization: `Bearer ${token}` } });
      setLocalMessage({ type: "success", text: "Organization deleted successfully" });
      // Rediriger ou gérer la déconnexion après suppression
      window.location.href = "/login";
    } catch (err: any) {
      setLocalMessage({ type: "error", text: err.response?.data?.message || "Failed to delete organization" });
    }
    setTimeout(() => setLocalMessage(null), 3000);
  };

  const renderProjectVisibilityRow = () => (
    <Box
      sx={{
        display: "flex",
        justifyContent: "space-between",
        alignItems: "center",
        py: 2.5,
        px: 3,
        borderBottom: "1px solid #2d3748",
      }}
    >
      <Box sx={{ display: "flex", alignItems: "center", gap: 1 }}>
        <Typography
          variant="body2"
          sx={{
            color: "#a0aec0",
            fontWeight: 500,
            fontSize: "14px",
            mb: 0.5,
          }}
        >
          Default Project Visibility
        </Typography>
        <Tooltip title="Choose who can view newly created projects in your organization" arrow>
          <IconButton size="small" sx={{ color: "#a0aec0", p: 0.25 }}>
            <HelpOutline sx={{ fontSize: "14px" }} />
          </IconButton>
        </Tooltip>
        <Chip
          label="pro"
          size="small"
          sx={{
            bgcolor: "#4299e1",
            color: "white",
            fontSize: "10px",
            height: "20px",
            "& .MuiChip-label": {
              px: 1,
            },
          }}
        />
      </Box>
      <Typography
        variant="body2"
        sx={{
          color: "#e2e8f0",
          fontSize: "14px",
          flexGrow: 1,
          textAlign: "center",
        }}
      >
        All organization members
      </Typography>
      <IconButton size="small" sx={{ color: "#4299e1", opacity: 0.5 }} disabled>
        <Edit sx={{ fontSize: "16px" }} />
      </IconButton>
    </Box>
  );

  return (
    <Box>
      <Box sx={{ display: "flex", justifyContent: "space-between", alignItems: "center", mb: 3 }}>
        <Typography variant="h4" sx={{ color: "#e2e8f0", fontWeight: 600 }}>
          Organization Settings
        </Typography>
        <Avatar sx={{ width: 40, height: 40, bgcolor: "transparent", border: "1px solid #2d3748" }}>
          <Settings sx={{ fontSize: "20px", color: "#4299e1" }} />
        </Avatar>
      </Box>

      <Box sx={{ display: "flex", alignItems: "center", mb: 3 }}>
        <Avatar sx={{ width: 32, height: 32, bgcolor: "transparent", border: "1px solid #2d3748", mr: 2 }}>
          <Business sx={{ fontSize: "16px", color: "#4299e1" }} />
        </Avatar>
        <Typography variant="body1" sx={{ color: "#e2e8f0" }}>{orgData.organizationName}</Typography>
      </Box>

      {localMessage && <Alert severity={localMessage.type} sx={{ mb: 3, borderRadius: "8px" }}>{localMessage.text}</Alert>}
      {message && <Alert severity={message.type} sx={{ mb: 3, borderRadius: "8px" }}>{message.text}</Alert>}
      {error && <Alert severity="error" sx={{ mb: 3, borderRadius: "8px" }}>{error}</Alert>}

      {loading ? (
        <Box sx={{ display: "flex", justifyContent: "center", py: 4 }}>
          <CircularProgress sx={{ color: "#4299e1" }} />
        </Box>
      ) : (
        <>
          <Card sx={{ bgcolor: "#1a202c", border: "1px solid #2d3748", mb: 3, borderRadius: "12px" }}>
            <CardContent sx={{ p: 0 }}>
              <Box sx={{ p: 3, borderBottom: "1px solid #2d3748" }}>
                <Typography variant="h6" sx={{ color: "#e2e8f0", fontWeight: 500, fontSize: "16px" }}>
                  Organization Information
                </Typography>
              </Box>
              <Box>
                {renderSettingsField("Organization Name", orgData.organizationName, "organizationName")}
                {renderSettingsField("Team Size", orgData.teamSize, "teamSize")}
                {renderSettingsField("Region", orgData.region, "region")}
              </Box>
            </CardContent>
          </Card>

          <Card sx={{ bgcolor: "#1a202c", border: "1px solid #2d3748", mb: 3, borderRadius: "12px" }}>
            <CardContent sx={{ p: 0 }}>
              <Box sx={{ p: 3, borderBottom: "1px solid #2d3748", display: "flex", alignItems: "center", gap: 1 }}>
                <Typography variant="h6" sx={{ color: "#e2e8f0", fontWeight: 500, fontSize: "16px" }}>
                  Project Settings
                </Typography>
                <Tooltip title="This option is not available now" arrow>
                  <IconButton size="small" sx={{ color: "#a0aec0" }}>
                    <HelpOutline sx={{ fontSize: "16px" }} />
                  </IconButton>
                </Tooltip>
              </Box>
              <Box>{renderProjectVisibilityRow()}</Box>
            </CardContent>
          </Card>

          <Box sx={{ p: 3, textAlign: "center" }}>
            <Typography variant="body2" sx={{ color: "#a0aec0", mb: 1, fontSize: "14px" }}>
              Once you delete an organization, there is no going back.
            </Typography>
            <Button
              onClick={handleDeleteOrganization}
              variant="outlined"
              sx={{
                color: "#ef4444",
                borderColor: "#ef4444",
                borderRadius: "8px",
                fontSize: "14px",
                px: 3,
                py: 1,
                "&:hover": { borderColor: "#c53030", bgcolor: "rgba(239, 68, 68, 0.1)" },
              }}
            >
              Delete Organization
            </Button>
          </Box>
        </>
      )}
    </Box>
  );
}