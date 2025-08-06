// src/components/ManageOrganization/SettingsPage.tsx
"use client";

import React, { useState, useCallback, useEffect } from "react";
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
  Dialog,
  DialogTitle,
  DialogContent,
  DialogActions,
  TextField,
  Chip,
  Tooltip,
} from "@mui/material";
import { 
  Business, 
  Settings, 
  Edit, 
  HelpOutline,
  Warning 
} from "@mui/icons-material";
import { useAuth } from "contexts/AuthContext"; // Importation du contexte d'authentification
import axios from "utils/axios";

interface SettingsPageProps {
  orgData: { name: string; organizationSize: number; id: string };
  message?: { type: "success" | "error"; text: string } | null; // Rendu optionnel
  error: string | null;
  loading: boolean;
  renderSettingsField: (label: string, value: string | number, field: "name" | "organizationSize" | "id", editable?: boolean) => React.ReactNode;
  onSave?: (field: "name" | "organizationSize", value: string | number) => void; // Rendu optionnel
}

export default function SettingsPage({
  orgData,
  message, // Utilisé maintenant pour compatibilité
  error,
  loading,
   // Conservé mais optionnel
}: SettingsPageProps) {
  const { token } = useAuth(); // Accès au token via le contexte
  const [editDialogOpen, setEditDialogOpen] = useState(false);
  const [editField, setEditField] = useState<"name" | "organizationSize" | null>(null);
  const [editValue, setEditValue] = useState<string | number>("");
  const [localOrgData, setLocalOrgData] = useState<{ name: string; organizationSize: number; id: string }>(orgData);
  const [localMessage, setLocalMessage] = useState<{ type: "success" | "error"; text: string } | null>(null); // Message local

  // Synchroniser localOrgData avec orgData quand orgData change
  useEffect(() => {
    setLocalOrgData(orgData);
  }, [orgData]);

  const handleEditClick = (field: "name" | "organizationSize", currentValue: string | number) => {
    setEditField(field);
    setEditValue(currentValue);
    setEditDialogOpen(true);
  };

  const handleSaveEdit = useCallback(async () => {
    if (editField && editValue !== "" && token) {
      const updatedValue = editField === "organizationSize" ? Number(editValue) : editValue;
      const updatedData = {
        ...localOrgData,
        [editField]: updatedValue,
      };
      setLocalOrgData(updatedData);

      try {
        await axios.put("/col/organization", { [editField]: updatedValue }, { headers: { Authorization: `Bearer ${token}` } });
        setLocalMessage({ type: "success", text: `${editField.charAt(0).toUpperCase() + editField.slice(1)} updated successfully` });
        console.log(`Saving ${editField}: ${editValue}`);
      } catch (err: any) {
        setLocalMessage({ type: "error", text: err.response?.data?.message || `Failed to update ${editField}` });
      } finally {
        setEditDialogOpen(false);
        setEditField(null);
        // Réinitialiser le message après 3 secondes
        setTimeout(() => setLocalMessage(null), 3000);
      }
    }
  }, [editField, editValue, localOrgData, token]);

  const handleCancelEdit = () => {
    setEditDialogOpen(false);
    setEditField(null);
    setEditValue("");
  };

  const renderSettingsRow = (
    label: string, 
    value: string | number, 
    field: "name" | "organizationSize",
    editable = true
  ) => (
    <Box sx={{ 
      display: "flex", 
      justifyContent: "space-between", 
      alignItems: "center", 
      py: 2.5, 
      px: 3,
      borderBottom: "1px solid #2d3748"
    }}>
      <Typography 
        variant="body2" 
        sx={{ 
          color: "#a0aec0", 
          fontWeight: 500, 
          fontSize: "14px", 
          mb: 0.5 
        }}
      >
        {label}
      </Typography>
      <Typography 
        variant="body2" 
        sx={{ 
          color: "#e2e8f0", 
          fontSize: "14px", 
          mx: 2, // Espacement centré
          flexGrow: 1,
          textAlign: "center"
        }}
      >
        {value}
      </Typography>
      {editable && (
        <IconButton 
          size="small" 
          sx={{ color: "#4299e1" }}
          onClick={() => handleEditClick(field, value)}
        >
          <Edit sx={{ fontSize: "16px" }} />
        </IconButton>
      )}
    </Box>
  );

  const renderProjectVisibilityRow = () => (
    <Box sx={{ 
      display: "flex", 
      justifyContent: "space-between", 
      alignItems: "center", 
      py: 2.5, 
      px: 3,
      borderBottom: "1px solid #2d3748"
    }}>
      <Box sx={{ display: "flex", alignItems: "center", gap: 1 }}>
        <Typography 
          variant="body2" 
          sx={{ 
            color: "#a0aec0", 
            fontWeight: 500, 
            fontSize: "14px", 
            mb: 0.5 
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
              px: 1
            }
          }} 
        />
      </Box>
      <Typography 
        variant="body2" 
        sx={{ 
          color: "#e2e8f0", 
          fontSize: "14px", 
          mx: 2,
          flexGrow: 1,
          textAlign: "center"
        }}
      >
        All organization members
      </Typography>
      <IconButton 
        size="small" 
        sx={{ color: "#4299e1", opacity: 0.5 }}
        disabled
      >
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
        <Typography variant="body1" sx={{ color: "#e2e8f0" }}>{localOrgData.name}</Typography>
      </Box>

      {localMessage && <Alert severity={localMessage.type} sx={{ mb: 3, borderRadius: "8px" }}>{localMessage.text}</Alert>}
      {message && <Alert severity={message.type} sx={{ mb: 3, borderRadius: "8px" }}>{message.text}</Alert>} {/* Utilisation de message pour compatibilité */}
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
                {renderSettingsRow("Organization Name", localOrgData.name, "name")}
                {renderSettingsRow("Organization Size", localOrgData.organizationSize, "organizationSize")}
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
              <Box>
                {renderProjectVisibilityRow()}
              </Box>
            </CardContent>
          </Card>

          <Card sx={{ bgcolor: "#1a202c", border: "1px solid #ef4444", borderRadius: "12px" }}>
            <CardContent sx={{ p: 0 }}>
              <Box sx={{ p: 3, borderBottom: "1px solid #2d3748", display: "flex", alignItems: "center", gap: 1 }}>
                <Warning sx={{ color: "#ef4444", fontSize: "20px" }} />
                <Typography variant="h6" sx={{ color: "#ef4444", fontWeight: 500, fontSize: "16px" }}>
                  Danger Zone
                </Typography>
              </Box>
              <Box sx={{ p: 3, textAlign: "center" }}>
                <Typography variant="body2" sx={{ color: "#a0aec0", mb: 1, fontSize: "14px" }}>
                  Once you delete an organization, there is no going back.
                </Typography>
                <Typography variant="body2" sx={{ color: "#a0aec0", mb: 3, fontSize: "14px" }}>
                  This will permanently delete your organization and all associated data.
                </Typography>
                <Button
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
            </CardContent>
          </Card>
        </>
      )}

      {/* Dialog pour l'édition */}
      <Dialog 
        open={editDialogOpen} 
        onClose={handleCancelEdit}
        PaperProps={{
          sx: {
            bgcolor: "#1a202c",
            border: "1px solid #2d3748",
            borderRadius: "12px",
            minWidth: "400px"
          }
        }}
      >
        <DialogTitle sx={{ color: "#e2e8f0", borderBottom: "1px solid #2d3748" }}>
          Edit {editField === "name" ? "Organization Name" : "Organization Size"}
        </DialogTitle>
        <DialogContent sx={{ pt: 3 }}>
          <TextField
            autoFocus
            fullWidth
            label={editField === "name" ? "Organization Name" : "Organization Size"}
            type={editField === "organizationSize" ? "number" : "text"}
            variant="outlined"
            value={editValue}
            onChange={(e) => setEditValue(
              editField === "organizationSize" 
                ? parseInt(e.target.value) || 0 
                : e.target.value
            )}
            sx={{
              "& .MuiOutlinedInput-root": {
                bgcolor: "#2d3748",
                color: "#e2e8f0",
                "& fieldset": { borderColor: "#4a5568" },
                "&:hover fieldset": { borderColor: "#718096" },
                "&.Mui-focused fieldset": { borderColor: "#4299e1" },
              },
              "& .MuiInputLabel-root": { 
                color: "#a0aec0", 
                "&.Mui-focused": { color: "#4299e1" }
              },
            }}
          />
        </DialogContent>
        <DialogActions sx={{ p: 3, borderTop: "1px solid #2d3748" }}>
          <Button 
            onClick={handleCancelEdit}
            sx={{ 
              color: "#a0aec0",
              "&:hover": { bgcolor: "rgba(160, 174, 192, 0.1)" }
            }}
          >
            Cancel
          </Button>
          <Button 
            onClick={handleSaveEdit}
            variant="contained"
            sx={{ 
              bgcolor: "#4299e1", 
              "&:hover": { bgcolor: "#3182ce" }
            }}
          >
            Save
          </Button>
        </DialogActions>
      </Dialog>
    </Box>
  );
}