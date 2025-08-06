"use client";

import React, { useState, useEffect } from "react";
import { Box, TableRow, TableCell, Typography, IconButton, TextField, Button } from "@mui/material";
import { Edit, GetApp } from "@mui/icons-material";
import Sidebar from "./Sidebar";
import SettingsPage from "./SettingsPage";
import CollaboratorPage from "./CollaboratorPage";
import FeedbackPage from "./FeedbackPage";
import { useAuth } from "contexts/AuthContext";
import axios from "utils/axios";

interface Member {
  _id: string;
  name: string;
  surname?: string;
  email: string;
  status: "pending" | "accepted" | "expired | Administrator";
  invitedDate?: string;
  canInvite?: boolean;
  invitedBy?: string;
}

export default function OrganizationPage() {
  const { token, logout, account } = useAuth();
  const [activeMenu, setActiveMenu] = useState("settings");
  const [orgData, setOrgData] = useState<{ name: string; organizationSize: number; id: string } | null>(null); // Initialisé à null pour indiquer un état de chargement
  const [members, setMembers] = useState<Member[]>([]);
  const [message, setMessage] = useState<{ type: "success" | "error"; text: string } | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [inviteDialogOpen, setInviteDialogOpen] = useState(false);
  const [deleteDialogOpen, setDeleteDialogOpen] = useState(false);
  const [inviteEmail, setInviteEmail] = useState("");
  const [selectedMember, setSelectedMember] = useState<Member | null>(null);
  const [filterText, setFilterText] = useState("");
  const [membersTab, setMembersTab] = useState(0);

  // Feedback states
  const [rating, setRating] = useState<number | null>(null);
  const [feedbackText, setFeedbackText] = useState("");
  const [suggestFeatures, setSuggestFeatures] = useState(false);
  const [featureSuggestions, setFeatureSuggestions] = useState<string[]>([""]);

  // Fonction pour charger les données de l'organisation
  const fetchOrganizationData = async () => {
    const authToken = token || localStorage.getItem("token");
    if (!authToken) {
      setError("No token available, please log in again");
      window.location.href = "/login";
      return;
    }
    try {
      const response = await axios.get("/col/organization", { headers: { Authorization: `Bearer ${authToken}` } });
      console.log("Données reçues de /col/organization:", response.data); // Débogage
      setOrgData({
        name: response.data.data.name || "Organization Name",
        organizationSize: response.data.data.organizationSize || 5,
        id: response.data.data.id || "",
      });
    } catch (err: any) {
      console.error("Erreur dans fetchOrganizationData:", err);
      setError(err.response?.data?.message || "Failed to fetch organization data");
      setOrgData({ name: "Organization Name", organizationSize: 5, id: "" }); // Valeur par défaut en cas d'erreur
    } finally {
      setLoading(false);
    }
  };

  // Charger les membres
  const fetchMembers = async () => {
    const authToken = token || localStorage.getItem("token");
    if (!authToken) return;
    try {
      const response = await axios.get("/col/accounts", { headers: { Authorization: `Bearer ${authToken}` } });
      const enrichedMembers = response.data.map((m: any) => ({
        _id: m.id,
        name: m.name || "",
        surname: m.surname || (account && account._id === m.id ? account.surname : ""),
        email: m.email,
        status: m.status || "pending",
        invitedDate: m.invitedDate || null,
        canInvite: m.canInvite || (account && account._id === m.id && account.canInvite),
        invitedBy: m.invitedBy || null,
      }));
      setMembers(enrichedMembers);
    } catch (err: any) {
      setError(err.response?.data?.message || "Failed to fetch members");
      if (err.response?.status === 401) logout();
    }
  };

  useEffect(() => {
    if (account && token) {
      fetchOrganizationData();
      fetchMembers();
    } else if (!token && !localStorage.getItem("token")) {
      setError("No token available, please log in again");
      window.location.href = "/login";
    }
  }, [account, token]);

  const handleSave = async (field: "name" | "organizationSize") => {
    const authToken = token || localStorage.getItem("token");
    if (!authToken || !orgData) return;
    try {
      await axios.put("/col/organization", { [field]: orgData[field] }, { headers: { Authorization: `Bearer ${authToken}` } });
      setMessage({ type: "success", text: `${field.charAt(0).toUpperCase() + field.slice(1)} updated successfully` });
      fetchOrganizationData(); // Recharger les données après la sauvegarde
    } catch (err: any) {
      setMessage({ type: "error", text: err.response?.data?.message || `Failed to update ${field}` });
    }
    setTimeout(() => setMessage(null), 3000);
  };

  const handleEditToggle = (field: "name" | "organizationSize") => {
    if (orgData) {
      setOrgData((prev) => (prev ? { ...prev, [field]: prev[field] } : null)); // No-op, ajuste si nécessaire
    }
  };

  const handleExportCSV = () => {
    const csvContent = [["Name", "Email", "Status", "Invited Date"], ...members.map((m) => [m.name, m.email, m.status, m.invitedDate || "-"])].map((row) => row.join(",")).join("\n");
    const blob = new Blob([csvContent], { type: "text/csv" });
    const url = window.URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url;
    a.download = "organization_members.csv";
    a.click();
    window.URL.revokeObjectURL(url);
  };

  const handleOpenInviteDialog = () => {
    if (!account?.canInvite || (account?.invitedBy && account?.invitedBy !== account?._id) || !account?._id) {
      setError("You do not have permission to invite collaborators.");
      return;
    }
    setInviteDialogOpen(true);
  };

  const handleInviteMember = async () => {
    if (!inviteEmail.trim()) {
      setMessage({ type: "error", text: "Please fill in the email" });
      return;
    }
    const authToken = token || localStorage.getItem("token");
    if (!authToken) return;
    try {
      await axios.post("/col/invite-account", { email: inviteEmail, language: "en" }, { headers: { Authorization: `Bearer ${authToken}` } });
      setMessage({ type: "success", text: "Invitation sent successfully" });
      setInviteDialogOpen(false);
      setInviteEmail("");
      fetchMembers();
    } catch (err: any) {
      setMessage({ type: "error", text: err.response?.data?.message || "Failed to send invitation" });
      if (err.response?.status === 401) logout();
    }
    setTimeout(() => setMessage(null), 3000);
  };

  const handleRemoveMember = async () => {
    if (!selectedMember) return;
    const authToken = token;
    if (!authToken) return;
    try {
      await axios.delete(`/col/accounts/${selectedMember._id}`, { headers: { Authorization: `Bearer ${authToken}` } });
      setMessage({ type: "success", text: "Member deleted successfully" });
      setSelectedMember(null);
      setDeleteDialogOpen(false);
      fetchMembers();
    } catch (err: any) {
      setMessage({ type: "error", text: err.response?.data?.message || "Failed to delete member" });
      if (err.response?.status === 401) logout();
    }
    setTimeout(() => setMessage(null), 3000);
  };

  const handleAddFeatureSuggestion = () => setFeatureSuggestions([...featureSuggestions, ""]);
  const handleRemoveFeatureSuggestion = (index: number) => setFeatureSuggestions(featureSuggestions.filter((_, i) => i !== index));
  const handleFeatureSuggestionChange = (index: number, value: string) => {
    const newSuggestions = [...featureSuggestions];
    newSuggestions[index] = value;
    setFeatureSuggestions(newSuggestions);
  };

  const handleSubmitFeedback = async () => {
    const authToken = token || localStorage.getItem("token");
    if (!authToken) return;
    try {
      await axios.post("/col/feedback", {
        rating,
        feedbackText,
        suggestFeatures,
        featureSuggestions: suggestFeatures ? featureSuggestions.filter((s) => s.trim()) : [],
      }, { headers: { Authorization: `Bearer ${authToken}` } });
      setMessage({ type: "success", text: "Thank you for your feedback!" });
      setRating(null);
      setFeedbackText("");
      setSuggestFeatures(false);
      setFeatureSuggestions([""]);
    } catch (err: any) {
      setMessage({ type: "error", text: err.response?.data?.message || "Failed to submit feedback" });
    }
    setTimeout(() => setMessage(null), 3000);
  };

  const renderSettingsField = (label: string, value: string | number, field: "name" | "organizationSize" | "id", editable = true) => (
    <React.Fragment>
      {orgData && (
        <TableRow sx={{ borderBottom: "1px solid #2d3748" }}>
          <TableCell sx={{ color: "#a0aec0", fontWeight: 500, fontSize: "14px", borderBottom: "1px solid #2d3748", py: 2.5, px: 3, width: "30%" }}>{label}</TableCell>
          <TableCell sx={{ borderBottom: "1px solid #2d3748", py: 2.5, px: 3 }}>
            {field === "id" ? (
              <Box sx={{ display: "flex", alignItems: "center", gap: 1.5 }}>
                <Typography variant="body2" sx={{ color: "#e2e8f0", fontSize: "14px" }}>{value}</Typography>
                <IconButton size="small" sx={{ color: "#4299e1" }}>
                  <GetApp sx={{ fontSize: "16px" }} />
                </IconButton>
              </Box>
            ) : (
              <Box sx={{ display: "flex", alignItems: "center", gap: 2 }}>
                <TextField
                  value={value}
                  onChange={(e: React.ChangeEvent<HTMLInputElement>) => {
                    if (orgData) {
                      setOrgData((prev) =>
                        prev ? { ...prev, [field]: field === "organizationSize" ? parseInt(e.target.value) || 5 : e.target.value } : null
                      );
                    }
                  }}
                  disabled={!editable}
                  variant="outlined"
                  size="small"
                  type={field === "organizationSize" ? "number" : "text"}
                  sx={{
                    flexGrow: 1,
                    maxWidth: 400,
                    "& .MuiOutlinedInput-root": {
                      bgcolor: editable ? "#2d3748" : "transparent",
                      color: "#e2e8f0",
                      "& fieldset": { borderColor: editable ? "#4a5568" : "transparent" },
                      "&:hover fieldset": { borderColor: "#718096" },
                      "&.Mui-disabled": { "& fieldset": { borderColor: "transparent" } },
                    },
                    "& .MuiInputLabel-root": { color: "#a0aec0", fontSize: "14px" },
                  }}
                />
                {editable && (
                  <Box sx={{ display: "flex", gap: 1 }}>
                    <IconButton size="small" sx={{ color: "#4299e1" }} onClick={() => handleEditToggle(field)}>
                      <Edit sx={{ fontSize: "16px" }} />
                    </IconButton>
                    {editable && (
                      <Button
                        variant="contained"
                        size="small"
                        onClick={() => handleSave(field)}
                        sx={{ bgcolor: "#4299e1", "&:hover": { bgcolor: "#3182ce" }, fontSize: "12px", px: 2, py: 0.5 }}
                      >
                        Save
                      </Button>
                    )}
                  </Box>
                )}
              </Box>
            )}
          </TableCell>
        </TableRow>
      )}
    </React.Fragment>
  );

  const renderContent = () => {
    switch (activeMenu) {
      case "settings":
        return (
          <SettingsPage
            orgData={orgData || { name: "Loading...", organizationSize: 0, id: "" }} // Valeur par défaut pendant le chargement
            message={message}
            error={error}
            loading={loading}
            renderSettingsField={renderSettingsField}
          />
        );
      case "members":
        return (
          <CollaboratorPage
            orgData={orgData || { name: "Loading...", organizationSize: 0, id: "" }}
            members={members}
            message={message}
            error={error}
            loading={loading}
            account={account}
            handleExportCSV={handleExportCSV}
            handleOpenInviteDialog={handleOpenInviteDialog}
            inviteDialogOpen={inviteDialogOpen}
            setInviteDialogOpen={setInviteDialogOpen}
            inviteEmail={inviteEmail}
            setInviteEmail={setInviteEmail}
            handleInviteMember={handleInviteMember}
            selectedMember={selectedMember}
            setSelectedMember={setSelectedMember}
            deleteDialogOpen={deleteDialogOpen}
            setDeleteDialogOpen={setDeleteDialogOpen}
            handleRemoveMember={handleRemoveMember}
            filterText={filterText}
            setFilterText={setFilterText}
            membersTab={membersTab}
            setMembersTab={setMembersTab}
          />
        );
      case "feedback":
        return (
          <FeedbackPage
            orgData={orgData || { name: "Loading...", organizationSize: 0, id: "" }}
            rating={rating}
            setRating={setRating}
            feedbackText={feedbackText}
            setFeedbackText={setFeedbackText}
            suggestFeatures={suggestFeatures}
            setSuggestFeatures={setSuggestFeatures}
            featureSuggestions={featureSuggestions}
            handleAddFeatureSuggestion={handleAddFeatureSuggestion}
            handleRemoveFeatureSuggestion={handleRemoveFeatureSuggestion}
            handleFeatureSuggestionChange={handleFeatureSuggestionChange}
            handleSubmitFeedback={handleSubmitFeedback}
            message={message}
          />
        );
      default:
        return null;
    }
  };

  return (
    <Box sx={{ display: "flex", minHeight: "100vh", bgcolor: "#0f172a", flexDirection: { xs: "column", sm: "row" } }}>
      <Sidebar activeMenu={activeMenu} setActiveMenu={setActiveMenu} />
      <Box sx={{ flexGrow: 1, p: { xs: 2, sm: 4 }, overflow: "auto", bgcolor: "#0f172a" }}>{renderContent()}</Box>
    </Box>
  );
}