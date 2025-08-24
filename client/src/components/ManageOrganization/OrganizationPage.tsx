"use client";

import React, { useState, useEffect } from "react";
import {
  Box,
  Typography,
  IconButton,
  Dialog,
  DialogTitle,
  DialogContent,
  DialogActions,
  TextField,
  Button,
  FormControl,
  InputLabel,
  Select,
  MenuItem,
} from "@mui/material";
import { Edit } from "@mui/icons-material";
import Sidebar from "./Sidebar";
import SettingsPage from "./SettingsPage";
import CollaboratorPage from "./CollaboratorPage";
import FeedbackPage from "./FeedbackPage";
import { useAuth } from "contexts/AuthContext";
import axios from "utils/axios";

export interface Member {
  _id: string;
  name: string;
  surname?: string;
  email: string;
  status: "pending" | "accepted" | "expired" | "AdministratorOrganization";
  invitedDate?: string;
  canInvite?: boolean;
  invitedBy?: string;
}

const teamSizeOptions = [
  { value: "1", label: "1" },
  { value: "2-10", label: "2-10" },
  { value: "11-20", label: "11-20" },
  { value: "21-50", label: "21-50" },
  { value: "51-100", label: "51-100" },
  { value: "101-200", label: "101-200" },
  { value: "201-500", label: "201-500" },
  { value: "500+", label: "500+" },
];

const regionOptions = [
  { value: "Europe", label: "Europe" },
  { value: "US East", label: "US East" },
  { value: "US West", label: "US West" },
  { value: "South America", label: "South America" },
  { value: "East Asia", label: "East Asia" },
  { value: "Australia", label: "Australia" },
  { value: "Singapore", label: "Singapore" },
  { value: "China", label: "China" },
];

export default function OrganizationPage() {
  const { token, logout, account } = useAuth();
  const [activeMenu, setActiveMenu] = useState("settings");
  const [orgData, setOrgData] = useState<{ username: string; link: string; api: string; organizationName: string; teamSize: string; region: string; id: string } | null>(null);
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

  // Edit dialog states
  const [editDialogOpen, setEditDialogOpen] = useState(false);
  const [editField, setEditField] = useState<"organizationName" | "username" | "link" | "api" | "teamSize" | "region" | null>(null);
  const [editValue, setEditValue] = useState<string>("");

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
      console.log("Données reçues de /col/organization:", response.data);
      setOrgData({
        username: response.data.data.username || "",
        link: response.data.data.link || "",
        api: response.data.data.api || "",
        organizationName: response.data.data.organizationName || "Organization Name",
        teamSize: response.data.data.teamSize || "Unknown",
        region: response.data.data.region || "Unknown",
        id: response.data.data.id || "",
      });
    } catch (err: any) {
  console.error("Erreur dans fetchOrganizationData:", err);
  setError(err.response?.data?.message || "Failed to fetch organization data");
  setOrgData({ username: "", link: "", api: "", organizationName: "Organization Name", teamSize: "Unknown", region: "Unknown", id: "" });
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

  // Fonction pour inviter un membre
  const handleInviteMember = async () => {
    if (!inviteEmail.trim()) {
      setMessage({ type: "error", text: "Please fill in the email" });
      return;
    }
    const authToken = token || localStorage.getItem("token");
    if (!authToken) return;
    try {
      await axios.post(
        "/col/invite-account",
        { email: inviteEmail, language: "en", region: orgData?.region, teamSize: orgData?.teamSize },
        { headers: { Authorization: `Bearer ${authToken}` } }
      );
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

  useEffect(() => {
    if (account && token) {
      fetchOrganizationData();
      fetchMembers();
    } else if (!token && !localStorage.getItem("token")) {
      setError("No token available, please log in again");
      window.location.href = "/login";
    }
  }, [account, token]);

  const handleSave = async () => {
    const authToken = token || localStorage.getItem("token");
    if (!authToken || !orgData || !editField) return;
    try {
      await axios.put(
        "/col/organization",
        { [editField]: editValue },
        { headers: { Authorization: `Bearer ${authToken}` } }
      );
      setMessage({ type: "success", text: `${editField.charAt(0).toUpperCase() + editField.slice(1)} updated successfully` });
      setOrgData((prev) => (prev ? { ...prev, [editField]: editValue } : null));
      setEditDialogOpen(false);
      setEditField(null);
      setEditValue("");
    } catch (err: any) {
      setMessage({ type: "error", text: err.response?.data?.message || `Failed to update ${editField}` });
    }
    setTimeout(() => setMessage(null), 3000);
  };

  const handleEditClick = (field: "organizationName" | "username" | "link" | "api" | "teamSize" | "region", currentValue: string) => {
    if (account?.status !== "AdministratorOrganization") {
      setError("You do not have permission to edit this field.");
      return;
    }
    setEditField(field);
    setEditValue(currentValue);
    setEditDialogOpen(true);
  };

  const handleCancelEdit = () => {
    setEditDialogOpen(false);
    setEditField(null);
    setEditValue("");
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
      await axios.post(
        "/col/feedback",
        {
          rating,
          feedbackText,
          suggestFeatures,
          featureSuggestions: suggestFeatures ? featureSuggestions.filter((s) => s.trim()) : [],
        },
        { headers: { Authorization: `Bearer ${authToken}` } }
      );
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

  const renderSettingsField = (label: string, value: string, field: "organizationName" | "username" | "link" | "api" | "teamSize" | "region", editable = true) => (
    <React.Fragment>
      {orgData && (
        <Box
          sx={{
            display: "flex",
            alignItems: "center",
            py: 2.5,
            px: 3,
            borderBottom: "1px solid #2d3748",
          }}
        >
          <Typography
            variant="body2"
            sx={{
              color: "#a0aec0",
              fontWeight: 500,
              fontSize: "14px",
              width: "150px", // Largeur fixe pour aligner les labels
            }}
          >
            {label}
          </Typography>
          <Typography
            variant="body2"
            sx={{
              color: "#e2e8f0",
              fontSize: "14px",
              flexGrow: 1,
              textAlign: "center",
            }}
          >
            {value}
          </Typography>
          {editable && account?.status === "AdministratorOrganization" && (
            <IconButton size="small" sx={{ color: "#4299e1", width: "40px" }} onClick={() => handleEditClick(field, value)}>
              <Edit sx={{ fontSize: "16px" }} />
            </IconButton>
          )}
        </Box>
      )}
    </React.Fragment>
  );

  // Adapter orgData pour CollaboratorPage et FeedbackPage
  const adaptOrgData = (data: { organizationName: string; teamSize: string; region: string; id: string } | null) => {
    if (!data) {
      return { name: "Loading...", organizationSize: 0, id: "", region: "Unknown", teamSize: "Unknown" };
    }
    const teamSizeNumber = data.teamSize === "Unknown" ? 0 : parseInt(data.teamSize.split("-")[0]) || 0;
    return {
      name: data.organizationName,
      organizationSize: teamSizeNumber,
      id: data.id,
      region: data.region,
      teamSize: data.teamSize,
    };
  };

  const renderContent = () => {
  const defaultOrgData = { username: "", link: "", api: "", organizationName: "Loading...", teamSize: "Unknown", region: "Unknown", id: "" };
    const adaptedOrgData = adaptOrgData(orgData || defaultOrgData);
    switch (activeMenu) {
      case "settings":
        return (
          <SettingsPage
            orgData={orgData || defaultOrgData}
            message={message}
            error={error}
            loading={loading}
            renderSettingsField={renderSettingsField}
          />
        );
      case "members":
        return (
          <CollaboratorPage
            orgData={adaptedOrgData}
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
            orgData={adaptedOrgData}
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
      <Dialog
        open={editDialogOpen}
        onClose={handleCancelEdit}
        PaperProps={{
          sx: {
            bgcolor: "#1a202c",
            border: "1px solid #2d3748",
            borderRadius: "12px",
            minWidth: "400px",
          },
        }}
      >
        <DialogTitle sx={{ color: "#e2e8f0", borderBottom: "1px solid #2d3748" }}>
          Edit {
            editField === "organizationName"
              ? "Organization Name"
              : editField === "teamSize"
              ? "Team Size"
              : editField === "region"
              ? "Region"
              : editField === "username"
              ? "Username"
              : editField === "link"
              ? "Ftrack Link"
              : editField === "api"
              ? "API Key"
              : "Field"
          }
        </DialogTitle>
        <DialogContent sx={{ pt: 3 }}>
          {editField === "organizationName" || editField === "username" || editField === "link" || editField === "api" ? (
            <TextField
              autoFocus
              fullWidth
              label={
                editField === "organizationName"
                  ? "Organization Name"
                  : editField === "username"
                  ? "Username"
                  : editField === "link"
                  ? "Ftrack Link"
                  : "API Key"
              }
              value={editValue}
              onChange={(e) => setEditValue(e.target.value)}
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
                  "&.Mui-focused": { color: "#4299e1" },
                },
              }}
            />
          ) : (
            <FormControl fullWidth>
              <InputLabel sx={{ color: "#a0aec0", "&.Mui-focused": { color: "#4299e1" } }}>
                {editField === "teamSize" ? "Team Size" : "Region"}
              </InputLabel>
              <Select
                value={editValue}
                onChange={(e) => setEditValue(e.target.value)}
                sx={{
                  bgcolor: "#2d3748",
                  color: "#e2e8f0",
                  "& .MuiOutlinedInput-notchedOutline": { borderColor: "#4a5568" },
                  "&:hover .MuiOutlinedInput-notchedOutline": { borderColor: "#718096" },
                  "&.Mui-focused .MuiOutlinedInput-notchedOutline": { borderColor: "#4299e1" },
                }}
              >
                {(editField === "teamSize" ? teamSizeOptions : regionOptions).map((option) => (
                  <MenuItem key={option.value} value={option.value}>
                    {option.label}
                  </MenuItem>
                ))}
              </Select>
            </FormControl>
          )}
        </DialogContent>
        <DialogActions sx={{ p: 3, borderTop: "1px solid #2d3748" }}>
          <Button
            onClick={handleCancelEdit}
            sx={{
              color: "#a0aec0",
              "&:hover": { bgcolor: "rgba(160, 174, 192, 0.1)" },
            }}
          >
            Cancel
          </Button>
          <Button
            onClick={handleSave}
            variant="contained"
            sx={{
              bgcolor: "#4299e1",
              "&:hover": { bgcolor: "#3182ce" },
            }}
          >
            Save
          </Button>
        </DialogActions>
      </Dialog>
    </Box>
  );
}