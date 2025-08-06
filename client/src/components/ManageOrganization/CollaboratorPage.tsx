"use client";

import React from "react";
import {
  Box,
  Typography,
  Button,
  Card,
  CardContent,
  Table,
  TableBody,
  TableCell,
  TableContainer,
  TableHead,
  TableRow,
  Paper,
  Avatar,
  IconButton,
  TextField,
  Tabs,
  Tab,
  Alert,
  Chip,
  Dialog,
  DialogTitle,
  DialogContent,
  DialogActions,
} from "@mui/material";
import { Business, PersonAdd, GetApp, Delete, InfoOutlined } from "@mui/icons-material";
import { getStatusDisplay, getStatusColor, getStatusIcon, getAvatarInitials } from "utils/organizationUtils";

interface Member {
  _id: string;
  name: string;
  surname?: string;
  email: string;
  role?: "Owner" | "Admin" | "Member" | "Guest";
  status: "pending" | "accepted" | "expired | Administrator";
  invitedDate?: string;
  canInvite?: boolean;
  invitedBy?: string;
}

interface CollaboratorPageProps {
  orgData: { name: string; organizationSize: number; id: string };
  members: Member[];
  message: { type: "success" | "error"; text: string } | null;
  error: string | null;
  loading: boolean;
  account: any;
  handleExportCSV: () => void;
  handleOpenInviteDialog: () => void;
  inviteDialogOpen: boolean;
  setInviteDialogOpen: (open: boolean) => void;
  inviteEmail: string;
  setInviteEmail: (email: string) => void;
  handleInviteMember: () => void;
  selectedMember: Member | null;
  setSelectedMember: (member: Member | null) => void;
  deleteDialogOpen: boolean;
  setDeleteDialogOpen: (open: boolean) => void;
  handleRemoveMember: () => void;
  filterText: string;
  setFilterText: (text: string) => void;
  membersTab: number;
  setMembersTab: (value: number) => void;
}

export default function CollaboratorPage({
  orgData,
  members,
  message,
  error,
  loading,
  account,
  handleExportCSV,
  handleOpenInviteDialog,
  inviteDialogOpen,
  setInviteDialogOpen,
  inviteEmail,
  setInviteEmail,
  handleInviteMember,
  selectedMember,
  setSelectedMember,
  deleteDialogOpen,
  setDeleteDialogOpen,
  handleRemoveMember,
  filterText,
  setFilterText,
  membersTab,
  setMembersTab,
}: CollaboratorPageProps) {
  return (
    <Box>
      <Box sx={{ display: "flex", justifyContent: "space-between", alignItems: "center", mb: 3 }}>
        <Typography variant="h4" sx={{ color: "#e2e8f0", fontWeight: 600 }}>
          Organization Members
        </Typography>
        <Box sx={{ display: "flex", gap: 2 }}>
          <Button
            variant="outlined"
            startIcon={<GetApp />}
            onClick={handleExportCSV}
            sx={{ borderColor: "#4a5568", color: "#e2e8f0", "&:hover": { borderColor: "#718096" } }}
          >
            Export CSV
          </Button>
          <Button
            variant="contained"
            startIcon={<PersonAdd />}
            onClick={handleOpenInviteDialog}
            disabled={!account?.canInvite || (account?.invitedBy && account?.invitedBy !== account?._id) || !account?._id}
            sx={{
              bgcolor: "#4299e1",
              "&:hover": { bgcolor: "#3182ce" },
              "&:disabled": { bgcolor: "#4a5568" },
            }}
          >
            Invite organization members
          </Button>
        </Box>
      </Box>

      <Box sx={{ display: "flex", alignItems: "center", mb: 3 }}>
        <Avatar sx={{ width: 32, height: 32, bgcolor: "transparent", border: "1px solid #2d3748", mr: 2 }}>
          <Business sx={{ fontSize: "16px", color: "#4299e1" }} />
        </Avatar>
        <Typography variant="body1" sx={{ color: "#e2e8f0" }}>{orgData.name}</Typography>
      </Box>

      {message && <Alert severity={message.type} sx={{ mb: 3 }}>{message.text}</Alert>}
      {error && <Alert severity="error" sx={{ mb: 3 }}>{error}</Alert>}

      <Box sx={{ borderBottom: "1px solid #2d3748", mb: 3 }}>
        <Tabs
          value={membersTab}
          onChange={(_, newValue) => setMembersTab(newValue)}
          sx={{
            "& .MuiTab-root": { color: "#a0aec0", textTransform: "none", fontSize: "14px", fontWeight: 500 },
            "& .Mui-selected": { color: "#4299e1" },
            "& .MuiTabs-indicator": { backgroundColor: "#4299e1" },
          }}
        >
          <Tab label="Members" />
          <Tab label="Pending invites" />
        </Tabs>
      </Box>

      <Box sx={{ display: "flex", gap: 2, mb: 3 }}>
        <Box sx={{ flexGrow: 1 }} />
        <TextField
          placeholder="Search organization members"
          value={filterText}
          onChange={(e) => setFilterText(e.target.value)}
          size="small"
          sx={{
            width: 300,
            "& .MuiOutlinedInput-root": {
              bgcolor: "#2d3748",
              color: "#e2e8f0",
              "& fieldset": { borderColor: "#4a5568" },
              "&:hover fieldset": { borderColor: "#718096" },
            },
            "& .MuiInputBase-input::placeholder": { color: "#a0aec0" },
          }}
        />
      </Box>

      <Card sx={{ bgcolor: "#1a202c", border: "1px solid #2d3748", borderRadius: "12px" }}>
        <CardContent sx={{ p: 0 }}>
          <TableContainer component={Paper} sx={{ bgcolor: "transparent" }}>
            <Table>
              <TableHead>
                <TableRow sx={{ borderBottom: "1px solid #2d3748" }}>
                  <TableCell sx={{ color: "#a0aec0", fontWeight: 600, borderBottom: "1px solid #2d3748", fontSize: "14px" }}>
                    Organization members ↑
                  </TableCell>
                  <TableCell sx={{ color: "#a0aec0", fontWeight: 600, borderBottom: "1px solid #2d3748", fontSize: "14px" }}>
                    Roles
                  </TableCell>
                  <TableCell sx={{ color: "#a0aec0", fontWeight: 600, borderBottom: "1px solid #2d3748", fontSize: "14px" }}>
                    Invited Date
                  </TableCell>
                  <TableCell sx={{ width: 50, borderBottom: "1px solid #2d3748" }}></TableCell>
                </TableRow>
              </TableHead>
              <TableBody>
                {loading ? (
                  <TableRow>
                    <TableCell colSpan={4} sx={{ textAlign: "center", py: 2 }}>
                      <Typography color="#a0aec0">Loading...</Typography>
                    </TableCell>
                  </TableRow>
                ) : membersTab === 1 && !members.some((m) => m.status === "pending" && m.email.toLowerCase().includes(filterText.toLowerCase()) && m._id !== account?._id) ? (
                  <TableRow>
                    <TableCell colSpan={4} sx={{ textAlign: "center", py: 4, color: "#a0aec0" }}>
                      <Box sx={{ display: "flex", alignItems: "center", justifyContent: "center", gap: 1 }}>
                        <InfoOutlined sx={{ color: "#a0aec0" }} />
                        <Typography>No pending invitations</Typography>
                      </Box>
                    </TableCell>
                  </TableRow>
                ) : (
                  members
                    .filter((member) => member.email.toLowerCase().includes(filterText.toLowerCase()))
                    .filter((member) => {
                      const isMemberTab = membersTab === 0;
                      const isPendingTab = membersTab === 1;
                      const isOwner = account?._id && member._id === account._id && account.canInvite;
                      const showInMembers = isMemberTab && (member.status === "accepted" || isOwner);
                      const showInPending = isPendingTab && member.status === "pending" && !isOwner;
                      return showInMembers || showInPending;
                    })
                    .map((member) => (
                      <TableRow key={member._id} sx={{ borderBottom: "1px solid #2d3748" }}>
                        <TableCell sx={{ borderBottom: "1px solid #2d3748", py: 2 }}>
                          <Box sx={{ display: "flex", alignItems: "center" }}>
                            <Avatar
                              sx={{
                                width: 32,
                                height: 32,
                                bgcolor: account?._id && member._id === account._id && account.canInvite ? "#4299e1" : "#38b2ac",
                                fontSize: "14px",
                                mr: 2,
                              }}
                            >
                              {getAvatarInitials(member.name, member.surname)}
                            </Avatar>
                            <Box>
                              <Box sx={{ display: "flex", alignItems: "center", gap: 1 }}>
                                <Typography
                                  variant="body2"
                                  sx={{ color: "#e2e8f0", fontWeight: 500, fontSize: "14px" }}
                                >
                                  {member._id === account?._id ? ` ${member.name} ${member.surname || ""} (You) ` : `${member.name} ${member.surname || ""}`}
                                </Typography>
                                <Chip
                                  label={
                                    <Box sx={{ display: "flex", alignItems: "center", gap: 0.5 }}>
                                      <span>{getStatusIcon(member, account)}</span>
                                      <span>{getStatusDisplay(member, account)}</span>
                                    </Box>
                                  }
                                  size="small"
                                  sx={{
                                    bgcolor: `${getStatusColor(member, account)}20`,
                                    color: getStatusColor(member, account),
                                    border: `1px solid ${getStatusColor(member, account)}40`,
                                    height: "20px",
                                    fontSize: "11px",
                                  }}
                                />
                              </Box>
                              <Typography variant="caption" sx={{ color: "#a0aec0", fontSize: "12px" }}>
                                {member.email}
                              </Typography>
                            </Box>
                          </Box>
                        </TableCell>
                        <TableCell sx={{ borderBottom: "1px solid #2d3748", color: "#a0aec0", fontSize: "14px" }}>
                          {account?._id && member._id === account._id && account.canInvite ? "-" : "-"}
                        </TableCell>
                        <TableCell sx={{ borderBottom: "1px solid #2d3748", color: "#a0aec0", fontSize: "14px" }}>
                          {member.invitedDate || "-"}
                        </TableCell>
                        <TableCell sx={{ borderBottom: "1px solid #2d3748" }}>
                          {account?._id && account._id !== member._id && account?.canInvite ? (
                            <IconButton
                              onClick={() => {
                                setSelectedMember(member);
                                setDeleteDialogOpen(true);
                              }}
                              sx={{ color: "#ef4444" }}
                              size="small"
                            >
                              <Delete sx={{ fontSize: "18px" }} />
                            </IconButton>
                          ) : null}
                        </TableCell>
                      </TableRow>
                    ))
                )}
              </TableBody>
            </Table>
          </TableContainer>

          <Box
            sx={{
              p: 2,
              borderTop: "1px solid #2d3748",
              display: "flex",
              justifyContent: "space-between",
              alignItems: "center",
            }}
          >
            <Typography variant="caption" sx={{ color: "#a0aec0" }}>
              Rows per page: 50 ▼
            </Typography>
            <Typography variant="caption" sx={{ color: "#a0aec0" }}>
              {members
                .filter((m) => m.email.toLowerCase().includes(filterText.toLowerCase()))
                .filter((m) => {
                  const isMemberTab = membersTab === 0;
                  const isPendingTab = membersTab === 1;
                  const isOwner = account?._id && m._id === account._id && account.canInvite;
                  const showInMembers = isMemberTab && (m.status === "accepted" || isOwner);
                  const showInPending = isPendingTab && m.status === "pending" && !isOwner;
                  return showInMembers || showInPending;
                }).length}{" "}
              of {members.length}
            </Typography>
          </Box>
        </CardContent>
      </Card>

      <Dialog
        open={inviteDialogOpen}
        onClose={() => setInviteDialogOpen(false)}
        maxWidth="sm"
        fullWidth
        PaperProps={{ sx: { bgcolor: "#1a202c", border: "1px solid #2d3748" } }}
      >
        <DialogTitle sx={{ color: "#e2e8f0", borderBottom: "1px solid #2d3748" }}>Invite New Member</DialogTitle>
        <DialogContent sx={{ pt: 3 }}>
          <TextField
            autoFocus
            label="Email Address"
            type="email"
            fullWidth
            variant="outlined"
            value={inviteEmail}
            onChange={(e) => setInviteEmail(e.target.value)}
            sx={{
              mb: 2,
              "& .MuiOutlinedInput-root": {
                bgcolor: "#2d3748",
                color: "#e2e8f0",
              },
              "& .MuiInputLabel-root": { color: "#a0aec0" },
            }}
          />
        </DialogContent>
        <DialogActions sx={{ p: 3, borderTop: "1px solid #2d3748" }}>
          <Button onClick={() => setInviteDialogOpen(false)} sx={{ color: "#a0aec0" }}>
            Cancel
          </Button>
          <Button
            onClick={handleInviteMember}
            variant="contained"
            disabled={!account?.canInvite || (account?.invitedBy && account?.invitedBy !== account?._id) || !account?._id}
            sx={{
              bgcolor: "#4299e1",
              "&:hover": { bgcolor: "#3182ce" },
              "&:disabled": { bgcolor: "#4a5568" },
            }}
          >
            Send Invitation
          </Button>
        </DialogActions>
      </Dialog>

      <Dialog
        open={deleteDialogOpen}
        onClose={() => setDeleteDialogOpen(false)}
        maxWidth="sm"
        fullWidth
        PaperProps={{ sx: { bgcolor: "#1a202c", border: "1px solid #2d3748" } }}
      >
        <DialogTitle sx={{ color: "#e2e8f0", borderBottom: "1px solid #2d3748", pb: 2 }}>
          <Typography variant="h6" sx={{ fontWeight: 600 }}>
            Confirm deletion
          </Typography>
          <Typography variant="body2" color="#a0aec0" sx={{ mt: 1 }}>
            Are you sure you want to delete the member {selectedMember?.email}?
          </Typography>
        </DialogTitle>
        <DialogActions sx={{ borderTop: "1px solid #2d3748", pt: 2, px: 3, pb: 3 }}>
          <Button onClick={() => setDeleteDialogOpen(false)} sx={{ color: "#a0aec0", textTransform: "none" }}>
            Cancel
          </Button>
          <Button
            onClick={handleRemoveMember}
            variant="contained"
            sx={{
              bgcolor: "#ef4444",
              "&:hover": { bgcolor: "#dc2626" },
              textTransform: "none",
              px: 3,
            }}
          >
            Delete
          </Button>
        </DialogActions>
      </Dialog>
    </Box>
  );
}