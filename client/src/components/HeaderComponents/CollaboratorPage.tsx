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
  Table,
  TableBody,
  TableCell,
  TableContainer,
  TableHead,
  TableRow,
  Dialog,
  DialogTitle,
  DialogContent,
  DialogActions,
  Avatar,
  Chip,
  IconButton,
} from "@mui/material";
import DeleteIcon from '@mui/icons-material/Delete';
import { useState, useEffect } from "react";
import axios from 'utils/axios';
import { useAuth } from 'contexts/AuthContext';

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
    MuiPaper: { styleOverrides: { root: { backgroundImage: "none", backgroundColor: "#1a202c" } } },
    MuiTableCell: { styleOverrides: { root: { borderBottom: "1px solid #2d3748", color: "#e2e8f0", fontSize: "14px" } } },
    MuiTableHead: { styleOverrides: { root: { backgroundColor: "#2d3748" } } },
  },
});

interface Account {
  _id: string;
  name: string;
  surname?: string;
  email: string;
  password?: string;
  role: 'user' | 'admin';
  isVerified: boolean;
  verificationCode?: string;
  verificationCodeExpires?: Date;
  mfaSecret?: string;
  mfaEnabled: boolean;
  organizationName: string;
  invitedBy?: string;
  canInvite: boolean;
  mustCompleteProfile: boolean;
  status?: "pending" | "accepted" | "expired";
  invitedDate?: string;
}

export default function CollaboratorPage() {
  const { token, logout, account } = useAuth();
  const [filterText, setFilterText] = useState("");
  const [accounts, setAccounts] = useState<Account[]>([]);
  const [openInviteDialog, setOpenInviteDialog] = useState(false);
  const [openDeleteDialog, setOpenDeleteDialog] = useState(false);
  const [inviteEmail, setInviteEmail] = useState("");
  const [accountToDelete, setAccountToDelete] = useState<Account | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [success, setSuccess] = useState<string | null>(null);

  useEffect(() => {
    console.log('Heure du client:', new Date().toISOString());
    console.log('Token depuis AuthContext:', token);
    console.log('Utilisateur connecté (account):', account);
    const localStorageToken = localStorage.getItem('token');
    console.log('Token depuis localStorage:', localStorageToken);
    if (!token && !localStorageToken) {
      console.log('Aucun token trouvé, redirection vers /login');
      setError('Aucun token disponible, veuillez vous reconnecter');
      setLoading(false);
      window.location.href = '/login';
      return;
    }
    fetchAccounts();
  }, [token]);

  const fetchAccounts = async () => {
    const authToken = token || localStorage.getItem('token');
    console.log('Token utilisé pour fetchAccounts:', authToken);
    if (!authToken) {
      console.log('Aucun token trouvé dans fetchAccounts, redirection vers /login');
      setError('Aucun token disponible, veuillez vous reconnecter');
      setLoading(false);
      window.location.href = '/login';
      return;
    }
    try {
      const response = await axios.get('/col/accounts', { headers: { Authorization: `Bearer ${authToken}` } });
      console.log('Réponse des comptes:', response.data);
      const enrichedAccounts = response.data.map((a: any) => ({
        ...a,
        _id: a._id?.$oid || a._id || a.id, // Gérer le format MongoDB avec $oid
        canInvite: a.canInvite || false,
        invitedBy: a.invitedBy || null,
        status: a.status || "pending",
        invitedDate: a.invitedDate || null,
        name: a.name || '', // Assurer que name est défini
        surname: a.surname || '', // Assurer que surname est défini
      }));
      setAccounts(enrichedAccounts);
      setLoading(false);
    } catch (err: any) {
      const message = err.response?.data?.message || 'Échec de la récupération des comptes';
      setError(message);
      setLoading(false);
      console.error('Erreur de fetch:', err.response ? err.response.data : err.message);
      if (err.response?.status === 401 && err.response?.data?.message.includes('Token invalide ou expiré')) {
        console.log('Token invalide détecté, déconnexion forcée');
        logout();
      }
    }
  };

  const handleOpenInviteDialog = () => {
    if (!account?.canInvite || (account?.invitedBy && account?.invitedBy !== account?._id) || !account?._id) {
      setError('Vous n\'avez pas la permission d\'inviter des collaborateurs.');
      return;
    }
    setOpenInviteDialog(true);
  };

  const handleCloseInviteDialog = () => {
    setOpenInviteDialog(false);
    setInviteEmail("");
    setError(null);
    setSuccess(null);
  };

  const handleInvite = async () => {
    if (inviteEmail.trim()) {
      try {
        const authToken = token || localStorage.getItem('token');
        console.log('Token avant envoi:', authToken);
        const response = await axios.post('/col/invite-account', { email: inviteEmail, language: 'fr' }, { headers: { Authorization: `Bearer ${authToken}` } });
        console.log('Réponse de l\'invitation:', response.data);
        setSuccess('Invitation envoyée avec succès.');
        handleCloseInviteDialog();
        fetchAccounts();
      } catch (err: any) {
        const status = err.response?.status;
        const message = err.response?.data?.message || "Échec de l'envoi de l'invitation";
        console.error('Erreur détaillée:', err.response ? err.response.data : err.message);
        setError(message);
        if (status === 401 && message.includes('Token invalide ou expiré')) {
          console.log('Token invalide détecté, déconnexion forcée');
          logout();
        } else if (status === 403) {
          console.log('Permission refusée pour inviter des collaborateurs');
          setError('Vous n\'avez pas la permission d\'inviter des collaborateurs.');
        } else if (status === 400) {
          console.log('Erreur de validation:', message);
          setError(message);
        }
      }
    } else {
      setError("Veuillez remplir l'email.");
    }
  };

  const handleOpenDeleteDialog = (accountItem: Account) => {
    if (!account || account._id === accountItem._id || !account?.canInvite) {
      return;
    }
    setAccountToDelete(accountItem);
    setOpenDeleteDialog(true);
  };

  const handleCloseDeleteDialog = () => {
    setOpenDeleteDialog(false);
    setAccountToDelete(null);
    setError(null);
    setSuccess(null);
  };

  const handleDelete = async () => {
    if (!accountToDelete) return;
    try {
      const authToken = token || localStorage.getItem('token');
      console.log('Token avant suppression:', authToken);
      await axios.delete(`/col/accounts/${accountToDelete._id}`, { headers: { Authorization: `Bearer ${authToken}` } });
      console.log('Compte supprimé:', accountToDelete._id);
      setSuccess('Compte supprimé avec succès.');
      handleCloseDeleteDialog();
      fetchAccounts();
    } catch (err: any) {
      const status = err.response?.status;
      const message = err.response?.data?.message || "Échec de la suppression du compte";
      console.error('Erreur détaillée:', err.response ? err.response.data : err.message);
      setError(message);
      if (status === 401 && message.includes('Token invalide ou expiré')) {
        console.log('Token invalide détecté, déconnexion forcée');
        logout();
      } else if (status === 403) {
        console.log('Permission refusée pour supprimer des collaborateurs');
        setError('Vous n\'avez pas la permission de supprimer des collaborateurs.');
      }
    }
  };

  const getStatusDisplay = (accountItem: Account) => {
    if (accountItem.canInvite) return "Administrator";
    if (accountItem.invitedBy && accountItem.invitedBy !== account?._id) return "-";
    return accountItem.status || "pending";
  };

  const getStatusColor = (accountItem: Account) => {
    if (accountItem.canInvite) return "#4299e1"; // Couleur spéciale pour Administrator
    if (accountItem.invitedBy && accountItem.invitedBy !== account?._id) return "#6b7280";
    return {
      accepted: "#10b981",
      pending: "#f59e0b",
      expired: "#ef4444",
    }[accountItem.status || "pending"] || "#6b7280";
  };

  const getStatusIcon = (accountItem: Account) => {
    if (accountItem.canInvite) return "👑"; // Icône spéciale pour Administrator
    if (accountItem.invitedBy && accountItem.invitedBy !== account?._id) return "-";
    return {
      accepted: "✓",
      pending: "⏳",
      expired: "⚠️",
    }[accountItem.status || "pending"] || "?";
  };

  const getAvatarInitials = (name: string, surname?: string) => {
    return `${name.charAt(0)}${surname?.charAt(0) || ''}`.toUpperCase();
  };

  const filteredAccounts = accounts.filter(accountItem =>
    accountItem.email.toLowerCase().includes(filterText.toLowerCase())
  );

  return (
    <ThemeProvider theme={darkTheme}>
      <CssBaseline />
      <Box sx={{ display: "flex", flexDirection: "column", height: "100vh", bgcolor: "#0f172a" }}>
        <Box sx={{ flexGrow: 1, display: "flex", flexDirection: "column", overflow: "hidden" }}>
          <Box sx={{ borderBottom: "1px solid #2d3748", bgcolor: "#1a202c", px: 3, py: 3 }}>
            <Box sx={{ display: "flex", justifyContent: "space-between", alignItems: "center" }}>
              <Box>
                <Typography variant="h4" color="#e2e8f0" sx={{ fontWeight: 600, mb: 1 }}>
                  Team Collaborators
                </Typography>
                <Typography variant="body1" color="#a0aec0">
                  Manage your project team members and invitations
                </Typography>
              </Box>
              <Box sx={{ display: "flex", gap: 2 }}>
                <Button
                  variant="contained"
                  onClick={handleOpenInviteDialog}
                  disabled={!account?.canInvite || (account?.invitedBy && account?.invitedBy !== account?._id) || !account?._id}
                  sx={{ bgcolor: "#4299e1", "&:hover": { bgcolor: "#3182ce" }, "&:disabled": { bgcolor: "#4a5568" }, textTransform: "none", fontWeight: 500, px: 3, py: 1.5 }}
                >
                  Invite Collaborator
                </Button>
              </Box>
            </Box>
          </Box>

          <Box sx={{ borderBottom: "1px solid #2d3748", bgcolor: "#1a202c", px: 3, py: 2 }}>
            <TextField
              placeholder="Search collaborators by email..."
              value={filterText}
              onChange={(e) => setFilterText(e.target.value)}
              size="small"
              sx={{
                width: 450,
                "& .MuiOutlinedInput-root": {
                  bgcolor: "#2d3748",
                  "& fieldset": { borderColor: "#4a5568" },
                  "&:hover fieldset": { borderColor: "#718096" },
                  "&.Mui-focused fieldset": { borderColor: "#4299e1" },
                },
                "& .MuiInputBase-input": { color: "#e2e8f0", fontSize: "14px" },
                "& .MuiInputBase-input::placeholder": { color: "#a0aec0", opacity: 1 },
              }}
            />
          </Box>

          <Box sx={{ borderBottom: "1px solid #2d3748", bgcolor: "#1a202c", px: 3, py: 2 }}>
            <Box sx={{ display: "flex", justifyContent: "space-between", alignItems: "center" }}>
              <Box sx={{ display: "flex", alignItems: "center", gap: 2 }}>
                <Typography variant="body2" color="#718096" sx={{ fontSize: "13px" }}>
                  Total Collaborators
                </Typography>
                <Typography variant="body2" color="#e2e8f0" sx={{ fontSize: "13px", fontWeight: 600 }}>
                  {filteredAccounts.length}
                </Typography>
              </Box>
              <Box sx={{ display: "flex", alignItems: "center", gap: 6 }}>
                <Box sx={{ display: "flex", alignItems: "center", gap: 1 }}>
                  <Typography variant="body2" color="#718096" sx={{ fontSize: "13px" }}>
                    Accepted
                  </Typography>
                  <Typography variant="body2" color="#10b981" sx={{ fontSize: "13px", fontWeight: 600 }}>
                    {filteredAccounts.filter(a => a.status === "accepted").length}
                  </Typography>
                </Box>
                <Box sx={{ display: "flex", alignItems: "center", gap: 1 }}>
                  <Typography variant="body2" color="#718096" sx={{ fontSize: "13px" }}>
                    Pending
                  </Typography>
                  <Typography variant="body2" color="#f59e0b" sx={{ fontSize: "13px", fontWeight: 600 }}>
                    {filteredAccounts.filter(a => a.status === "pending").length}
                  </Typography>
                </Box>
                <Box sx={{ display: "flex", alignItems: "center", gap: 1 }}>
                  <Typography variant="body2" color="#718096" sx={{ fontSize: "13px" }}>
                    Expired
                  </Typography>
                  <Typography variant="body2" color="#ef4444" sx={{ fontSize: "13px", fontWeight: 600 }}>
                    {filteredAccounts.filter(a => a.status === "expired").length}
                  </Typography>
                </Box>
              </Box>
            </Box>
          </Box>

          <Box sx={{ flexGrow: 1, overflow: "auto", bgcolor: "#1a202c" }}>
            <TableContainer>
              <Table stickyHeader>
                <TableHead>
                  <TableRow>
                    <TableCell sx={{ bgcolor: "#2d3748", color: "#a0aec0", fontWeight: 600, fontSize: "13px" }}>
                      COLLABORATOR
                    </TableCell>
                    <TableCell sx={{ bgcolor: "#2d3748", color: "#a0aec0", fontWeight: 600, fontSize: "13px" }}>
                      NAME
                    </TableCell>
                    <TableCell sx={{ bgcolor: "#2d3748", color: "#a0aec0", fontWeight: 600, fontSize: "13px" }}>
                      SURNAME
                    </TableCell>
                    <TableCell sx={{ bgcolor: "#2d3748", color: "#a0aec0", fontWeight: 600, fontSize: "13px" }}>
                      STATUS
                    </TableCell>
                    <TableCell sx={{ bgcolor: "#2d3748", color: "#a0aec0", fontWeight: 600, fontSize: "13px" }}>
                      INVITED DATE
                    </TableCell>
                    <TableCell sx={{ bgcolor: "#2d3748", color: "#a0aec0", fontWeight: 600, fontSize: "13px" }}>
                      ACTIONS
                    </TableCell>
                  </TableRow>
                </TableHead>
                <TableBody>
                  {loading ? (
                    <TableRow>
                      <TableCell colSpan={6} sx={{ textAlign: 'center', py: 2 }}>
                        <Typography color="#a0aec0">Loading...</Typography>
                      </TableCell>
                    </TableRow>
                  ) : error ? (
                    <TableRow>
                      <TableCell colSpan={6} sx={{ textAlign: 'center', py: 2 }}>
                        <Typography color="#ef4444">{error}</Typography>
                      </TableCell>
                    </TableRow>
                  ) : success ? (
                    <TableRow>
                      <TableCell colSpan={6} sx={{ textAlign: 'center', py: 2 }}>
                        <Typography color="#10b981">{success}</Typography>
                      </TableCell>
                    </TableRow>
                  ) : filteredAccounts.length === 0 ? (
                    <TableRow>
                      <TableCell colSpan={6} sx={{ textAlign: 'center', py: 2 }}>
                        <Typography color="#a0aec0">No collaborators found</Typography>
                      </TableCell>
                    </TableRow>
                  ) : (
                    filteredAccounts.map((accountItem) => (
                      <TableRow key={accountItem._id} sx={{ "&:hover": { bgcolor: "#2d3748" }, bgcolor: "#1a202c" }}>
                        <TableCell sx={{ py: 2 }}>
                          <Box sx={{ display: "flex", alignItems: "center", gap: 3 }}>
                            <Avatar
                              sx={{ 
                                width: 36, 
                                height: 36, 
                                bgcolor: "#4299e1",
                                fontSize: "14px",
                                fontWeight: 600
                              }}
                            >
                              {getAvatarInitials(accountItem.name, accountItem.surname)}
                            </Avatar>
                            <Box>
                              <Typography variant="caption" color="#a0aec0" sx={{ fontSize: "12px" }}>
                                {accountItem.email}
                              </Typography>
                            </Box>
                          </Box>
                        </TableCell>
                        <TableCell>
                          <Typography variant="body2" color="#e2e8f0" sx={{ fontSize: "14px" }}>
                            {accountItem.name}
                          </Typography>
                        </TableCell>
                        <TableCell>
                          <Typography variant="body2" color="#e2e8f0" sx={{ fontSize: "14px" }}>
                            {accountItem.surname || '-'}
                          </Typography>
                        </TableCell>
                        <TableCell>
                          <Chip
                            label={
                              <Box sx={{ display: "flex", alignItems: "center", gap: 0.5 }}>
                                <span>{getStatusIcon(accountItem)}</span>
                                <span>{getStatusDisplay(accountItem)}</span>
                              </Box>
                            }
                            size="small"
                            sx={{
                              bgcolor: getStatusColor(accountItem),
                              color: "#ffffff",
                              fontSize: "11px",
                              fontWeight: 500,
                              textTransform: "capitalize",
                              "& .MuiChip-label": { px: 1 },
                            }}
                          />
                        </TableCell>
                        <TableCell>
                          <Typography variant="body2" color="#a0aec0" sx={{ fontSize: "14px" }}>
                            {accountItem.invitedDate}
                          </Typography>
                        </TableCell>
                        <TableCell>
                          {account?._id !== accountItem._id && account?.canInvite ? (
                            <IconButton
                              onClick={() => handleOpenDeleteDialog(accountItem)}
                              sx={{ color: "#ef4444" }}
                            >
                              <DeleteIcon />
                            </IconButton>
                          ) : null}
                        </TableCell>
                      </TableRow>
                    ))
                  )}
                </TableBody>
              </Table>
            </TableContainer>
          </Box>
        </Box>
      </Box>

      <Dialog
        open={openInviteDialog}
        onClose={handleCloseInviteDialog}
        maxWidth="sm"
        fullWidth
        PaperProps={{ sx: { bgcolor: "#1a202c", border: "1px solid #2d3748" } }}
      >
        <DialogTitle sx={{ color: "#e2e8f0", borderBottom: "1px solid #2d3748", pb: 2 }}>
          <Typography variant="h6" sx={{ fontWeight: 600 }}>
            Invite New Collaborator
          </Typography>
          <Typography variant="body2" color="#a0aec0" sx={{ mt: 1 }}>
            Enter the email address of the person you want to invite to your project
          </Typography>
        </DialogTitle>
        <DialogContent sx={{ pt: 3 }}>
          <TextField
            label="Email Address"
            type="email"
            value={inviteEmail}
            onChange={(e) => setInviteEmail(e.target.value)}
            fullWidth
            placeholder="colleague@company.com"
            sx={{
              "& .MuiOutlinedInput-root": {
                bgcolor: "#2d3748",
                "& fieldset": { borderColor: "#4a5568" },
                "&:hover fieldset": { borderColor: "#718096" },
                "&.Mui-focused fieldset": { borderColor: "#4299e1" },
              },
              "& .MuiInputBase-input": { color: "#e2e8f0" },
              "& .MuiInputLabel-root": { color: "#a0aec0" },
            }}
          />
        </DialogContent>
        <DialogActions sx={{ borderTop: "1px solid #2d3748", pt: 2, px: 3, pb: 3 }}>
          <Button onClick={handleCloseInviteDialog} sx={{ color: "#a0aec0", textTransform: "none" }}>
            Cancel
          </Button>
          <Button
            onClick={handleInvite}
            variant="contained"
            disabled={!inviteEmail.trim() || !account?.canInvite || (account?.invitedBy && account?.invitedBy !== account?._id) || !account?._id}
            sx={{
              bgcolor: "#4299e1",
              "&:hover": { bgcolor: "#3182ce" },
              "&:disabled": { bgcolor: "#4a5568", color: "#a0aec0" },
              textTransform: "none",
              px: 3,
            }}
          >
            Send Invitation
          </Button>
        </DialogActions>
      </Dialog>

      <Dialog
        open={openDeleteDialog}
        onClose={handleCloseDeleteDialog}
        maxWidth="sm"
        fullWidth
        PaperProps={{ sx: { bgcolor: "#1a202c", border: "1px solid #2d3748" } }}
      >
        <DialogTitle sx={{ color: "#e2e8f0", borderBottom: "1px solid #2d3748", pb: 2 }}>
          <Typography variant="h6" sx={{ fontWeight: 600 }}>
            Confirmer la suppression
          </Typography>
          <Typography variant="body2" color="#a0aec0" sx={{ mt: 1 }}>
            Voulez-vous vraiment supprimer le collaborateur {accountToDelete?.email} ?
          </Typography>
        </DialogTitle>
        <DialogActions sx={{ borderTop: "1px solid #2d3748", pt: 2, px: 3, pb: 3 }}>
          <Button onClick={handleCloseDeleteDialog} sx={{ color: "#a0aec0", textTransform: "none" }}>
            Annuler
          </Button>
          <Button
            onClick={handleDelete}
            variant="contained"
            sx={{
              bgcolor: "#ef4444",
              "&:hover": { bgcolor: "#dc2626" },
              textTransform: "none",
              px: 3,
            }}
          >
            Supprimer
          </Button>
        </DialogActions>
      </Dialog>
    </ThemeProvider>
  );
}