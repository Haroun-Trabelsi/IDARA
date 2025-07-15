"use client"

import React, { useState, MouseEventHandler, useEffect } from "react";
import { 
  AppBar, 
  Toolbar, 
  Button, 
  IconButton, 
  Box, 
  Menu, 
  MenuItem, 
  Avatar, 
  Popover, 
  List, 
  ListItemButton,
  Dialog,
  DialogTitle,
  DialogContent,
  TextField,
  Typography,
  Card,
  CardContent,
  Divider
} from "@mui/material";
import { 
  KeyboardArrowDown, 
  Bookmark, 
  Notifications, 
  ViewList, 
  Edit, 
  Logout,
  Search,
  Settings as SettingsIcon,
  Business,
  Star,
  Language as LanguageIcon,
  Brightness4,
  Help
} from "@mui/icons-material";
import { useAuth } from 'contexts/AuthContext';
import { useModalStore } from 'store/useModalStore';
import OnlineIndicator from 'components/OnlineIndicator';
import { useNavigate } from 'react-router-dom';

// Créer un composant Settings local
const Settings: React.FC = () => {
  const [settingsAnchorEl, setSettingsAnchorEl] = useState<null | HTMLElement>(null);
  const navigate = useNavigate();

  const openSettingsMenu = (e: React.MouseEvent<HTMLButtonElement>) => {
    setSettingsAnchorEl(e.currentTarget);
  };

  const closeSettingsMenu = () => {
    setSettingsAnchorEl(null);
  };

  const handleCollaboratorClick = () => {
    navigate('/collaborator');
    closeSettingsMenu();
  };

  return (
    <>
      <IconButton 
        size="small" 
        sx={{ color: "#a0aec0", "&:hover": { bgcolor: "rgba(255, 255, 255, 0.08)", color: "#e2e8f0" } }} 
        onClick={openSettingsMenu}
      >
        <SettingsIcon sx={{ fontSize: "18px" }} />
      </IconButton>
      <Menu
        anchorEl={settingsAnchorEl}
        open={Boolean(settingsAnchorEl)}
        onClose={closeSettingsMenu}
        PaperProps={{ 
          sx: { 
            bgcolor: "#2d3748", 
            border: "1px solid #4a5568",
            minWidth: 180
          } 
        }}
      >
        <MenuItem sx={{ color: "#e2e8f0" }} onClick={handleCollaboratorClick}>
          <SettingsIcon sx={{ fontSize: "16px", marginRight: 1 }} />
          Collaborator
        </MenuItem>
      </Menu>
    </>
  );
};

export default function Header() {
  const { isLoggedIn, account, logout, token } = useAuth();
  const { setCurrentModal } = useModalStore();
  const navigate = useNavigate();

  const [projectsAnchorEl, setProjectsAnchorEl] = useState<null | HTMLElement>(null);
  const [bookmarksAnchorEl, setBookmarksAnchorEl] = useState<null | HTMLElement>(null);
  const [anchorEl, setAnchorEl] = useState<null | HTMLElement>(null);
  const [popover, setPopover] = useState(false);
  const [projects, setProjects] = useState<string[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  // États pour les modales
  const [accountSecurityOpen, setAccountSecurityOpen] = useState(false);
  const [editProfileOpen, setEditProfileOpen] = useState(false);

  // États pour les formulaires
  const [profileData, setProfileData] = useState({
    name: account?.name || 'John',
    surname: account?.surname || 'Doe',
    email: account?.email || 'ines.dahmani@esprit.tn'
  });
  
  const [passwordData, setPasswordData] = useState({
    newPassword: '',
    confirmPassword: ''
  });

  // États pour MFA
  const [mfaQrCode, setMfaQrCode] = useState<string | null>(null);
  const [mfaSecret, setMfaSecret] = useState<string | null>(null);
  const [mfaCode, setMfaCode] = useState('');
  const [mfaEnabled, setMfaEnabled] = useState(!!account?.mfaEnabled);
  const [mfaError, setMfaError] = useState<string | null>(null);

  useEffect(() => {
    const fetchProjects = async () => {
      try {
        const response = await fetch("http://localhost:8080/api/projects");
        if (!response.ok) {
          throw new Error(`Failed to fetch projects: ${response.status}`);
        }
        const data = await response.json();
        setProjects(data);
      } catch (err) {
        console.error("❌ Fetch error:", err);
        setError(err instanceof Error ? err.message : 'Failed to fetch projects');
      } finally {
        setLoading(false);
      }
    };

    fetchProjects();
  }, []);

  useEffect(() => {
    setMfaEnabled(!!account?.mfaEnabled);
    setProfileData({
      name: account?.name || 'John',
      surname: account?.surname || 'Doe',
      email: account?.email || 'ines.dahmani@esprit.tn'
    });
  }, [account]);

  const openPopover: MouseEventHandler<HTMLButtonElement> = (e) => {
    setPopover(true);
    setAnchorEl(e.currentTarget);
  };

  const closePopover = () => {
    setPopover(false);
    setAnchorEl(null);
  };

  const clickLogin = () => {
    setCurrentModal('LOGIN');
    closePopover();
  };

  const clickRegister = () => {
    setCurrentModal('REGISTER');
    closePopover();
  };

  const handleEditProfile = () => {
    navigate('/Account');
    closePopover();
  };

  const handleLogout = () => {
    logout();
    closePopover();
  };

  const handleSaveProfile = () => {
    console.log('Profile saved:', profileData);
    setEditProfileOpen(false);
  };

  const handleChangePassword = () => {
    if (passwordData.newPassword !== passwordData.confirmPassword) {
      setMfaError('Passwords do not match');
      return;
    }
    console.log('Password changed:', passwordData);
    setPasswordData({ newPassword: '', confirmPassword: '' });
  };

  const handleEnableMFA = async () => {
    if (!token) {
      setMfaError('You must be logged in to enable MFA');
      return;
    }
    try {
      const response = await fetch('http://localhost:8080/auth/mfa/setup', {
        method: 'POST',
        headers: { 
          'Content-Type': 'application/json',
          'Authorization': `Bearer ${token}`
        },
        body: JSON.stringify({ email: profileData.email })
      });
      if (!response.ok) {
        throw new Error(`Failed to setup MFA: ${response.status}`);
      }
      const data = await response.json();
      setMfaQrCode(data.qrCode);
      setMfaSecret(data.secret);
      setMfaError(null);
    } catch (err) {
      console.error('❌ MFA setup error:', err);
      setMfaError(err instanceof Error ? err.message : 'Failed to setup MFA');
    }
  };

  const handleVerifyMFA = async () => {
    if (!token) {
      setMfaError('You must be logged in to verify MFA');
      return;
    }
    try {
      const response = await fetch('http://localhost:8080/auth/mfa/verify', {
        method: 'POST',
        headers: { 
          'Content-Type': 'application/json',
          'Authorization': `Bearer ${token}`
        },
        body: JSON.stringify({ email: profileData.email, code: mfaCode, secret: mfaSecret })
      });
      if (!response.ok) {
        throw new Error(`Failed to verify MFA: ${response.status}`);
      }
      const data = await response.json();
      setMfaEnabled(true);
      setMfaQrCode(null);
      setMfaSecret(null);
      setMfaCode('');
      setMfaError(data.message || 'MFA enabled successfully!');
      setTimeout(() => setMfaError(null), 3000);
    } catch (err) {
      console.error('❌ MFA verify error:', err);
      setMfaError(err instanceof Error ? err.message : 'Invalid MFA code');
    }
  };

  return (
    <>
      <AppBar position="static" elevation={0} sx={{ bgcolor: "#2d3748", borderBottom: "1px solid #4a5568" }}>
        <Toolbar sx={{ justifyContent: "space-between", minHeight: "48px !important", px: 2 }}>
          {/* Left Navigation */}
          <Box sx={{ display: "flex", alignItems: "center", gap: 3 }}>
            <Button 
              endIcon={<KeyboardArrowDown sx={{ fontSize: "16px" }} />} 
              onClick={(e) => setProjectsAnchorEl(e.currentTarget)} 
              sx={{ 
                color: "#e2e8f0", 
                textTransform: "none", 
                fontSize: "14px", 
                fontWeight: 400, 
                "&:hover": { bgcolor: "rgba(255, 255, 255, 0.08)" } 
              }}
            >
              Projects
            </Button>
            <Menu 
              anchorEl={projectsAnchorEl} 
              open={Boolean(projectsAnchorEl)} 
              onClose={() => setProjectsAnchorEl(null)} 
              PaperProps={{ 
                sx: { 
                  bgcolor: "#2d3748", 
                  border: "1px solid #4a5568",
                  maxHeight: 300,
                  width: 200
                } 
              }}
            >
              {loading ? (
                <MenuItem sx={{ color: "#e2e8f0" }}>Loading projects...</MenuItem>
              ) : error ? (
                <MenuItem sx={{ color: "#e2e8f0" }}>Error loading projects</MenuItem>
              ) : projects.length === 0 ? (
                <MenuItem sx={{ color: "#e2e8f0" }}>No projects found</MenuItem>
              ) : (
                projects.map((project, index) => (
                  <MenuItem 
                    key={index} 
                    sx={{ color: "#e2e8f0" }}
                    onClick={() => {
                      setProjectsAnchorEl(null);
                    }}
                  >
                    {project}
                  </MenuItem>
                ))
              )}
            </Menu>

            <Button sx={{ color: "#e2e8f0", textTransform: "none", fontSize: "14px", fontWeight: 400, "&:hover": { bgcolor: "rgba(255, 255, 255, 0.08)" } }}>
              My Tasks
            </Button>

            <Button startIcon={<Bookmark sx={{ fontSize: "16px" }} />} endIcon={<KeyboardArrowDown sx={{ fontSize: "16px" }} />} onClick={(e) => setBookmarksAnchorEl(e.currentTarget)} sx={{ color: "#e2e8f0", textTransform: "none", fontSize: "14px", fontWeight: 400, "&:hover": { bgcolor: "rgba(255, 255, 255, 0.08)" } }}>
              Bookmarks
            </Button>
            <Menu anchorEl={bookmarksAnchorEl} open={Boolean(bookmarksAnchorEl)} onClose={() => setBookmarksAnchorEl(null)} PaperProps={{ sx: { bgcolor: "#2d3748", border: "1px solid #4a5568" } }}>
              <MenuItem sx={{ color: "#e2e8f0" }}>Bookmark 1</MenuItem>
              <MenuItem sx={{ color: "#e2e8f0" }}>Bookmark 2</MenuItem>
            </Menu>
          </Box>

          {/* Right Icons */}
          <Box sx={{ display: "flex", alignItems: "center", gap: 1 }}>
            <IconButton size="small" sx={{ color: "#a0aec0", "&:hover": { bgcolor: "rgba(255, 255, 255, 0.08)", color: "#e2e8f0" } }}>
              <Notifications sx={{ fontSize: "18px" }} />
            </IconButton>

            <IconButton size="small" sx={{ color: "#a0aec0", "&:hover": { bgcolor: "rgba(255, 255, 255, 0.08)", color: "#e2e8f0" } }}>
              <ViewList sx={{ fontSize: "18px" }} />
            </IconButton>

            <IconButton size="small" sx={{ color: "#a0aec0", "&:hover": { bgcolor: "rgba(255, 255, 255, 0.08)", color: "#e2e8f0" } }}>
              <Search sx={{ fontSize: "18px" }} />
            </IconButton>

            <Settings />

            <IconButton onClick={openPopover}>
              <OnlineIndicator online={isLoggedIn}>
                <Avatar sx={{ width: 28, height: 28, bgcolor: "#4299e1", fontSize: "12px", marginLeft: 1 }} src={account?.name || ''} alt={account?.name || 'Guest'}>
                  {account?.name ? account.name[0].toUpperCase() : 'G'}
                </Avatar>
              </OnlineIndicator>
            </IconButton>

            <Popover
              anchorEl={anchorEl}
              open={popover}
              onClose={closePopover}
              anchorOrigin={{ vertical: "bottom", horizontal: "right" }}
              transformOrigin={{ vertical: "top", horizontal: "right" }}
              PaperProps={{
                sx: {
                  bgcolor: "#1a202c",
                  border: "1px solid #2d3748",
                  borderRadius: "8px",
                  minWidth: "280px",
                  boxShadow: "0 10px 25px rgba(0, 0, 0, 0.5)"
                }
              }}
            >
              <Box sx={{ p: 0 }}>
                {isLoggedIn ? (
                  <>
                    {/* Section Header - Centrée */}
                    <Box sx={{ 
                      display: 'flex', 
                      flexDirection: 'column',
                      alignItems: 'center', 
                      gap: 1, 
                      py: 3,
                      px: 2,
                      borderBottom: "1px solid #2d3748"
                    }}>
                      <Avatar 
                        sx={{ 
                          width: 48, 
                          height: 48, 
                          bgcolor: "#4299e1", 
                          fontSize: "18px"
                        }} 
                        alt={account?.name || 'Guest'}
                      >
                        {account?.name ? account.name[0].toUpperCase() : 'G'}
                      </Avatar>
                      <Typography variant="body1" sx={{ 
                        fontWeight: 600, 
                        color: "#e2e8f0",
                        fontSize: "16px"
                      }}>
                        {`${account?.name || "John"} ${account?.surname || "Doe"}`}
                      </Typography>
                      <Typography variant="body2" sx={{ 
                        color: "#a0aec0",
                        fontSize: "14px"
                      }}>
                        {account?.email || "ines.dahmani@esprit.tn"}
                      </Typography>
                      <Button
                        variant="text"
                        onClick={handleEditProfile}
                        sx={{ 
                          color: "#4299e1",
                          textTransform: "none",
                          fontSize: "14px",
                          fontWeight: 500,
                          "&:hover": { 
                            bgcolor: "rgba(66, 153, 225, 0.1)"
                          }
                        }}
                      >
                        Account
                      </Button>
                    </Box>

                    {/* Menu Items - Non centrés */}
                    <List sx={{ py: 1 }}>
                      <ListItemButton 
                        onClick={() => navigate('/organizations')}
                        sx={{ 
                          color: "#e2e8f0",
                          "&:hover": { bgcolor: "rgba(255, 255, 255, 0.08)" },
                          py: 1.5
                        }}
                      >
                        <Business sx={{ fontSize: "20px", marginRight: 2, color: "#a0aec0" }} />
                        <Typography variant="body2" sx={{ fontSize: "14px" }}>
                          Manage organization
                        </Typography>
                      </ListItemButton>

                      <ListItemButton 
                        onClick={() => console.log('Upgrade to Review pro')}
                        sx={{ 
                          color: "#e2e8f0",
                          "&:hover": { bgcolor: "rgba(255, 255, 255, 0.08)" },
                          py: 1.5
                        }}
                      >
                        <Star sx={{ fontSize: "20px", marginRight: 2, color: "#a0aec0" }} />
                        <Typography variant="body2" sx={{ fontSize: "14px" }}>
                          Upgrade to Review pro
                        </Typography>
                      </ListItemButton>

                      <ListItemButton 
                        onClick={() => console.log('Language')}
                        sx={{ 
                          color: "#e2e8f0",
                          "&:hover": { bgcolor: "rgba(255, 255, 255, 0.08)" },
                          py: 1.5
                        }}
                      >
                        <LanguageIcon sx={{ fontSize: "20px", marginRight: 2, color: "#a0aec0" }} />
                        <Typography variant="body2" sx={{ fontSize: "14px" }}>
                          Language
                        </Typography>
                      </ListItemButton>

                      <ListItemButton 
                        onClick={() => console.log('Switch theme')}
                        sx={{ 
                          color: "#e2e8f0",
                          "&:hover": { bgcolor: "rgba(255, 255, 255, 0.08)" },
                          py: 1.5
                        }}
                      >
                        <Brightness4 sx={{ fontSize: "20px", marginRight: 2, color: "#a0aec0" }} />
                        <Typography variant="body2" sx={{ fontSize: "14px" }}>
                          Switch theme
                        </Typography>
                      </ListItemButton>

                      <ListItemButton 
                        onClick={() => console.log('Help and support')}
                        sx={{ 
                          color: "#e2e8f0",
                          "&:hover": { bgcolor: "rgba(255, 255, 255, 0.08)" },
                          py: 1.5
                        }}
                      >
                        <Help sx={{ fontSize: "20px", marginRight: 2, color: "#a0aec0" }} />
                        <Typography variant="body2" sx={{ fontSize: "14px" }}>
                          Help and support
                        </Typography>
                      </ListItemButton>

                      <Divider sx={{ bgcolor: "#2d3748", my: 1 }} />

                      <ListItemButton 
                        onClick={handleLogout}
                        sx={{ 
                          color: "#e2e8f0",
                          "&:hover": { bgcolor: "rgba(255, 255, 255, 0.08)" },
                          py: 1.5
                        }}
                      >
                        <Logout sx={{ fontSize: "20px", marginRight: 2, color: "#a0aec0" }} />
                        <Typography variant="body2" sx={{ fontSize: "14px" }}>
                          Sign out
                        </Typography>
                      </ListItemButton>
                    </List>
                  </>
                ) : (
                  <List sx={{ py: 1 }}>
                    <ListItemButton 
                      onClick={clickLogin}
                      sx={{ 
                        color: "#e2e8f0",
                        "&:hover": { bgcolor: "rgba(255, 255, 255, 0.08)" }
                      }}
                    >
                      Login
                    </ListItemButton>
                    <ListItemButton 
                      onClick={clickRegister}
                      sx={{ 
                        color: "#e2e8f0",
                        "&:hover": { bgcolor: "rgba(255, 255, 255, 0.08)" }
                      }}
                    >
                      Register
                    </ListItemButton>
                  </List>
                )}
              </Box>
            </Popover>
          </Box>
        </Toolbar>
      </AppBar>

      {/* Modal Account Security */}
      <Dialog 
        open={accountSecurityOpen} 
        onClose={() => {
          setAccountSecurityOpen(false);
          setMfaQrCode(null);
          setMfaSecret(null);
          setMfaCode('');
          setMfaError(null);
        }}
        maxWidth="sm"
        fullWidth
      >
        <DialogTitle sx={{ textAlign: 'center', bgcolor: '#f7fafc', color: '#4a5568', fontWeight: 600 }}>
          Account Security
        </DialogTitle>
        <DialogContent sx={{ bgcolor: '#f7fafc', p: 3 }}>
          <Typography variant="body2" sx={{ textAlign: 'center', color: '#718096', mb: 3 }}>
            Edit your account settings and change your password here.
          </Typography>

          {/* Email Section */}
          <Box sx={{ mb: 3 }}>
            <Typography variant="subtitle1" sx={{ mb: 1, fontWeight: 600, color: '#4a5568' }}>
              Email:
            </Typography>
            <Box sx={{ display: 'flex', gap: 1 }}>
              <TextField
                fullWidth
                variant="outlined"
                value={`Your email address is ${profileData.email}`}
                disabled
                sx={{ 
                  '& .MuiOutlinedInput-root': {
                    bgcolor: 'white',
                    '& .Mui-disabled': {
                      color: '#4a5568'
                    }
                  }
                }}
              />
              <IconButton 
                sx={{ 
                  bgcolor: '#4299e1', 
                  color: 'white',
                  '&:hover': { bgcolor: '#3182ce' }
                }}
              >
                <Edit />
              </IconButton>
            </Box>
          </Box>

          {/* Password Section */}
          <Box sx={{ mb: 3 }}>
            <Typography variant="subtitle1" sx={{ mb: 1, fontWeight: 600, color: '#4a5568' }}>
              New password
            </Typography>
            <TextField
              fullWidth
              type="password"
              placeholder="Enter new password"
              variant="outlined"
              value={passwordData.newPassword}
              onChange={(e) => setPasswordData(prev => ({ ...prev, newPassword: e.target.value }))}
              sx={{ 
                mb: 2,
                '& .MuiOutlinedInput-root': { bgcolor: 'white' }
              }}
            />
            
            <Typography variant="subtitle1" sx={{ mb: 1, fontWeight: 600, color: '#4a5568' }}>
              Confirm new password
            </Typography>
            <TextField
              fullWidth
              type="password"
              placeholder="Re-type new password"
              variant="outlined"
              value={passwordData.confirmPassword}
              onChange={(e) => setPasswordData(prev => ({ ...prev, confirmPassword: e.target.value }))}
              sx={{ 
                mb: 2,
                '& .MuiOutlinedInput-root': { bgcolor: 'white' }
              }}
            />
            
            <Button 
              variant="contained"
              onClick={handleChangePassword}
              sx={{ 
                bgcolor: '#4299e1',
                '&:hover': { bgcolor: '#3182ce' },
                textTransform: 'none',
                px: 3
              }}
            >
              Change password
            </Button>
          </Box>

          {/* MFA Section */}
          <Card sx={{ bgcolor: 'white', border: '1px solid #e2e8f0' }}>
            <CardContent>
              <Typography variant="h6" sx={{ mb: 1, fontWeight: 600, color: '#4a5568' }}>
                Multi-factor Authentication
              </Typography>
              <Typography variant="body2" sx={{ mb: 2, color: '#718096' }}>
                Increase your account security by requiring that a code from the Microsoft Authenticator app be entered when you log in. For more information, refer to our{' '}
                <Typography component="span" sx={{ color: '#4299e1', textDecoration: 'underline', cursor: 'pointer' }}>
                  Help Center article
                </Typography>
                .
              </Typography>

              {mfaEnabled ? (
                <Typography variant="body2" sx={{ color: '#48bb78', textAlign: 'center' }}>
                  MFA is enabled for your account!
                </Typography>
              ) : mfaQrCode ? (
                <>
                  <Typography variant="body2" sx={{ mb: 2, color: '#718096' }}>
                    Scan this QR code with the Microsoft Authenticator app on your phone:
                  </Typography>
                  <Box sx={{ textAlign: 'center', mb: 2 }}>
                    <img src={mfaQrCode} alt="MFA QR Code" style={{ maxWidth: '100%', height: 'auto' }} />
                  </Box>
                  <TextField
                    fullWidth
                    label="Enter code from Authenticator"
                    variant="outlined"
                    value={mfaCode}
                    onChange={(e) => setMfaCode(e.target.value)}
                    sx={{ mb: 2, '& .MuiOutlinedInput-root': { bgcolor: 'white' } }}
                  />
                  <Button 
                    variant="contained"
                    onClick={handleVerifyMFA}
                    sx={{ 
                      bgcolor: '#4299e1',
                      '&:hover': { bgcolor: '#3182ce' },
                      textTransform: 'none'
                    }}
                  >
                    Verify and Enable
                  </Button>
                </>
              ) : (
                <Button 
                  variant="contained"
                  onClick={handleEnableMFA}
                  sx={{ 
                    bgcolor: '#4299e1',
                    '&:hover': { bgcolor: '#3182ce' },
                    textTransform: 'none'
                  }}
                >
                  Enable
                </Button>
              )}

              {mfaError && (
                <Typography variant="body2" sx={{ color: '#e53e3e', textAlign: 'center', mt: 1 }}>
                  {mfaError}
                </Typography>
              )}
            </CardContent>
          </Card>
        </DialogContent>
      </Dialog>

      {/* Modal Edit Profile - Simplified */}
      <Dialog 
        open={editProfileOpen} 
        onClose={() => setEditProfileOpen(false)}
        maxWidth="sm"
        fullWidth
      >
        <DialogTitle sx={{ textAlign: 'center', bgcolor: '#f7fafc', color: '#4a5568', fontWeight: 600 }}>
          Edit Profile
        </DialogTitle>
        <DialogContent sx={{ bgcolor: '#f7fafc', p: 3 }}>
          <Typography variant="body2" sx={{ textAlign: 'center', color: '#718096', mb: 3 }}>
            Update your personal information.
          </Typography>

          {/* Name Fields */}
          <Box sx={{ mb: 3 }}>
            <Typography variant="subtitle1" sx={{ mb: 1, fontWeight: 600, color: '#4a5568' }}>
              First Name:
            </Typography>
            <TextField
              fullWidth
              variant="outlined"
              placeholder="Enter your first name"
              value={profileData.name}
              onChange={(e) => setProfileData(prev => ({ ...prev, name: e.target.value }))}
              sx={{ 
                mb: 2,
                '& .MuiOutlinedInput-root': { bgcolor: 'white' }
              }}
            />
            
            <Typography variant="subtitle1" sx={{ mb: 1, fontWeight: 600, color: '#4a5568' }}>
              Last Name:
            </Typography>
            <TextField
              fullWidth
              variant="outlined"
              placeholder="Enter your last name"
              value={profileData.surname}
              onChange={(e) => setProfileData(prev => ({ ...prev, surname: e.target.value }))}
              sx={{ 
                mb: 3,
                '& .MuiOutlinedInput-root': { bgcolor: 'white' }
              }}
            />
            
            <Box sx={{ display: 'flex', gap: 2, justifyContent: 'center' }}>
              <Button 
                variant="outlined"
                onClick={() => setEditProfileOpen(false)}
                sx={{ 
                  borderColor: '#e2e8f0',
                  color: '#4a5568',
                  '&:hover': { 
                    borderColor: '#cbd5e0',
                    bgcolor: 'rgba(0, 0, 0, 0.04)'
                  },
                  textTransform: 'none',
                  px: 3
                }}
              >
                Cancel
              </Button>
              <Button 
                variant="contained"
                onClick={handleSaveProfile}
                sx={{ 
                  bgcolor: '#4299e1',
                  '&:hover': { bgcolor: '#3182ce' },
                  textTransform: 'none',
                  px: 3
                }}
              >
                Save Changes
              </Button>
            </Box>
          </Box>
        </DialogContent>
      </Dialog>
    </>
  );
}