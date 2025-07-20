"use client";

import React, { useState, useEffect } from "react";
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
  Typography
} from "@mui/material";
import { 
  KeyboardArrowDown, 
  Bookmark, 
  Notifications, 
  ViewList, 
  Search,
  Business,
  Star,
  Language as LanguageIcon,
  Brightness4,
  Help,
  Logout
} from "@mui/icons-material";
import { useAuth } from 'contexts/AuthContext';
import { useNavigate } from 'react-router-dom';
import OnlineIndicator from 'components/OnlineIndicator';

export default function Header() {
  const { isLoggedIn, account, logout } = useAuth();
  const navigate = useNavigate();

  const [projectsAnchorEl, setProjectsAnchorEl] = useState<null | HTMLElement>(null);
  const [bookmarksAnchorEl, setBookmarksAnchorEl] = useState<null | HTMLElement>(null);
  const [anchorEl, setAnchorEl] = useState<null | HTMLElement>(null);
  const [popover, setPopover] = useState(false);
  const [projects, setProjects] = useState<string[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

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

  const openPopover = (e: React.MouseEvent<HTMLButtonElement>) => {
    setPopover(true);
    setAnchorEl(e.currentTarget);
  };

  const closePopover = () => {
    setPopover(false);
    setAnchorEl(null);
  };

  const handleLogout = () => {
    logout();
    closePopover();
  };

  return (
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
                  onClick={() => setProjectsAnchorEl(null)}
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
                <List sx={{ py: 1 }}>
                  <Box sx={{ display: 'flex', flexDirection: 'column', alignItems: 'center', gap: 1, py: 3, px: 2, borderBottom: "1px solid #2d3748" }}>
                    <Avatar 
                      sx={{ width: 48, height: 48, bgcolor: "#4299e1", fontSize: "18px" }} 
                      alt={account?.name || 'Guest'}
                    >
                      {account?.name ? account.name[0].toUpperCase() : 'G'}
                    </Avatar>
                    <Typography variant="body1" sx={{ fontWeight: 600, color: "#e2e8f0", fontSize: "16px" }}>
                      {`${account?.name || "John"} ${account?.surname || "Doe"}`}
                    </Typography>
                    <Typography variant="body2" sx={{ color: "#a0aec0", fontSize: "14px" }}>
                      {account?.email || "ines.dahmani@esprit.tn"}
                    </Typography>
                    <Button
                      variant="text"
                      onClick={() => navigate('/account')}
                      sx={{ 
                        color: "#4299e1",
                        textTransform: "none",
                        fontSize: "14px",
                        fontWeight: 500,
                        "&:hover": { bgcolor: "rgba(66, 153, 225, 0.1)" }
                      }}
                    >
                      Account
                    </Button>
                  </Box>

                  <ListItemButton 
                    onClick={() => navigate('/organizations')}
                    sx={{ color: "#e2e8f0", "&:hover": { bgcolor: "rgba(255, 255, 255, 0.08)" }, py: 1.5 }}
                  >
                    <Business sx={{ fontSize: "20px", marginRight: 2, color: "#a0aec0" }} />
                    <Typography variant="body2" sx={{ fontSize: "14px" }}>
                      Manage organization
                    </Typography>
                  </ListItemButton>

                  <ListItemButton 
                    onClick={() => console.log('Upgrade to Review pro')}
                    sx={{ color: "#e2e8f0", "&:hover": { bgcolor: "rgba(255, 255, 255, 0.08)" }, py: 1.5 }}
                  >
                    <Star sx={{ fontSize: "20px", marginRight: 2, color: "#a0aec0" }} />
                    <Typography variant="body2" sx={{ fontSize: "14px" }}>
                      Upgrade to Review pro
                    </Typography>
                  </ListItemButton>

                  <ListItemButton 
                    onClick={() => console.log('Language')}
                    sx={{ color: "#e2e8f0", "&:hover": { bgcolor: "rgba(255, 255, 255, 0.08)" }, py: 1.5 }}
                  >
                    <LanguageIcon sx={{ fontSize: "20px", marginRight: 2, color: "#a0aec0" }} />
                    <Typography variant="body2" sx={{ fontSize: "14px" }}>
                      Language
                    </Typography>
                  </ListItemButton>

                  <ListItemButton 
                    onClick={() => console.log('Switch theme')}
                    sx={{ color: "#e2e8f0", "&:hover": { bgcolor: "rgba(255, 255, 255, 0.08)" }, py: 1.5 }}
                  >
                    <Brightness4 sx={{ fontSize: "20px", marginRight: 2, color: "#a0aec0" }} />
                    <Typography variant="body2" sx={{ fontSize: "14px" }}>
                      Switch theme
                    </Typography>
                  </ListItemButton>

                  <ListItemButton 
                    onClick={() => console.log('Help and support')}
                    sx={{ color: "#e2e8f0", "&:hover": { bgcolor: "rgba(255, 255, 255, 0.08)" }, py: 1.5 }}
                  >
                    <Help sx={{ fontSize: "20px", marginRight: 2, color: "#a0aec0" }} />
                    <Typography variant="body2" sx={{ fontSize: "14px" }}>
                      Help and support
                    </Typography>
                  </ListItemButton>

                  <ListItemButton 
                    onClick={handleLogout}
                    sx={{ color: "#e2e8f0", "&:hover": { bgcolor: "rgba(255, 255, 255, 0.08)" }, py: 1.5 }}
                  >
                    <Logout sx={{ fontSize: "20px", marginRight: 2, color: "#a0aec0" }} />
                    <Typography variant="body2" sx={{ fontSize: "14px" }}>
                      Sign out
                    </Typography>
                  </ListItemButton>
                </List>
              ) : (
                <List sx={{ py: 1 }}>
                  <ListItemButton 
                    onClick={() => navigate('/login')}
                    sx={{ color: "#e2e8f0", "&:hover": { bgcolor: "rgba(255, 255, 255, 0.08)" } }}
                  >
                    Login
                  </ListItemButton>
                  <ListItemButton 
                    onClick={() => navigate('/register')}
                    sx={{ color: "#e2e8f0", "&:hover": { bgcolor: "rgba(255, 255, 255, 0.08)" } }}
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
  );
}