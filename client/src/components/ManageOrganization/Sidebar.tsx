"use client";

import React from "react";
import { Box, Typography, List, ListItem, ListItemIcon, ListItemText } from "@mui/material";
import { Settings, People, Feedback } from "@mui/icons-material";

interface MenuItem {
  id: string;
  label: string;
  icon: React.ReactNode;
}

const menuItems: MenuItem[] = [
  { id: "settings", label: "Settings", icon: <Settings sx={{ color: "#4299e1" }} /> },
  { id: "members", label: "Members", icon: <People sx={{ color: "#4299e1" }} /> },
  { id: "feedback", label: "Feedback", icon: <Feedback sx={{ color: "#4299e1" }} /> },
];

interface SidebarProps {
  activeMenu: string;
  setActiveMenu: (id: string) => void;
}

export default function Sidebar({ activeMenu, setActiveMenu }: SidebarProps) {
  return (
    <Box
      sx={{
        width: { xs: "100%", sm: 280 },
        bgcolor: "#1a202c",
        borderRight: { sm: "1px solid #2d3748" },
        borderBottom: { xs: "1px solid #2d3748", sm: "none" },
        display: "flex",
        flexDirection: "column",
      }}
    >
      <Box sx={{ p: 3, borderBottom: "1px solid #2d3748" }}>
        <Typography variant="h6" sx={{ color: "#e2e8f0", fontWeight: 600 }}>
          Organization Management
        </Typography>
      </Box>

      <List sx={{ flexGrow: 1, px: 1, py: 2 }}>
        {menuItems.map((item) => (
          <ListItem
            component="button"
            key={item.id}
            onClick={() => setActiveMenu(item.id)}
            sx={{
              borderRadius: 2,
              mb: 1,
              bgcolor: activeMenu === item.id ? "rgba(66, 153, 225, 0.1)" : "transparent",
              "&:hover": {
                bgcolor: activeMenu === item.id ? "rgba(66, 153, 225, 0.2)" : "rgba(255, 255, 255, 0.05)",
              },
            }}
          >
            <ListItemIcon sx={{ minWidth: 40 }}>{item.icon}</ListItemIcon>
            <ListItemText
              primary={item.label}
              primaryTypographyProps={{
                color: activeMenu === item.id ? "#4299e1" : "#e2e8f0",
                fontWeight: activeMenu === item.id ? 600 : 400,
                fontSize: "14px",
              }}
            />
          </ListItem>
        ))}
      </List>
    </Box>
  );
}