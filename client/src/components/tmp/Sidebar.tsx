"use client"

import { Drawer, Box, Typography, List, ListItem, ListItemIcon, ListItemText, LinearProgress } from "@mui/material"
import { Home } from "@mui/icons-material"
import React from "react"

interface SidebarItem {
  id: string
  progress: number
  color: string
}

const sidebarItems: SidebarItem[] = [
  { id: "129", progress: 85, color: "#f59e0b" },
  { id: "131", progress: 90, color: "#f59e0b" },
  { id: "132", progress: 75, color: "#f59e0b" },
  { id: "134", progress: 60, color: "#f59e0b" },
  { id: "139", progress: 45, color: "#f59e0b" },
  { id: "146", progress: 30, color: "#f59e0b" },
  { id: "149", progress: 95, color: "#ef4444" },
  { id: "156", progress: 80, color: "#f59e0b" },
  { id: "159", progress: 70, color: "#ef4444" },
  { id: "160", progress: 55, color: "#f59e0b" },
  { id: "164", progress: 40, color: "#ef4444" },
]

export default function Sidebar() {
  return (
    <Drawer
      variant="permanent"
      sx={{
        width: 240,
        flexShrink: 0,
        "& .MuiDrawer-paper": {
          width: 240,
          boxSizing: "border-box",
          position: "relative",
          height: "100%",
          bgcolor: "#1a202c",
          borderRight: "1px solid #2d3748",
        },
      }}
    >
      <Box sx={{ p: 2 }}>
        <Box sx={{ display: "flex", alignItems: "center", gap: 1, mb: 3 }}>
          <Home fontSize="small" sx={{ color: "#a0aec0" }} />
          <Typography variant="body2" fontWeight="medium" color="#e2e8f0">
            7760
          </Typography>
        </Box>

        <Typography variant="body2" color="#718096" sx={{ mb: 1, fontSize: "12px" }}>
          Team
        </Typography>

        <Typography variant="body2" color="#718096" sx={{ mb: 2, fontSize: "12px" }}>
          Lists
        </Typography>

        <List dense disablePadding>
          {sidebarItems.map((item) => (
            <ListItem key={item.id} disablePadding sx={{ py: 0.5 }}>
              <ListItemIcon sx={{ minWidth: 24 }}>
                <Box
                  sx={{
                    width: 16,
                    height: 16,
                    bgcolor: "#2d3748",
                    borderRadius: 0.5,
                    display: "flex",
                    alignItems: "center",
                    justifyContent: "center",
                  }}
                >
                  <Box sx={{ width: 8, height: 8, bgcolor: "#4a5568", borderRadius: 0.5 }} />
                </Box>
              </ListItemIcon>
              <ListItemText
                primary={item.id}
                primaryTypographyProps={{
                  variant: "body2",
                  color: "#e2e8f0",
                  fontSize: "13px",
                }}
              />
              <Box sx={{ width: 64, ml: 1 }}>
                <LinearProgress
                  variant="determinate"
                  value={item.progress}
                  sx={{
                    height: 4,
                    borderRadius: 1,
                    bgcolor: "#2d3748",
                    "& .MuiLinearProgress-bar": {
                      bgcolor: item.color,
                    },
                  }}
                />
              </Box>
            </ListItem>
          ))}
        </List>
      </Box>
    </Drawer>
  )
}
